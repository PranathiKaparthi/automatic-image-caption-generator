"""AI-Powered Image Caption Generator with Deblurring, High Accuracy, and Fast Inference.

Production Streamlit web application.
"""

import io
import time
from typing import Optional, Tuple
from PIL import Image
import streamlit as st
import torch

from src.blur_detection import BlurAssessmentResult, assess_image_blur
from src.captioning import CaptionGenerator, CaptionResult
from src.config import AppConfig, get_logger
from src.image_validation import ValidationResult, validate_and_normalize_image
from src.restoration import ImageRestorationPipeline, RestorationResult

logger = get_logger("App")

# Page Configuration
st.set_page_config(
    page_title="AI-Powered Image Caption Generator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished, professional UI
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .caption-box {
        background-color: #F8FAFC;
        border-left: 5px solid #3B82F6;
        padding: 1.2rem;
        border-radius: 0.5rem;
        font-size: 1.25rem;
        font-weight: 500;
        color: #0F172A;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }
    .metric-card {
        background-color: #F1F5F9;
        border-radius: 0.5rem;
        padding: 0.8rem;
        text-align: center;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 0.4rem;
    }
    .badge-blue { background-color: #DBEAFE; color: #1E40AF; }
    .badge-green { background-color: #DCFCE7; color: #166534; }
    .badge-amber { background-color: #FEF3C7; color: #92400E; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing Vision-Language Model...")
def get_caption_engine() -> CaptionGenerator:
    """Loads and caches the BLIP captioning model singleton."""
    engine = CaptionGenerator.get_instance()
    engine.load_model()
    return engine


@st.cache_resource(show_spinner="Initializing Restoration Pipeline...")
def get_restoration_pipeline() -> ImageRestorationPipeline:
    """Initializes the genuine optical and deep restoration pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return ImageRestorationPipeline(device=device)


def main():
    config = AppConfig()

    st.markdown('<div class="main-header">🖼️ AI-Powered Image Caption Generator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Intelligent visual scene understanding with optional multi-metric blur assessment and genuine deconvolution.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Sidebar: Pipeline Controls & Settings
    st.sidebar.header("⚙️ Pipeline Configuration")
    pipeline_mode = st.sidebar.radio(
        "Processing Mode",
        options=["Standard Mode", "Enhanced Mode (with Deblurring)"],
        help="Standard mode captions the uploaded image directly. Enhanced mode assesses blur, applies genuine deconvolution, and presents a side-by-side comparison.",
    )

    st.sidebar.subheader("Caption Options")
    caption_style = st.sidebar.selectbox(
        "Caption Style",
        options=["Concise", "Detailed"],
        index=0,
        help="Concise produces direct factual descriptions. Detailed generates richer contextual scene descriptions.",
    )

    num_candidates = st.sidebar.slider(
        "Candidate Captions (Beam Diversity)",
        min_value=1,
        max_value=5,
        value=3,
        help="Generates diverse caption candidates using beam search.",
    )

    use_reranking = st.sidebar.checkbox(
        "Enable Log-Likelihood Reranking",
        value=True,
        help="Scores candidate captions by length-normalized model probability.",
    )

    restoration_method = "adaptive"
    force_restoration = False
    if pipeline_mode == "Enhanced Mode (with Deblurring)":
        st.sidebar.subheader("Deblurring Options")
        restoration_method = st.sidebar.selectbox(
            "Restoration Algorithm",
            options=["adaptive", "richardson_lucy", "wiener", "guided", "deep_nafnet"],
            index=0,
            format_func=lambda x: {
                "adaptive": "Adaptive Selection (Auto-detects blur physics)",
                "richardson_lucy": "Richardson-Lucy Iterative Deconvolution",
                "wiener": "Wiener Frequency-Domain Filter",
                "guided": "Bilateral Edge Reconstruction",
                "deep_nafnet": "PyTorch NAFNet Architecture",
            }.get(x, x),
            help="Selects genuine optical deconvolution or edge-preserving restoration.",
        )
        force_restoration = st.sidebar.checkbox(
            "Force Restoration (even if sharp)",
            value=False,
            help="Apply deblurring even if optical clarity score indicates the image is sharp.",
        )

    # Sidebar Hardware Info
    with st.sidebar.expander("💻 Hardware & Runtime Info", expanded=False):
        device_str = "CUDA GPU" if torch.cuda.is_available() else "CPU"
        st.write(f"**Inference Device:** {device_str}")
        st.write(f"**PyTorch Version:** {torch.__version__}")
        st.write(f"**Caption Model:** `BLIP-base` (247M params)")

    # Main Area: Upload Section
    st.subheader("1. Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, PNG, WebP)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload standard RGB or RGBA photographs up to 15MB.",
    )

    if uploaded_file is None:
        st.info("👆 Upload an image to start visual understanding and caption generation.")
        # Provide sample demonstration info
        st.markdown(
            """
            ---
            ### ✨ Features
            - **Factual Vision Understanding:** Powered by pretrained BLIP vision-language transformer.
            - **Multi-Metric Blur Assessment:** Measures Laplacian second-derivative variance, 2D FFT high-frequency ratio, and Tenengrad gradient energy.
            - **Genuine Deblurring:** Utilizes authentic Richardson-Lucy iterative deconvolution and Wiener frequency filters—no artificial contrast gimmicks.
            - **Side-by-Side Verification:** Compares original vs enhanced image and evaluates both captions.
            """
        )
        return

    # Image Validation
    file_bytes = uploaded_file.read()
    val_result: ValidationResult = validate_and_normalize_image(file_bytes, config.image)

    if not val_result.is_valid:
        st.error(f"❌ Upload Error: {val_result.error_message}")
        return

    if val_result.warning_message:
        st.warning(f"⚠️ {val_result.warning_message}")

    original_pil = val_result.safe_image
    assert original_pil is not None

    # Load Model Singleton
    caption_engine = get_caption_engine()
    restoration_pipeline = get_restoration_pipeline()

    # Layout: Image Display
    st.subheader("2. Image Analysis & Preview")

    col_btn, col_regen = st.columns([1, 1])
    generate_clicked = col_btn.button("🚀 Generate Caption", type="primary", use_container_width=True)
    regen_clicked = col_regen.button("🔄 Regenerate / Diverse Candidates", use_container_width=True)

    if "last_processed" not in st.session_state:
        st.session_state.last_processed = None

    should_process = generate_clicked or regen_clicked or (st.session_state.last_processed == uploaded_file.name)

    if not should_process:
        st.image(original_pil, caption=f"Uploaded: {val_result.original_format} ({val_result.original_size[0]}x{val_result.original_size[1]})", use_container_width=True)
        return

    st.session_state.last_processed = uploaded_file.name

    # Execution Pipeline
    with st.spinner("Processing image through vision pipeline..."):
        t_start_total = time.time()

        if pipeline_mode == "Standard Mode":
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(original_pil, caption="Input Image", use_container_width=True)

            t_cap_start = time.time()
            cap_res: CaptionResult = caption_engine.generate_caption(
                original_pil,
                mode=caption_style.lower(),
                num_candidates=num_candidates,
                rerank=use_reranking,
            )
            t_cap_end = time.time()

            with col2:
                st.markdown("### 📝 Generated Caption")
                if cap_res.error:
                    st.error(cap_res.error)
                else:
                    st.markdown(f'<div class="caption-box">"{cap_res.caption}"</div>', unsafe_allow_html=True)
                    st.write(f"📋 **Copy Caption:** `{cap_res.caption}`")

                    if len(cap_res.candidates) > 1:
                        with st.expander("🔍 Alternative Beam Candidates"):
                            for i, c in enumerate(cap_res.candidates, 1):
                                st.write(f"{i}. {c}")

                # Performance Metrics
                st.markdown("#### ⏱️ Performance Metrics")
                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric("Inference Latency", f"{cap_res.inference_time_s:.2f} s")
                mcol2.metric("Tokens Generated", f"{cap_res.num_tokens}")
                mcol3.metric("Device", cap_res.device.upper())

        else:
            # ENHANCED MODE
            st.markdown("### 🔬 Clarity Assessment & Enhancement")
            
            # 1. Blur Assessment
            assessment: BlurAssessmentResult = assess_image_blur(original_pil, config.blur)
            
            bcol1, bcol2, bcol3, bcol4 = st.columns(4)
            bcol1.metric("Clarity Score", f"{assessment.clarity_score}/100")
            bcol2.metric("Laplacian Var", f"{assessment.laplacian_variance:.1f}")
            bcol3.metric("FFT High-Freq", f"{assessment.fft_high_freq_ratio:.4f}")
            bcol4.metric("Quality State", assessment.blur_level.replace("_", " ").title())

            if assessment.is_blurry:
                st.warning(f"⚠️ {assessment.recommendation}")
            else:
                st.success(f"✅ {assessment.recommendation}")

            # 2. Restoration
            restore_res: RestorationResult = restoration_pipeline.restore_image(
                original_pil,
                force=force_restoration,
                method=restoration_method,
                assessment=assessment,
            )

            # Side-by-Side Images
            col_orig, col_enh = st.columns(2)
            with col_orig:
                st.image(original_pil, caption=f"Original Image (Clarity: {assessment.clarity_score:.1f}/100)", use_container_width=True)

            with col_enh:
                st.image(
                    restore_res.enhanced_image,
                    caption=f"Enhanced Image (Clarity: {restore_res.restored_clarity_score:.1f}/100 via {restore_res.method_used})",
                    use_container_width=True,
                )
                
                # Download Enhanced Image
                buf = io.BytesIO()
                restore_res.enhanced_image.save(buf, format="JPEG", quality=95)
                st.download_button(
                    label="💾 Download Enhanced Image",
                    data=buf.getvalue(),
                    file_name="enhanced_image.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )

            # 3. Dual Captioning (Original vs Enhanced)
            st.markdown("### 📝 Comparative Caption Results")
            c_orig_col, c_enh_col = st.columns(2)

            with c_orig_col:
                st.markdown("**Original Image Caption:**")
                cap_orig = caption_engine.generate_caption(
                    original_pil,
                    mode=caption_style.lower(),
                    num_candidates=num_candidates,
                    rerank=use_reranking,
                )
                if cap_orig.error:
                    st.error(cap_orig.error)
                else:
                    st.markdown(f'<div class="caption-box">"{cap_orig.caption}"</div>', unsafe_allow_html=True)

            with c_enh_col:
                st.markdown("**Enhanced Image Caption:**")
                cap_enh = caption_engine.generate_caption(
                    restore_res.enhanced_image,
                    mode=caption_style.lower(),
                    num_candidates=num_candidates,
                    rerank=use_reranking,
                )
                if cap_enh.error:
                    st.error(cap_enh.error)
                else:
                    st.markdown(f'<div class="caption-box">"{cap_enh.caption}"</div>', unsafe_allow_html=True)

            # Side-by-side performance breakdown
            st.markdown("#### ⏱️ Latency & Efficiency Summary")
            pcol1, pcol2, pcol3, pcol4 = st.columns(4)
            pcol1.metric("Restoration Latency", f"{restore_res.processing_time_s:.2f} s")
            pcol2.metric("Caption Latency", f"{cap_enh.inference_time_s:.2f} s")
            pcol3.metric("Total Processing Time", f"{round(time.time() - t_start_total, 2)} s")
            pcol4.metric("Device", cap_enh.device.upper())

            st.caption(
                "ℹ️ Notice: Restoration removes optical blur and sharpens structural contours. "
                "Compare both generated captions above to observe how visual clarity influences vocabulary and factual scene details."
            )


if __name__ == "__main__":
    main()
