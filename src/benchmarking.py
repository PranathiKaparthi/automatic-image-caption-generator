"""Benchmarking suite for latency, cold start, and memory profiling."""

from dataclasses import asdict, dataclass
import json
import os
import platform
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image
import psutil
import torch

from src.blur_detection import assess_image_blur
from src.captioning import CaptionGenerator
from src.config import get_logger
from src.restoration import ImageRestorationPipeline

logger = get_logger("Benchmarking")


@dataclass
class SystemEnvironment:
    """Hardware and runtime platform metadata."""
    os_name: str
    os_version: str
    python_version: str
    processor: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    total_ram_gb: float
    torch_version: str
    cuda_available: bool
    device_used: str


@dataclass
class BenchmarkRunSummary:
    """Statistical summary of benchmark runs."""
    pipeline_name: str
    num_runs: int
    cold_load_time_s: float
    mean_latency_s: float
    median_latency_s: float
    p95_latency_s: float
    min_latency_s: float
    max_latency_s: float
    peak_memory_rss_mb: float
    failure_count: int


def get_current_memory_mb() -> float:
    """Returns current process Resident Set Size (RSS) memory in megabytes."""
    process = psutil.Process(os.getpid())
    return float(process.memory_info().rss / (1024 * 1024))


def gather_system_environment(device: str = "cpu") -> SystemEnvironment:
    """Gathers diagnostic hardware and software environment details."""
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    return SystemEnvironment(
        os_name=platform.system(),
        os_version=platform.version(),
        python_version=platform.python_version(),
        processor=platform.processor() or "AMD/Intel x86_64",
        cpu_cores_physical=psutil.cpu_count(logical=False) or 4,
        cpu_cores_logical=psutil.cpu_count(logical=True) or 8,
        total_ram_gb=round(ram_gb, 2),
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        device_used=device,
    )


class BenchmarkRunner:
    """Executes controlled latency and memory profiling across pipelines."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.caption_generator = CaptionGenerator(device=device)
        self.restoration_pipeline = ImageRestorationPipeline(device=device)

    def benchmark_pipeline(
        self,
        test_images: List[Image.Image],
        pipeline_type: str = "standard",  # 'standard', 'enhanced', or 'baseline'
        num_runs: int = 5,
        num_warmups: int = 1
    ) -> BenchmarkRunSummary:
        """
        Executes benchmark with warmup and repeated trials.
        
        Args:
            test_images: List of PIL test images.
            pipeline_type: 'standard' (caption only) or 'enhanced' (restore + caption).
            num_runs: Number of timed executions per image.
            num_warmups: Warm-up iterations before recording time.
        """
        # Measure cold start load time
        cold_load_time = self.caption_generator.load_model()
        
        # Warmup iterations
        sample_img = test_images[0]
        for _ in range(num_warmups):
            if pipeline_type == "enhanced":
                res = self.restoration_pipeline.restore_image(sample_img, force=True)
                _ = self.caption_generator.generate_caption(res.enhanced_image, mode="concise")
            else:
                _ = self.caption_generator.generate_caption(sample_img, mode="concise")

        latencies: List[float] = []
        failures = 0
        peak_memory = get_current_memory_mb()

        for img in test_images:
            for _ in range(num_runs):
                t0 = time.time()
                try:
                    if pipeline_type == "enhanced":
                        assess = assess_image_blur(img)
                        restored = self.restoration_pipeline.restore_image(img, assessment=assess)
                        _ = self.caption_generator.generate_caption(restored.enhanced_image, mode="concise")
                    else:
                        _ = self.caption_generator.generate_caption(img, mode="concise")
                    
                    elapsed = time.time() - t0
                    latencies.append(elapsed)
                except Exception as e:
                    logger.error(f"Benchmark iteration failure: {e}")
                    failures += 1

                current_mem = get_current_memory_mb()
                if current_mem > peak_memory:
                    peak_memory = current_mem

        lat_arr = np.array(latencies) if latencies else np.array([0.0])
        
        summary = BenchmarkRunSummary(
            pipeline_name=pipeline_type,
            num_runs=len(latencies),
            cold_load_time_s=round(cold_load_time, 3),
            mean_latency_s=round(float(np.mean(lat_arr)), 3),
            median_latency_s=round(float(np.median(lat_arr)), 3),
            p95_latency_s=round(float(np.percentile(lat_arr, 95)), 3),
            min_latency_s=round(float(np.min(lat_arr)), 3),
            max_latency_s=round(float(np.max(lat_arr)), 3),
            peak_memory_rss_mb=round(peak_memory, 1),
            failure_count=failures,
        )
        return summary


def calculate_improvement_percentage(baseline: float, current: float) -> float:
    """
    Computes percentage reduction/improvement using standard formula:
    Improvement (%) = ((Baseline - Current) / Baseline) * 100
    Positive indicates reduction (faster or less memory).
    Negative indicates increase.
    """
    if baseline == 0:
        return 0.0
    return round(((baseline - current) / baseline) * 100.0, 2)


def save_benchmark_report(
    env: SystemEnvironment,
    results: List[BenchmarkRunSummary],
    output_dir: str = "reports"
) -> str:
    """Saves benchmark results to formatted JSON and CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    report_data: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_environment": asdict(env),
        "results": [asdict(r) for r in results],
    }

    # Save JSON
    json_path = os.path.join(output_dir, "benchmark_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Save CSV
    csv_path = os.path.join(output_dir, "benchmark_report.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Pipeline,Runs,ColdLoadTime_s,MeanLatency_s,MedianLatency_s,P95Latency_s,PeakMemory_MB,Failures\n")
        for r in results:
            f.write(
                f"{r.pipeline_name},{r.num_runs},{r.cold_load_time_s},"
                f"{r.mean_latency_s},{r.median_latency_s},{r.p95_latency_s},"
                f"{r.peak_memory_rss_mb},{r.failure_count}\n"
            )

    logger.info(f"Saved benchmark reports to '{json_path}' and '{csv_path}'.")
    return json_path
