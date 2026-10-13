"""Benchmarking script measuring latency, memory usage, and throughput."""

import os
import sys
from PIL import Image

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.benchmarking import (
    BenchmarkRunner,
    calculate_improvement_percentage,
    gather_system_environment,
    save_benchmark_report,
)


def run_benchmarks(data_dir: str = "data/images", output_dir: str = "reports"):
    """Runs repeatable benchmark measuring cold-start, warm latency, and peak memory."""
    print("=" * 60)
    print("STARTING LATENCY AND RESOURCE CONSUMPTION BENCHMARK")
    print("=" * 60)

    env = gather_system_environment()
    print(f"Platform: {env.os_name} {env.os_version}")
    print(f"Processor: {env.processor} ({env.cpu_cores_physical} physical, {env.cpu_cores_logical} logical cores)")
    print(f"Total System RAM: {env.total_ram_gb} GB")
    print(f"PyTorch Version: {env.torch_version} | CUDA Available: {env.cuda_available}")
    print(f"Device: {env.device_used.upper()}\n")

    # Load benchmark images
    image_files = [
        "scene_01_park_clean.png",
        "scene_02_car_clean.png",
        "scene_03_desk_clean.png",
    ]
    test_images = []
    for f in image_files:
        p = os.path.join(data_dir, f)
        if os.path.exists(p):
            test_images.append(Image.open(p))

    if not test_images:
        print(f"No test images found in {data_dir}. Generating fallback image...")
        test_images = [Image.new("RGB", (320, 240), color=(120, 150, 180))]

    runner = BenchmarkRunner(device=env.device_used)

    # 1. Benchmark Standard Captioning Pipeline
    print("--> Benchmarking Standard Pipeline (BLIP Captioning)...")
    standard_summary = runner.benchmark_pipeline(
        test_images=test_images,
        pipeline_type="standard",
        num_runs=3,
        num_warmups=1,
    )
    print(f"    Cold Start Load Time: {standard_summary.cold_load_time_s:.2f} s")
    print(f"    Mean Warm Latency:    {standard_summary.mean_latency_s:.2f} s")
    print(f"    Median Latency:       {standard_summary.median_latency_s:.2f} s")
    print(f"    P95 Latency:          {standard_summary.p95_latency_s:.2f} s")
    print(f"    Peak RSS Memory:      {standard_summary.peak_memory_rss_mb:.1f} MB")

    # 2. Benchmark Enhanced Pipeline (Deblurring + Captioning)
    print("\n--> Benchmarking Enhanced Pipeline (Deblur + Captioning)...")
    enhanced_summary = runner.benchmark_pipeline(
        test_images=test_images,
        pipeline_type="enhanced",
        num_runs=3,
        num_warmups=1,
    )
    print(f"    Mean Latency:    {enhanced_summary.mean_latency_s:.2f} s")
    print(f"    Median Latency:  {enhanced_summary.median_latency_s:.2f} s")
    print(f"    P95 Latency:     {enhanced_summary.p95_latency_s:.2f} s")
    print(f"    Peak RSS Memory: {enhanced_summary.peak_memory_rss_mb:.1f} MB")

    # Save reports
    report_file = save_benchmark_report(
        env=env,
        results=[standard_summary, enhanced_summary],
        output_dir=output_dir,
    )
    print(f"\nBenchmark results saved successfully to '{report_file}'.")
    return report_file


if __name__ == "__main__":
    run_benchmarks()
