#!/usr/bin/env python3
"""
OpenVINO Multi-GPU Benchmark Script
Benchmarks a model on all available GPUs in parallel and displays FPS for each device.
"""

import argparse
import subprocess
import re
import openvino as ov
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional


def get_available_gpus() -> List[str]:
    """
    Get list of available GPU devices from OpenVINO.
    
    Returns:
        List of GPU device names (e.g., ['GPU.0', 'GPU.1'])
    """
    core = ov.Core()
    available_devices = core.available_devices
    
    # Filter only GPU devices
    gpu_devices = [device for device in available_devices if device.startswith('GPU')]
    
    return gpu_devices


def parse_fps_from_output(output: str) -> Optional[float]:
    """
    Parse FPS value from benchmark_app output.
    
    Args:
        output: String output from benchmark_app
        
    Returns:
        FPS value as float, or None if not found
    """
    # Look for pattern: "[ INFO ] Throughput:   208.36 FPS"
    pattern = r'\[\s*INFO\s*\]\s*Throughput:\s*([\d.]+)\s*FPS'
    match = re.search(pattern, output)
    
    if match:
        return float(match.group(1))
    return None


def benchmark_gpu(model_path: str, device: str, duration: int = 30) -> Tuple[str, Optional[float], str]:
    """
    Run benchmark on a specific GPU device.
    
    Args:
        model_path: Path to the model XML file
        device: Device name (e.g., 'GPU.0')
        duration: Benchmark duration in seconds
        
    Returns:
        Tuple of (device_name, fps, error_message)
    """
    cmd = [
        'benchmark_app',
        '-m', model_path,
        '-d', device,
        '-t', str(duration)
    ]
    
    try:
        print(f"[ INFO ] Starting benchmark on {device}...")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=duration + 60  # Add buffer time for initialization
        )
        
        # Combine stdout and stderr as benchmark_app writes to both
        output = result.stdout + result.stderr
        
        # Parse FPS from output
        fps = parse_fps_from_output(output)
        
        if fps is None:
            return device, None, "Failed to parse FPS from output"
        
        return device, fps, ""
        
    except subprocess.TimeoutExpired:
        return device, None, "Benchmark timed out"
    except FileNotFoundError:
        return device, None, "benchmark_app not found in PATH"
    except Exception as e:
        return device, None, f"Error: {str(e)}"


def run_parallel_benchmarks(model_path: str, gpu_devices: List[str], duration: int = 30) -> None:
    """
    Run benchmarks on all GPUs in parallel.
    
    Args:
        model_path: Path to the model XML file
        gpu_devices: List of GPU device names
        duration: Benchmark duration in seconds
    """
    print(f"\n[ INFO ] Starting parallel benchmarks on {len(gpu_devices)} GPU(s)")
    print(f"[ INFO ] Model: {model_path}")
    print(f"[ INFO ] Duration: {duration} seconds per GPU\n")
    
    # Run benchmarks in parallel using ThreadPoolExecutor
    results = {}
    
    with ThreadPoolExecutor(max_workers=len(gpu_devices)) as executor:
        # Submit all benchmark tasks
        future_to_device = {
            executor.submit(benchmark_gpu, model_path, device, duration): device
            for device in gpu_devices
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_device):
            device, fps, error = future.result()
            results[device] = (fps, error)
            
            if fps is not None:
                print(f"[ COMPLETED ] {device}: {fps:.2f} FPS")
            else:
                print(f"[ FAILED ] {device}: {error}")
    
    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    
    # Print summary
    print(f"\n{'-'*62}")
    
    successful_results = [(dev, fps) for dev, (fps, err) in results.items() if fps is not None]
    failed_results = [(dev, err) for dev, (fps, err) in results.items() if fps is None]
    
    if successful_results:
        # Sort by device name for consistent output
        successful_results.sort(key=lambda x: x[0])
        
        # Print model name in green
        print(f"{GREEN}\t Model: {model_path}{RESET}")
        
        # Print each GPU result in green
        for device, fps in successful_results:
            print(f"{GREEN}\t {device}: {fps:.2f} FPS{RESET}")
        
        # Calculate total throughput
        if len(successful_results) > 1:
            total_fps = sum(fps for _, fps in successful_results)
            print(f"{GREEN}\t Total: {total_fps:.2f} FPS{RESET}")
    
    if failed_results:
        print(f"\n{RED}[ FAILED ] Devices:{RESET}")
        for device, error in failed_results:
            print(f"{RED}\t {device}: {error}{RESET}")
    
    print(f"{'-'*62}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark OpenVINO model on all available GPUs in parallel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python benchmark.py -m /opt/models/yolo11s/FP16/yolo11s.xml
  python benchmark.py -m model.xml -t 60
        '''
    )
    
    parser.add_argument(
        '-m', '--model',
        required=True,
        help='Path to the model XML file'
    )
    
    parser.add_argument(
        '-t', '--duration',
        type=int,
        default=30,
        help='Benchmark duration in seconds (default: 30)'
    )
    
    args = parser.parse_args()
    
    # Get available GPU devices
    print("[ INFO ] Detecting available GPU devices...")
    gpu_devices = get_available_gpus()
    
    if not gpu_devices:
        print("[ ERROR ] No GPU devices found!")
        print("[ INFO ] Available devices:")
        core = ov.Core()
        for device in core.available_devices:
            print(f"  - {device}")
        return 1
    
    device_word = "device" if len(gpu_devices) == 1 else "devices"
    print(f"[ INFO ] Found {len(gpu_devices)} GPU {device_word}:")
    for device in gpu_devices:
        print(f"  - {device}")
    
    # Run parallel benchmarks
    run_parallel_benchmarks(args.model, gpu_devices, args.duration)
    
    return 0


if __name__ == '__main__':
    exit(main())
