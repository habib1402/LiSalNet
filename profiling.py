#!/usr/bin/env python3
"""Reprofile LiSalNet without RGB block 18 in FP32.

Times the complete forward returning P1/P3/P5, with inputs already on device.
Image loading, resizing, depth estimation, transfers and saving are excluded.
Conv-only MACs and optional THOP counts are explicitly distinguished.
"""
import argparse
import gc
import importlib.metadata
import io
import json
import platform
from pathlib import Path
import statistics
import time
import torch
from models import LiSalNet
from checkpoint_utils import load_checkpoint


def count_parameters(model):
    parameters = sum(p.numel() for p in model.parameters())
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return {
        'parameters': parameters, 'parameters_M': parameters / 1e6,
        'parameter_payload_MiB': parameter_bytes / 2**20,
        'parameter_and_buffer_payload_MiB': (parameter_bytes + buffer_bytes) / 2**20,
        'by_module': {name: sum(p.numel() for p in child.parameters()) for name, child in model.named_children()},
    }


@torch.inference_mode()
def operation_counts(height, width):
    # A fresh CPU model keeps counting hooks/buffers out of the timed model.
    model = LiSalNet(pretrained=False).cpu().eval()
    rgb, depth = torch.randn(1, 3, height, width), torch.randn(1, 1, height, width)
    convolution_macs = 0
    def count(module, inputs, output):
        nonlocal convolution_macs
        convolution_macs += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
    handles = [module.register_forward_hook(count) for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    try:
        model(rgb, depth)
    finally:
        for handle in handles:
            handle.remove()
    result = {
        'batch_size': 1, 'conv_only_MACs': convolution_macs,
        'conv_only_GMACs': convolution_macs / 1e9,
        'conv_only_GFLOPs_2_per_MAC': 2 * convolution_macs / 1e9,
        'conv_only_excludes': 'bias additions, normalization, pooling, activations, interpolation and elementwise operations',
    }
    try:
        from thop import profile
    except ImportError:
        result['THOP'] = {'available': False}
    else:
        value, _ = profile(model, inputs=(rgb, depth), verbose=False)
        versions = {}
        for distribution in ['thop', 'ultralytics-thop']:
            try:
                versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                pass
        result['THOP'] = {
            'available': True, 'distribution_versions': versions,
            'reported_count': value,
            'reported_GMACs_convention': value / 1e9,
            'reported_GFLOPs_2x_convention': 2 * value / 1e9,
            'note': 'THOP can include normalization/pooling counts while omitting functional operators. State its version and convention; 2x is not an exhaustive scalar-FLOP count.',
        }
    return result


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark(model, height, width, batch_size, runs, warmup, device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    rgb = torch.randn(batch_size, 3, height, width, device=device)
    depth = torch.randn(batch_size, 1, height, width, device=device)
    for _ in range(warmup):
        output = model(rgb, depth)
        del output
    synchronize(device)
    times = []
    for _ in range(runs):
        synchronize(device)
        start = time.perf_counter()
        output = model(rgb, depth)
        synchronize(device)
        times.append(time.perf_counter() - start)
        del output
    synchronize(device)
    start = time.perf_counter()
    for _ in range(runs):
        output = model(rgb, depth)
        del output
    synchronize(device)
    throughput_time = time.perf_counter() - start
    result = {
        'batch_size': batch_size,
        'synchronized_batch_latency_mean_ms': statistics.mean(times) * 1000,
        'synchronized_batch_latency_median_ms': statistics.median(times) * 1000,
        'synchronized_latency_equivalent_FPS': batch_size / statistics.mean(times),
        'paper_FPS_reciprocal_median_latency': batch_size / statistics.median(times),
        'continuous_loop_throughput_FPS': batch_size * runs / throughput_time,
    }
    if device.type == 'cuda':
        # Inputs/model are resident. Reset after warmup/timing, then measure one
        # forward with no previous output retained. Absolute and extra allocated
        # memory have different meanings and are reported separately.
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        output = model(rgb, depth)
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        result['memory_MiB'] = {
            'resident_model_and_inputs_allocated': baseline / 2**20,
            'absolute_peak_allocated': peak / 2**20,
            'extra_peak_above_resident_model_and_inputs': (peak - baseline) / 2**20,
            'peak_reserved': torch.cuda.max_memory_reserved(device) / 2**20,
            'scope': 'PyTorch allocator in this process; not total device usage from nvidia-smi.',
        }
        del output
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--height', type=int, default=320)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1])
    parser.add_argument('--runs', type=int, default=300)
    parser.add_argument('--warmup', type=int, default=60)
    parser.add_argument('--output', type=Path, default=Path('profiling_updated.json'))
    args = parser.parse_args(argv)
    if min(args.height, args.width, args.runs, args.warmup, *args.batch_sizes) <= 0:
        parser.error('Sizes, run counts and batch sizes must be positive.')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable.')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # Exact protocol used for the published LiSalNet row. cuDNN deterministic
    # kernels are requested, while PyTorch's global strict mode remains off.
    torch.use_deterministic_algorithms(False)
    model = LiSalNet(pretrained=False).float().eval()
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
    parameter_info = count_parameters(model)
    payload = io.BytesIO()
    torch.save(model.state_dict(), payload)
    parameter_info['serialized_state_dict_MiB'] = payload.tell() / 2**20
    parameter_info['serialized_state_dict_MB_decimal'] = payload.tell() / 1e6
    payload.close()
    result = {
        'architecture': 'LiSalNet_without_RGB_block_18',
        'checkpoint': str(args.checkpoint) if args.checkpoint else None,
        'weights': 'loaded checkpoint' if args.checkpoint else 'random initialization; efficiency measurement only',
        'python': platform.python_version(), 'torch': str(torch.__version__),
        'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
        'cpu_threads': torch.get_num_threads(), 'precision': 'FP32', 'TF32': False,
        'deterministic_algorithms': False, 'cudnn_deterministic': True,
        'cudnn_benchmark': False,
        'input_height_width': [args.height, args.width],
        'warmup_runs': args.warmup, 'measurement_runs': args.runs,
        'scope': 'Complete model forward with three heads; on-device inputs. Excludes preprocessing, depth estimation, transfers and saving.',
        'parameters': parameter_info,
        'operations': operation_counts(args.height, args.width),
    }
    model = model.to(device)
    result['benchmarks'] = [benchmark(model, args.height, args.width, b, args.runs, args.warmup, device) for b in args.batch_sizes]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
