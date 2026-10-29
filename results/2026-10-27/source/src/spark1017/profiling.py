"""Opt-in, bounded CUPTI trace. Instrumented runs are excluded from speed claims."""
import json
import os


def begin(decoder):
    path = os.environ.get('SPARK1017_KERNEL_TRACE')
    if not path or decoder.e.rank != 0 or decoder.rounds != 50:
        return
    import torch
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                  torch.profiler.ProfilerActivity.CUDA],
                                      record_shapes=False, profile_memory=False,
                                      with_stack=False)
    profiler.__enter__()
    decoder._spark1017_profiler = profiler


def end(decoder):
    profiler = getattr(decoder, '_spark1017_profiler', None)
    if profiler is None or decoder.rounds != 53:
        return
    profiler.__exit__(None, None, None)
    path = os.environ['SPARK1017_KERNEL_TRACE']
    profiler.export_chrome_trace(path + '.json')
    by_name = {}
    for event in profiler.events():
        if str(event.device_type).endswith('CUDA'):
            row = by_name.setdefault(event.name, {'count': 0, 'microseconds': 0.0})
            row['count'] += 1
            row['microseconds'] += event.time_range.elapsed_us()
    rows = [{'kernel': name, **row} for name, row in by_name.items()]
    rows.sort(key=lambda row: row['microseconds'], reverse=True)
    with open(path + '-summary.json', 'w') as f:
        json.dump({'rounds': 3, 'rank': 0, 'kernels': rows,
                   'timing_note': 'Sum of kernel durations may overlap across CUDA streams.'}, f, indent=2)
    del decoder._spark1017_profiler
