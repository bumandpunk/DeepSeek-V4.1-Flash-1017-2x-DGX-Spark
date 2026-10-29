"""Install a guarded allocator-budget refresh in the pinned prepared engine."""
import argparse
from pathlib import Path


def apply(directory):
    path = Path(directory) / 'multi.py'
    source = path.read_text()
    before = '''                allocated = torch.cuda.memory_allocated()
                # Graph-private reserved blocks'''
    after = '''                from spark1017.memory_budget import refresh_limit
                refresh_limit(self.e, torch.cuda, available)
                allocated = torch.cuda.memory_allocated()
                # Graph-private reserved blocks'''
    if source.count(before) != 1:
        raise ValueError('Unsupported memory-admission source; expected one guarded boundary')
    path.write_text(source.replace(before, after))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cuda_source', type=Path)
    apply(parser.parse_args().cuda_source)
