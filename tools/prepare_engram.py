"""Install the two-rank Engram sum/cast in the pinned prepared model."""
import argparse
from pathlib import Path


def apply(directory):
    path = Path(directory) / 'model.py'
    source = path.read_text()
    before = '        kv = self.comm.sum(mm(lay.engram_wkv, e, F32)).to(BF16)'
    after = ('        from spark1017.engram import sum_bf16\n'
             '        kv = sum_bf16(self.comm, mm(lay.engram_wkv, e, F32))')
    if source.count(before) != 1:
        raise ValueError('Unsupported Engram source; expected one projection reduction')
    path.write_text(source.replace(before, after))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cuda_source', type=Path)
    apply(parser.parse_args().cuda_source)
