"""Keep the pinned NGC CUDA runtime; never let pip replace torch implicitly."""
import importlib.metadata,json,sys
from pathlib import Path
names=('torch','torchvision','triton','numpy')
Path(sys.argv[1]).write_text(''.join(f'{name}=={importlib.metadata.version(name)}\n' for name in names))
print(json.dumps({name:importlib.metadata.version(name) for name in names}))
