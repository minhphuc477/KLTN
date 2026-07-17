import sys, pathlib, platform
print('Python:', sys.version)
print('Platform:', platform.platform())

for pkg in ['torch', 'numpy', 'networkx', 'scipy', 'torch_geometric', 'accelerate', 'yaml']:
    try:
        m = __import__(pkg)
        ver = getattr(m, '__version__', 'ok')
        print(f'  {pkg}: {ver}')
    except ImportError as e:
        print(f'  {pkg}: MISSING - {e}')

try:
    import torch
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('CUDA device:', torch.cuda.get_device_name(0))
        mem = torch.cuda.get_device_properties(0).total_memory // 1024**3
        print('CUDA memory:', mem, 'GB')
except Exception as e:
    print('CUDA check error:', e)

# Check data dir
data_dir = pathlib.Path('Data/The Legend of Zelda')
print('Data dir exists:', data_dir.exists())
if data_dir.exists():
    files = list(data_dir.rglob('*'))
    print('  Files:', len([f for f in files if f.is_file()]))
    exts = set(f.suffix for f in files if f.is_file())
    print('  Extensions:', sorted(exts))

# Check config
cfg = pathlib.Path('configs/zelda_hmolqd.yaml')
print('Config exists:', cfg.exists())

# Check main.py train subcommand
print('main.py exists:', pathlib.Path('main.py').exists())
