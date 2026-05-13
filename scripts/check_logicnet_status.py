"""Check LogicNet training status."""
from pathlib import Path
import json

def main():
    # Check LogicNet checkpoints
    checkpoints = [
        'outputs/global_logicnet_smoke_20260508/checkpoints/diffusion/best_logic_model.pth',
        'outputs/full_i_to_vii_qd/checkpoints/best_logic_model.pth',
        'outputs/full_i_to_vii_qd/checkpoints/logic_net_best.pth'
    ]
    
    print('=== LogicNet Training Status ===')
    for ckpt in checkpoints:
        path = Path(ckpt)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f'✅ {ckpt}')
            print(f'   Size: {size_mb:.1f} MB')
            
            # Check meta file
            meta_file = Path(str(path) + '.meta.json')
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                epoch = meta.get('epoch', 'N/A')
                best_metric = meta.get('best_metric', 'N/A')
                print(f'   Epoch: {epoch}')
                print(f'   Best metric: {best_metric}')
        else:
            print(f'❌ {ckpt} - NOT FOUND')
        print()
    
    # Check ablation results
    print('\n=== Ablation Results ===')
    ablations = [
        'results/ablation_logicnet_long/ablation_summary.csv',
        'results/ablation_no_logic_long/ablation_summary.csv'
    ]
    
    for abl in ablations:
        path = Path(abl)
        if path.exists():
            print(f'✅ {abl}')
            with open(path) as f:
                content = f.read()
            print(f'   Content preview:\n{content[:500]}...')
        else:
            print(f'❌ {abl} - NOT FOUND')
        print()

if __name__ == '__main__':
    main()
