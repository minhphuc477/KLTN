import json
import numpy as np
p = 'artifacts/visual_extracts_demo/tloz1_1.npz'
print('Loading', p)
d = np.load(p, allow_pickle=True)
print('files =', d.files)
grid = d['grid']
print('grid.shape =', grid.shape)
meta = json.loads(str(d['metadata']))
print('metadata =', meta)
ids = grid[:,:,0].astype(int)
conf = grid[:,:,1]
print('\nids rows 6..10:')
for r in range(6,11):
    print(ids[r].tolist())
print('\nconf mean =', float(conf.mean()))
