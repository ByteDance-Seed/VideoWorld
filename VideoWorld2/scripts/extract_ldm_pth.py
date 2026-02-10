import torch
import sys

if len(sys.argv) < 3:
    print(f"Usage: python {sys.argv[0]} EXP_NAME iter_name")
    sys.exit(1)

EXP_NAME = sys.argv[1]
iter_name = sys.argv[2]

pth = torch.load(f"checkpoints/posttraining/video2world/{EXP_NAME}/checkpoints/model/{iter_name}.pt")
new_pth = {}
for k, v in pth.items():
    if 'ldm_net' not in k or 'loss' in k:
        continue
    new_pth[k.replace('ldm_net.', '')] = v
torch.save(new_pth, f"checkpoints/posttraining/video2world/{EXP_NAME}/checkpoints/model/{iter_name}_ldm.pt")
import pdb;pdb.set_trace()