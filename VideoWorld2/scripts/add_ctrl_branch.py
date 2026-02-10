import torch
import math
pth = torch.load("checkpoints/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt", weights_only=False)
new_names = [('x_embedder2.proj.1.weight', torch.Size([2048, 516])), ('input_hint_block.0.weight', torch.Size([16, 2048])), ('input_hint_block.0.bias', torch.Size([16])), ('input_hint_block.2.weight', torch.Size([16, 16])), ('input_hint_block.2.bias', torch.Size([16])), ('input_hint_block.4.weight', torch.Size([32, 16])), ('input_hint_block.4.bias', torch.Size([32])), ('input_hint_block.6.weight', torch.Size([32, 32])), ('input_hint_block.6.bias', torch.Size([32])), ('input_hint_block.8.weight', torch.Size([96, 32])), ('input_hint_block.8.bias', torch.Size([96])), ('input_hint_block.10.weight', torch.Size([96, 96])), ('input_hint_block.10.bias', torch.Size([96])), ('input_hint_block.12.weight', torch.Size([256, 96])), ('input_hint_block.12.bias', torch.Size([256])), ('input_hint_block.14.weight', torch.Size([2048, 256])), ('input_hint_block.14.bias', torch.Size([2048])), ('zero_blocks.block0.weight', torch.Size([2048, 2048])), ('zero_blocks.block0.bias', torch.Size([2048])), ('zero_blocks.block1.weight', torch.Size([2048, 2048])), ('zero_blocks.block1.bias', torch.Size([2048])), ('zero_blocks.block2.weight', torch.Size([2048, 2048])), ('zero_blocks.block2.bias', torch.Size([2048]))]
for name, shape in new_names:
    _name = 'net.' + name
    # if "x_embedder" in _name:
    #     pth[_name] = pth['net.x_embedder.proj.1.weight']
    if "zero_blocks" in _name:
        pth[_name] = torch.zeros(shape, dtype=pth['net.x_embedder.proj.1.weight'].dtype)
    else:
        # import pdb;pdb.set_trace()
        weight = torch.randn(shape, dtype=pth['net.x_embedder.proj.1.weight'].dtype)
        std = 1.0 / math.sqrt(shape[-1])
        torch.nn.init.trunc_normal_(weight, std=std, a=-3 * std, b=3 * std)
        pth[_name] = weight
torch.save(pth, "checkpoints/Cosmos-Predict2-2B-Video2World/model-480p-16fps-ctrl.pt")
