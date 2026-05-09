import os
# Forces DGL to use the pure Python backend and avoid the problematic C++ library load
os.environ['DGLBACKEND'] = 'pytorch'
os.environ['DGL_FORCE_DEVICE'] = 'cuda:0' 

import torch
import dgl
import torch_geometric

print(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")
print(f"DGL Device: {dgl.tensor([1.0], device='cuda:0').device}")
print(f"PyG works: {torch_geometric.is_debug_enabled() == False}")

# Final Blackwell Test
x = torch.randn(10, 10).cuda()
print("Blackwell Matrix Op Success!")
