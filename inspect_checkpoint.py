import torch

checkpoint = torch.load("my_checkpoint.pth.tar", map_location="cpu")
print(checkpoint.keys())
# for k, v in checkpoint['state_dict'].items():
    # print(f"{k}: {v.shape}")
print(checkpoint['optimizer'])