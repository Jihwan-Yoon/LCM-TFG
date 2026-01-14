import torch
import torchvision.transforms.functional as TF

# 대조군을 위한 Dummy Loss
def identity_loss(images):
    return 0.0 * images.sum()

# red 강조
def red_loss(images):
    R, G, B = images[:, 0], images[:, 1], images[:, 2]
    contrast_score = R - (G + B) * 0.5
    return torch.mean(contrast_score)

# green 강조
def green_loss(images):
    R, G, B = images[:, 0], images[:, 1], images[:, 2]
    contrast_score = G - (R + B) * 0.5
    return torch.mean(contrast_score)

# blue 강조
def blue_loss(images):
    R, G, B = images[:, 0], images[:, 1], images[:, 2]
    contrast_score = B - (G + R) * 0.5
    return torch.mean(contrast_score)

# 이미지를 좌우 대칭으로
def symmetry_loss(images):
    images_flipped = torch.flip(images, dims=[-1])
    loss = torch.mean((images - images_flipped) ** 2)
    return -loss

# 이미지를 더 선명하게
def edge_enhancement_loss(images):
    dy = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :])
    dx = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1])
    total_variation = torch.mean(dy) + torch.mean(dx)
    return total_variation

# 가장자리를 어둡게
def vignette_loss(images):
    B, C, H, W = images.shape
    device = images.device
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    dist_sq = grid_y**2 + grid_x**2
    mask = dist_sq.unsqueeze(0).unsqueeze(0)
    brightness = torch.mean(images, dim=1, keepdim=True)
    loss = torch.mean(brightness * mask)
    return -loss

LOSS_DICT = {
    "none": identity_loss,
    "red": red_loss,
    "green": green_loss,
    "blue": blue_loss,
    "symmetry": symmetry_loss,
    "edge": edge_enhancement_loss,
    "center": vignette_loss
}