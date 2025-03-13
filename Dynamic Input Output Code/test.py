import os
import numpy as np
import torch
import tqdm
from PIL import Image
import data_loader
from torch.utils import data
import torch.nn.functional as F
from tools.tools import calculate_psnr, calculate_ssim, calculate_ae
from tools.MS_SWD import MS_SWD
from models import CycleGanNIR_net

def create_feathering_mask(patch_size, overlap):
    """
    Creates a 2D feathering mask of shape (patch_size, patch_size).
    The mask is ~1 in the central region and smoothly decays to 0 near edges.
    'overlap' controls how wide the feather region is.
    """
    # We'll create a 1D ramp from 0..1 across the overlap region,
    # then 1 in the flat center, then 1..0 ramp at the far side.
    if overlap <= 0:
        # No feathering if overlap=0
        return torch.ones(patch_size, patch_size)
    
    ramp = torch.linspace(0, 1, steps=overlap)
    # The center region (patch_size - 2*overlap) is fully 1
    center_length = patch_size - 2*overlap
    if center_length < 0:
        # Overlap is so large it exceeds half the patch.
        # We'll clamp it so we at least have a monotonic ramp.
        center_length = 0
    flat_center = torch.ones(center_length)
    mask_1d = torch.cat([ramp, flat_center, torch.flip(ramp, dims=[0])])
    
    # Outer product to create a 2D mask from the 1D ramp
    mask_2d = torch.ger(mask_1d, mask_1d)  # shape: (patch_size, patch_size)
    return mask_2d

def split_tensor_into_patches(tensor, patch_size=256, overlap=0):
    """
    Splits a tensor of shape (C, H, W) into patches of shape (C, patch_size, patch_size),
    ensuring full coverage of the image. If the image is smaller than patch_size in any dimension,
    it is padded (using reflection padding) to reach patch_size.
    """
    C, H, W = tensor.shape
    # Pad if image is smaller than patch_size
    pad_bottom = max(0, patch_size - H)
    pad_right = max(0, patch_size - W)
    if pad_bottom > 0 or pad_right > 0:
        tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode='reflect')
        H, W = tensor.shape[1], tensor.shape[2]
    
    stride = patch_size - overlap
    y_positions = list(range(0, H - patch_size + 1, stride))
    if y_positions[-1] != H - patch_size:
        y_positions.append(H - patch_size)
    x_positions = list(range(0, W - patch_size + 1, stride))
    if x_positions[-1] != W - patch_size:
        x_positions.append(W - patch_size)
    
    patches = []
    positions = []
    for y in y_positions:
        for x in x_positions:
            patch = tensor[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            positions.append((y, x))
    return patches, positions

def merge_patches_feathered(patches, positions, image_shape, patch_size=256, overlap=0, device='cuda'):
    """
    Merges a list of patch tensors (each of shape (C, patch_size, patch_size)) back into a full image
    of shape (C, H, W), using a feathering mask to smoothly blend overlapping regions.
    """
    C, H, W = image_shape
    output = torch.zeros((C, H, W), device=device)
    weight = torch.zeros((1, H, W), device=device)
    
    # Create the feathering mask (shape: (patch_size, patch_size))
    feather_2d = create_feathering_mask(patch_size, overlap).to(device)
    # Expand to (1, patch_size, patch_size) so we can multiply each patch easily
    feather_mask = feather_2d.unsqueeze(0)  # shape (1, patch_size, patch_size)
    
    for patch, (y, x) in zip(patches, positions):
        # patch shape: (C, patch_size, patch_size)
        # Multiply patch by the feather mask across channels
        weighted_patch = patch * feather_mask  # broadcasting feather_mask to shape (C, patch_size, patch_size)
        output[:, y:y+patch_size, x:x+patch_size] += weighted_patch
        weight[:, y:y+patch_size, x:x+patch_size] += feather_mask
    
    output = output / weight.clamp(min=1e-8)
    return output

def sliding_window_inference_pair(model, image1, image2, patch_size=256, overlap=0, device='cuda'):
    """
    Processes a pair of images (nir_gray, nir_hsv) via sliding-window inference.
    Both image1 and image2 are expected to be tensors of shape (1, C, H, W).
    The function splits the images into patches, processes each patch pair through the model,
    and then merges the output patches using feathered blending to produce a final image
    of the same size as the input.
    
    Returns:
        A tensor of shape (1, 3, H, W) containing the reconstructed RGB image.
    """
    # Remove batch dimension
    image1 = image1.squeeze(0)
    image2 = image2.squeeze(0)
    _, H, W = image1.shape
    
    # Split into patches
    patches1, positions = split_tensor_into_patches(image1, patch_size, overlap)
    patches2, _ = split_tensor_into_patches(image2, patch_size, overlap)
    
    processed_patches = []
    model.eval()
    with torch.no_grad():
        for p1, p2 in zip(patches1, patches2):
            p1 = p1.unsqueeze(0).to(device)
            p2 = p2.unsqueeze(0).to(device)
            output_patch = model(p1, p2)
            # If the model returns multiple outputs, the second is the final RGB
            if isinstance(output_patch, tuple):
                output_patch = output_patch[1]
            output_patch = output_patch.squeeze(0)  # shape: (3, patch_size, patch_size)
            processed_patches.append(output_patch)
    
    # Feathered merging
    merged_output = merge_patches_feathered(
        processed_patches, positions, (3, H, W),
        patch_size, overlap, device
    )
    return merged_output.unsqueeze(0)

if __name__ == '__main__':
    checkpoint_path = './save_weights_V3/weights_0.pth'
    os.makedirs('./results', exist_ok=True)
    device = 'cuda'
    patch_size = 256
    overlap = 64  # Larger overlap => better blending, but more repeated computation.
    
    # Load the generator model (ColorMamba network).
    model = CycleGanNIR_net.all_Generator(3, 3).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Load test data (original sizes).
    test_files = data_loader.get_test_paths()
    test_dataset = data_loader.Dataset_test(test_files, target_shape=None)
    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # MS-SWD for texture evaluation
    ms_swd_model = MS_SWD(num_scale=5, num_proj=128).to(device)
    
    psnr_list, ssim_list, ae_list, ms_swd_list = [], [], [], []
    
    for i, batch in enumerate(tqdm.tqdm(test_loader, desc="Testing")):
        nir_gray = batch['nir_gray'].to(device)
        nir_hsv = batch['nir_hsv'].to(device)
        real_rgb = batch['rgb_rgb'].to(device)
        
        # Sliding-window inference with feathered merging
        fake_rgb = sliding_window_inference_pair(model, nir_gray, nir_hsv, patch_size, overlap, device)
        
        # Convert to NumPy for metrics
        real_rgb_np = real_rgb.cpu().numpy()[0].transpose(1, 2, 0)
        fake_rgb_np = fake_rgb.cpu().numpy()[0].transpose(1, 2, 0)
        
        psnr_val = calculate_psnr(real_rgb_np, fake_rgb_np)
        ssim_val = calculate_ssim(real_rgb_np, fake_rgb_np)
        ae_val   = calculate_ae(real_rgb_np, fake_rgb_np)
        ms_swd_val = ms_swd_model(fake_rgb, real_rgb).item()
        
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        ae_list.append(ae_val)
        ms_swd_list.append(ms_swd_val)
        
        # Save the reconstructed output
        out_img = (fake_rgb_np * 255).astype(np.uint8)
        image_filename = os.path.join('./results', f'result_{i+1}.png')
        Image.fromarray(out_img).save(image_filename)
    
    print("Average PSNR:", np.mean(psnr_list))
    print("Average SSIM:", np.mean(ssim_list))
    print("Average AE:", np.mean(ae_list))
    print("Average MS-SWD:", np.mean(ms_swd_list))
    
    with open('best_test.txt', 'w') as f:
        f.write("Average PSNR: %f\n" % np.mean(psnr_list))
        f.write("Average SSIM: %f\n" % np.mean(ssim_list))
        f.write("Average AE: %f\n" % np.mean(ae_list))
        f.write("Average MS-SWD: %f\n" % np.mean(ms_swd_list))
