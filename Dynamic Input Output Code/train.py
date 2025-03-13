import os
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from tools.fit import CycleGAN
import data_loader
from torch.utils import data

def custom_collate_fn(batch):
    # Return the list of samples without stacking
    return batch

def split_tensor_into_patches(tensor, patch_size=256, overlap=0):
    """
    Splits a tensor of shape (C, H, W) into patches of shape (C, patch_size, patch_size)
    ensuring full coverage of the image. If the image is smaller than patch_size in any dimension,
    it is padded (using reflection padding) to reach patch_size.
    
    Returns:
        patches: list of patch tensors (each shape (C, patch_size, patch_size))
        positions: list of (y, x) positions of the top-left corner of each patch
    """
    C, H, W = tensor.shape

    # Pad if the image is smaller than patch_size in any dimension.
    pad_bottom = max(0, patch_size - H)
    pad_right = max(0, patch_size - W)
    if pad_bottom > 0 or pad_right > 0:
        tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode='reflect')
        H, W = tensor.shape[1], tensor.shape[2]
    
    stride = patch_size - overlap

    # Calculate starting positions ensuring full coverage.
    y_positions = list(range(0, H - patch_size + 1, stride))
    if not y_positions or y_positions[-1] != H - patch_size:
        y_positions.append(H - patch_size)
    x_positions = list(range(0, W - patch_size + 1, stride))
    if not x_positions or x_positions[-1] != W - patch_size:
        x_positions.append(W - patch_size)
    
    patches = []
    positions = []
    for y in y_positions:
        for x in x_positions:
            patch = tensor[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            positions.append((y, x))
    return patches, positions

def get_patches(tensor, patch_size=256, overlap=0):
    """
    Returns a list of patches from the tensor. If the image is smaller than patch_size,
    the tensor is padded and returned as a single patch.
    """
    _, H, W = tensor.shape
    if H < patch_size or W < patch_size:
        # Pad image if smaller than patch_size.
        pad_bottom = max(0, patch_size - H)
        pad_right = max(0, patch_size - W)
        tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode='reflect')
        return [tensor]
    else:
        patches, _ = split_tensor_into_patches(tensor, patch_size, overlap)
        return patches

if __name__ == '__main__':
    checkpoint_dir = './save_weights_V3'
    os.makedirs(checkpoint_dir, exist_ok=True)
    Lconst_penalty = 15
    batch_size = 16   # General batch size supported.
    n_epochs = 50   # Adjust as needed.
    schedule = 20
    gpu_ids = ['cuda:0']
    
    # Initialize the CycleGAN model (which includes your ColorMamba generator).
    model = CycleGAN(Lconst_penalty=Lconst_penalty, gpu_ids=gpu_ids)
    model.setup()
    model.print_networks(True)
    
    # Load training data (images are loaded at their original sizes).
    train_files, _ = data_loader.get_data_paths()
    train_dataset = data_loader.Dataset(train_files, target_shape=None)
    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=custom_collate_fn  # Use custom collate to avoid stacking variable-size images.
    )
    
    patch_size = 256
    overlap = 32  # You can adjust overlap as desired.
    
    for epoch in range(n_epochs):
        total_d_loss = 0.0
        total_g_loss = 0.0
        total_patch_count = 0
        
        dt_size = len(train_loader.dataset)
        pbar = tqdm.tqdm(total=dt_size, desc=f'Epoch {epoch+1}/{n_epochs}', miniters=1)
        
        model.netG.train()
        for batch in train_loader:
            # 'batch' is a list of sample dictionaries.
            for sample in batch:
                # Each sample has keys: 'nir_gray', 'nir_rgb', 'nir_hsv', 'rgb_gray', 'rgb_rgb', 'rgb_hsv'
                # Print image dimensions and number of patches for debugging.
                H = sample['nir_rgb'].shape[1]
                W = sample['nir_rgb'].shape[2]
                # print(f"Image size: ({H}, {W})")
                
                # For each modality, get patches (if the image is larger than patch_size, it splits; else, one patch is returned).
                patches_nir_gray = get_patches(sample['nir_gray'], patch_size, overlap)
                patches_nir_rgb  = get_patches(sample['nir_rgb'], patch_size, overlap)
                patches_nir_hsv  = get_patches(sample['nir_hsv'], patch_size, overlap)
                patches_rgb_gray = get_patches(sample['rgb_gray'], patch_size, overlap)
                patches_rgb_rgb  = get_patches(sample['rgb_rgb'], patch_size, overlap)
                patches_rgb_hsv  = get_patches(sample['rgb_hsv'], patch_size, overlap)
                
                num_patches = len(patches_nir_gray)
                # print("Number of patches for current image:", num_patches)
                
                # Ensure all modalities produce the same number of patches.
                assert num_patches == len(patches_nir_rgb) == len(patches_nir_hsv) == \
                       len(patches_rgb_gray) == len(patches_rgb_rgb) == len(patches_rgb_hsv), "Mismatch in patch counts."
                
                # Process each patch.
                for idx in range(num_patches):
                    patch_input = {
                        'nir_gray': patches_nir_gray[idx].unsqueeze(0).to(gpu_ids[0]),
                        'nir_rgb':  patches_nir_rgb[idx].unsqueeze(0).to(gpu_ids[0]),
                        'nir_hsv':  patches_nir_hsv[idx].unsqueeze(0).to(gpu_ids[0]),
                        'rgb_gray': patches_rgb_gray[idx].unsqueeze(0).to(gpu_ids[0]),
                        'rgb_rgb':  patches_rgb_rgb[idx].unsqueeze(0).to(gpu_ids[0]),
                        'rgb_hsv':  patches_rgb_hsv[idx].unsqueeze(0).to(gpu_ids[0]),
                    }
                    model.set_input(patch_input)
                    d_loss, g_loss = model.optimize_parameters()
                    total_d_loss += d_loss
                    total_g_loss += g_loss
                    total_patch_count += 1
            pbar.update(len(batch))
        pbar.close()
        
        avg_d_loss = total_d_loss / total_patch_count
        avg_g_loss = total_g_loss / total_patch_count
        print(f"Epoch {epoch+1}: Avg D Loss = {avg_d_loss}, Avg G Loss = {avg_g_loss}")
        
        if (epoch + 1) % schedule == 0:
            model.update_lr()
        if epoch % 50 == 0:
            torch.save(model.netG.state_dict(), os.path.join(checkpoint_dir, f'weights_{epoch}.pth'))



