import os
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from PIL import Image
from tools.fit import CycleGAN
import data_loader
from torch.utils import data
from tools.tools import calculate_psnr

def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized images without stacking."""
    return batch

def split_tensor_into_patches(tensor, patch_size=256, overlap=0):
    """
    Splits a tensor of shape (C, H, W) into patches of shape (C, patch_size, patch_size).
    Pads the tensor with reflection if smaller than patch_size.

    Returns:
        patches: List of patch tensors.
        positions: List of (y, x) positions for each patch.
    """
    C, H, W = tensor.shape
    pad_bottom = max(0, patch_size - H)
    pad_right = max(0, patch_size - W)
    if pad_bottom > 0 or pad_right > 0:
        tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode='reflect')
        H, W = tensor.shape[1], tensor.shape[2]
    
    stride = patch_size - overlap
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
    """Returns a list of patches from the tensor, padding if necessary."""
    _, H, W = tensor.shape
    if H < patch_size or W < patch_size:
        pad_bottom = max(0, patch_size - H)
        pad_right = max(0, patch_size - W)
        tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode='reflect')
        return [tensor]
    else:
        patches, _ = split_tensor_into_patches(tensor, patch_size, overlap)
        return patches

def merge_patches(patches, positions, image_shape, patch_size=256, overlap=0, device='cuda'):
    """Merges patches back into a full image, averaging overlapping regions."""
    C, H, W = image_shape
    output = torch.zeros((C, H, W), device=device)
    weight = torch.zeros((1, H, W), device=device)
    for patch, (y, x) in zip(patches, positions):
        output[:, y:y+patch_size, x:x+patch_size] += patch
        weight[:, y:y+patch_size, x:x+patch_size] += 1.0
    output = output / weight.clamp(min=1.0)
    return output

def sliding_window_inference_pair(model, image1, image2, patch_size=256, overlap=32, device='cuda'):
    """
    Processes a pair of images through the model using sliding-window inference.

    Args:
        model: The CycleGAN model.
        image1, image2: Tensors of shape (1, C, H, W).
        patch_size: Size of patches.
        overlap: Overlap between patches.
        device: Device to run inference on.

    Returns:
        Tensor of shape (1, 3, H, W) with the generated RGB image.
    """
    image1 = image1.squeeze(0)
    image2 = image2.squeeze(0)
    _, H, W = image1.shape
    
    patches1, positions = split_tensor_into_patches(image1, patch_size, overlap)
    patches2, _ = split_tensor_into_patches(image2, patch_size, overlap)
    
    processed_patches = []
    model.eval()
    with torch.no_grad():
        for p1, p2 in zip(patches1, patches2):
            p1 = p1.unsqueeze(0).to(device)
            p2 = p2.unsqueeze(0).to(device)
            output_patch = model(p1, p2)
            if isinstance(output_patch, tuple):
                output_patch = output_patch[1]  # Assume index 1 is the RGB output
            processed_patches.append(output_patch.squeeze(0))
    
    merged_output = merge_patches(processed_patches, positions, (3, H, W), patch_size, overlap, device)
    return merged_output.unsqueeze(0)

if __name__ == '__main__':
    # Directory for saving model weights
    checkpoint_dir = './Results_&_weights/weights_exp5_hyperparameter'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # GPU setup
    gpu_ids = ['cuda:0']
    model = CycleGAN(Lconst_penalty=15, gpu_ids=gpu_ids)
    model.setup()
    
    # Directory for validation results
    val_results_dir = './validation_results'
    os.makedirs(val_results_dir, exist_ok=True)
    
    # Load datasets
    train_files, val_files = data_loader.get_data_paths()
    train_dataset = data_loader.Dataset(train_files, target_shape=None)
    # Enable file name return for validation dataset
    val_dataset = data_loader.Dataset(val_files, target_shape=None, return_name=True)
    
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn
    )
    
    # Training parameters
    n_epochs = 10  # Adjust as needed
    patch_size = 256
    overlap = 32
    schedule = 5  # Learning rate update frequency
    
    best_psnr = -1.0
    
    for epoch in range(n_epochs):
        total_d_loss = 0.0
        total_g_loss = 0.0
        total_patch_count = 0
        
        dt_size = len(train_loader.dataset)
        pbar = tqdm.tqdm(total=dt_size, desc=f'Epoch {epoch+1}/{n_epochs}', miniters=1)
        
        model.netG.train()
        for batch in train_loader:
            for sample in batch:
                patches_nir_gray = get_patches(sample['nir_gray'], patch_size, overlap)
                patches_nir_rgb  = get_patches(sample['nir_rgb'], patch_size, overlap)
                patches_nir_hsv  = get_patches(sample['nir_hsv'], patch_size, overlap)
                patches_rgb_gray = get_patches(sample['rgb_gray'], patch_size, overlap)
                patches_rgb_rgb  = get_patches(sample['rgb_rgb'], patch_size, overlap)
                patches_rgb_hsv  = get_patches(sample['rgb_hsv'], patch_size, overlap)
                
                num_patches = len(patches_nir_gray)
                assert num_patches == len(patches_nir_rgb) == len(patches_nir_hsv) == \
                       len(patches_rgb_gray) == len(patches_rgb_rgb) == len(patches_rgb_hsv)
                
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
        
        # Create subdirectory for this epoch's validation images
        epoch_val_dir = os.path.join(val_results_dir, f'epoch_{epoch+1}')
        os.makedirs(epoch_val_dir, exist_ok=True)
        
        # Validation phase
        val_psnr_list = []
        model.netG.eval()
        with torch.no_grad():
            val_pbar = tqdm.tqdm(val_loader, desc=f'Validation Epoch {epoch+1}', leave=False)
            for batch in val_pbar:
                sample = batch[0]
                nir_gray = sample['nir_gray'].unsqueeze(0).to(gpu_ids[0])
                nir_hsv = sample['nir_hsv'].unsqueeze(0).to(gpu_ids[0])
                real_rgb = sample['rgb_rgb'].unsqueeze(0).to(gpu_ids[0])
                
                # Generate fake RGB image
                fake_rgb = sliding_window_inference_pair(model.netG, nir_gray, nir_hsv, patch_size, overlap, gpu_ids[0])
                
                # Calculate PSNR
                real_rgb_np = real_rgb.cpu().numpy()[0].transpose(1, 2, 0)
                fake_rgb_np = fake_rgb.cpu().numpy()[0].transpose(1, 2, 0)
                psnr_val = calculate_psnr(real_rgb_np, fake_rgb_np)
                val_psnr_list.append(psnr_val)
                
                # Save generated image
                fake_rgb_img = (fake_rgb_np * 255).astype(np.uint8)
                base_name = os.path.splitext(os.path.basename(sample['rgb_path']))[0]
                image_filename = os.path.join(epoch_val_dir, f'generated_{base_name}.png')
                Image.fromarray(fake_rgb_img).save(image_filename)
            
            val_pbar.close()
            avg_val_psnr = np.mean(val_psnr_list)
            print(f"Epoch {epoch+1}: Avg Validation PSNR = {avg_val_psnr}")
        
        # Update learning rate
        if (epoch + 1) % schedule == 0:
            model.update_lr()
        
        # Save best model
        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            torch.save(model.netG.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
        
        # Save weights periodically
        if epoch % 50 == 0:
            torch.save(model.netG.state_dict(), os.path.join(checkpoint_dir, f'weights_{epoch}.pth'))