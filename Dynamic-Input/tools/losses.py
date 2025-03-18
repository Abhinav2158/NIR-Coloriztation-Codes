
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import torchvision.models as models

# ----------------------------
# Differentiable Histogram Matching Loss Functions
# ----------------------------
def compute_soft_histogram(img, num_bins=256, value_range=(0.0, 1.0), eps=1e-6):
    """
    Computes a differentiable (soft) histogram for each channel.
    
    Args:
        img (Tensor): Input tensor of shape [B, C, H, W].
        num_bins (int): Number of histogram bins.
        value_range (tuple): (min, max) pixel range.
        eps (float): Small constant to avoid division by zero.
    
    Returns:
        Tensor: Normalized soft histogram of shape [B, C, num_bins].
    """
    B, C, H, W = img.shape
    # Create bin centers
    bin_centers = torch.linspace(value_range[0], value_range[1], num_bins, device=img.device)
    bin_centers = bin_centers.view(1, 1, 1, num_bins)  # shape: [1, 1, 1, num_bins]
    
    # Compute bin width
    bin_width = (value_range[1] - value_range[0]) / (num_bins - 1)
    
    # Flatten spatial dimensions
    img_flat = img.view(B, C, -1).unsqueeze(-1)  # shape: [B, C, H*W, 1]
    
    # Soft-assignment to bins (triangular kernel)
    weights = torch.clamp(1 - torch.abs(img_flat - bin_centers) / (bin_width + eps), min=0)
    hist = weights.sum(dim=2)  # shape: [B, C, num_bins]
    hist = hist / (H * W)
    return hist

def histogram_matching_loss(img1, img2, num_bins=256, value_range=(0.0, 1.0)):
    """
    Computes histogram matching loss as the L1 difference between the cumulative histograms
    of two images.
    
    Args:
        img1 (Tensor): Predicted image tensor [B, C, H, W].
        img2 (Tensor): Target image tensor [B, C, H, W].
        num_bins (int): Number of bins.
        value_range (tuple): Pixel range.
        
    Returns:
        Tensor: Scalar loss value.
    """
    hist1 = compute_soft_histogram(img1, num_bins, value_range)
    hist2 = compute_soft_histogram(img2, num_bins, value_range)
    cdf1 = torch.cumsum(hist1, dim=-1)
    cdf2 = torch.cumsum(hist2, dim=-1)
    loss = torch.mean(torch.abs(cdf1 - cdf2))
    return loss

# ----------------------------
# VGG-based Perceptual Loss Module
# ----------------------------
class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_idx=21):
        """
        Uses a pretrained VGG-19 network to extract perceptual features.
        layer_idx=21 (roughly 'relu4_2') is used by default.
        """
        super(VGGPerceptualLoss, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        self.slice = nn.Sequential()
        for i in range(layer_idx):
            self.slice.add_module(str(i), vgg_features[i])
        # Freeze VGG parameters
        for param in self.slice.parameters():
            param.requires_grad = False
        # Normalization buffers (assuming input range [0, 1])
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, input, target):
        # Normalize the images
        input_norm = (input - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        input_features = self.slice(input_norm)
        target_features = self.slice(target_norm)
        return F.mse_loss(input_features, target_features)

# ----------------------------
# Composite Reconstruction Loss Module
# ----------------------------
class ReconstructionLoss(nn.Module):
    def __init__(self, autoencoder, alpha=1.0, beta=1.0, gamma=1.0, delta=0.1):
        """
        Combines the following loss components:
          - Feature-level MSE loss computed on features from a fixed autoencoder.
          - Cosine similarity loss on the same features.
          - Histogram matching loss computed on the raw concatenated images.
          - VGG-based perceptual loss computed on the original prediction and target.
          
        Args:
            autoencoder (nn.Module): A pretrained autoencoder with attributes conv1, conv2, conv3, conv4.
            alpha (float): Weight for feature-based MSE loss.
            beta (float): Weight for the histogram matching loss.
            gamma (float): Weight for cosine similarity loss.
            delta (float): Weight for the VGG perceptual loss.
        """
        super(ReconstructionLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        # Set the autoencoder to evaluation mode and freeze parameters.
        self.autoencoder = autoencoder.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        
        # Use the first few convolution layers from the autoencoder as a feature extractor.
        self.encoder = nn.Sequential(
            self.autoencoder.conv1,
            self.autoencoder.conv2,
            self.autoencoder.conv3,
            self.autoencoder.conv4
        )
        
        # Initialize VGG-based perceptual loss.
        self.vgg_loss_fn = VGGPerceptualLoss(layer_idx=21)
    
    def forward(self, prediction, target):
        # Duplicate the image along channel dimension as done previously.
        # This assumes the autoencoder expects 6 channels (for example, if it was trained on concatenated inputs).
        prediction_cat = torch.cat([prediction, prediction], dim=1)
        target_cat = torch.cat([target, target], dim=1)
        
        # Feature extraction using the fixed encoder.
        feat_pred = self.encoder(prediction_cat)
        feat_target = self.encoder(target_cat)
        
        # Compute feature-level MSE loss.
        mse_loss = F.mse_loss(feat_pred, feat_target)
        # Compute cosine similarity loss (1 - mean cosine similarity).
        cosine_loss = 1.0 - torch.mean(F.cosine_similarity(feat_pred, feat_target, dim=1))
        # Compute histogram matching loss on the concatenated images.
        hist_loss = histogram_matching_loss(prediction_cat, target_cat, num_bins=256, value_range=(0.0, 1.0))
        # Compute VGG perceptual loss on the original prediction and target.
        vgg_loss = self.vgg_loss_fn(prediction, target)
        
        total_loss = (self.alpha * mse_loss +
                      self.gamma * cosine_loss +
                      self.beta * hist_loss +
                      self.delta * vgg_loss)
        return total_loss