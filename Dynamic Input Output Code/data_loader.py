import glob
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import torch
from torch.utils import data

def get_data_paths():
    path_train_nir = './datasets/NIR/train'
    path_train_rgb = './datasets/RGB/train'
    path_val = './datasets/val'
    nir_files = glob.glob(os.path.join(path_train_nir, '*.png'))
    nir_files.sort()
    nir_files_val = glob.glob(os.path.join(path_val, 'nir', '*_nir.png'))
    nir_files_val.sort()
    # rgb_files = glob.glob(os.path.join(path_train_rgb, '*.png'))
    rgb_files_png = sorted(glob.glob(os.path.join(path_train_rgb, '*.png')))
    rgb_files_jpg = sorted(glob.glob(os.path.join(path_train_rgb, '*.jpg')))

# Merge the lists and sort by filename (or any custom key)
    rgb_files = sorted(rgb_files_png + rgb_files_jpg)
    rgb_files.sort()
    rgb_files_val = glob.glob(os.path.join(path_val, 'rgb', '*_rgb_reg.png'))
    rgb_files_val.sort()
    print("Number of NIR files:", len(nir_files))
    print("Number of RGB files:", len(rgb_files))
    
    train_files = np.stack([nir_files, rgb_files], axis=1)
    val_files = np.stack([nir_files_val, rgb_files_val], axis=1)
    return train_files, val_files

def get_test_paths():
    path_val = './datasets/Testing'
    nir_files_png = glob.glob(os.path.join(path_val, '*_nir_reg.png'))
    nir_files_jpg = sorted(glob.glob(os.path.join(path_val, 'NIR*.png')))
    nir_files_val = sorted(nir_files_png + nir_files_jpg)
    nir_files_val.sort()
    # rgb_files_val = glob.glob(os.path.join(path_val, '*_rgb_reg.png'))
    rgb_files_png = sorted(glob.glob(os.path.join(path_val, '*_rgb_reg.png')))
    rgb_files_jpg = sorted(glob.glob(os.path.join(path_val, 'RGBR*.jpg')))
    rgb_files_val = sorted(rgb_files_png + rgb_files_jpg)
    rgb_files_val.sort()
    val_files = np.stack([nir_files_val, rgb_files_val], axis=1)
    return val_files

def to_hsv(img):
    hsv_images = []
    for img_ in img:
        hsv_image = cv2.cvtColor(img_, cv2.COLOR_RGB2HSV)
        hsv_images.append(hsv_image)
    hsv_images = np.stack(hsv_images, axis=0)
    return hsv_images

def nor(img):
    img = img.astype(np.float32)
    img -= np.min(img)
    img = img / (np.max(img) + 1e-3)
    return img

def randomCrop(img, crop_width, crop_height):
    # img is a PIL Image
    w, h = img.size
    if w < crop_width or h < crop_height:
        raise ValueError("Image size is smaller than crop size")
    x = random.randint(0, w - crop_width)
    y = random.randint(0, h - crop_height)
    cropped_img = img.crop((x, y, x + crop_width, y + crop_height))
    return cropped_img, (x, y, x + crop_width, y + crop_height)

def crop_resize(img, position, resize_size):
    # Crop the image using the given position and then resize it to resize_size.
    img = img.crop(position)
    img = img.resize(resize_size, Image.BICUBIC)
    return img

class Dataset(data.Dataset):
    def __init__(self, files, target_shape=None, return_name=False):
        """
        files: numpy array with shape (N, 2) where column 0 is the NIR image path and column 1 is the RGB image path.
        target_shape: tuple (width, height) to which the images will be resized.
                      If None, the original image dimensions are preserved.
        return_name: whether to return file names along with the data.
        """
        self.files = files
        self.return_name = return_name
        self.target_shape = target_shape

    def __len__(self):
        return len(self.files)

    def read_data(self, img_path):
        # Load the image in RGB and grayscale
        img = Image.open(img_path).convert('RGB')
        img_gray = Image.open(img_path).convert('L')
        
        if self.target_shape is not None:
            # Optionally perform a random crop (e.g., a 200x200 region) and then resize to target_shape.
            crop_width, crop_height = 200, 200
            img_cropped, position = randomCrop(img, crop_width, crop_height)
            img = img_cropped.resize(self.target_shape, Image.BICUBIC)
            img_gray = img_gray.crop(position)
            img_gray = img_gray.resize(self.target_shape, Image.BICUBIC)
        else:
            position = None

        # Convert images to numpy arrays and normalize them to [0, 1]
        img_rgb = np.array(img)
        img_rgb = nor(img_rgb)
        img_gray = np.array(img_gray)
        img_gray = nor(img_gray)
        img_hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
        img_hsv = nor(img_hsv)

        # Convert to torch tensors (channel-first ordering)
        img_rgb = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float()
        img_hsv = torch.from_numpy(img_hsv.transpose(2, 0, 1)).float()
        img_gray = torch.from_numpy(img_gray).unsqueeze(0).float()
        
        return img_gray, img_rgb, img_hsv, position

    def __getitem__(self, index):
        nir_path = self.files[index][0]
        rgb_path = self.files[index][1]
        nir_gray, nir_rgb, nir_hsv, _ = self.read_data(nir_path)
        rgb_gray, rgb_rgb, rgb_hsv, _ = self.read_data(rgb_path)
        sample = {
            'nir_gray': nir_gray,
            'nir_rgb': nir_rgb,
            'nir_hsv': nir_hsv,
            'rgb_gray': rgb_gray,
            'rgb_rgb': rgb_rgb,
            'rgb_hsv': rgb_hsv
        }
        if self.return_name:
            sample['nir_path'] = nir_path
            sample['rgb_path'] = rgb_path
        return sample

class Dataset_test(data.Dataset):
    def __init__(self, files, target_shape=None, return_name=True):
        """
        For the test dataset, target_shape controls resizing.
        If target_shape is None, the original dimensions are maintained.
        """
        self.files = files
        self.return_name = return_name
        self.target_shape = target_shape

    def __len__(self):
        return len(self.files)

    def read_data(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img_gray = Image.open(img_path).convert('L')
        if self.target_shape is not None:
            img = img.resize(self.target_shape, Image.BICUBIC)
            img_gray = img_gray.resize(self.target_shape, Image.BICUBIC)
        position = None

        img_rgb = np.array(img)
        img_rgb = nor(img_rgb)
        img_gray = np.array(img_gray)
        img_gray = nor(img_gray)
        img_hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
        img_hsv = nor(img_hsv)

        img_rgb = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float()
        img_hsv = torch.from_numpy(img_hsv.transpose(2, 0, 1)).float()
        img_gray = torch.from_numpy(img_gray).unsqueeze(0).float()

        return img_gray, img_rgb, img_hsv, position

    def __getitem__(self, index):
        nir_path = self.files[index][0]
        rgb_path = self.files[index][1]
        nir_gray, nir_rgb, nir_hsv, _ = self.read_data(nir_path)
        rgb_gray, rgb_rgb, rgb_hsv, _ = self.read_data(rgb_path)
        sample = {
            'nir_gray': nir_gray,
            'nir_rgb': nir_rgb,
            'nir_hsv': nir_hsv,
            'rgb_gray': rgb_gray,
            'rgb_rgb': rgb_rgb,
            'rgb_hsv': rgb_hsv,
            'rgb_path': rgb_path
        }
        return sample

if __name__ == '__main__':
    # Quick test: load the training data without resizing (i.e. preserve original dimensions)
    train_files, _ = get_data_paths()
    dataset = Dataset(train_files, target_shape=None)
    sample = dataset[0]
    print("NIR image shape:", sample['nir_rgb'].shape)
    print("RGB image shape:", sample['rgb_rgb'].shape)
