import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
import torchvision.transforms as T
from torchvision.transforms.functional import scale
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random
from utils import * 

## CUSTOM DATASET TO GET GROUNF TRUTH AND AUGMENTED IMAGES
class NewFenceDataset(Dataset):
    """Train Fence dataset."""

    def __init__(self, coco_img_root, coco_meta_data, fence_list, img_size = (256,256), num_samples=10000, transform=None):
        """
        Args:
            coco_img_root (string): Directory with all the images.
            coco_meta_data (dictionary): coco JSON annotation
            fence_list (list): list of fence names
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.coco_img_root = coco_img_root 
        self.coco_meta_data = coco_meta_data
        self.fence_list = fence_list
        
        self.num_samples = num_samples
        
        self.img_size = img_size

        self.transform = T.Compose([
                                    T.ToTensor(),
#                                     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ])
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.coco_img_root + '/' +random.sample(self.coco_meta_data['images'],1)[0]["file_name"]

        gt_image = Image.open(img_name)

        gt_image = gt_image.convert("RGB")
        
        #resize to 256x256     
        # if (gt_image.size[0] < self.img_size[0]) or (gt_image.size[1] < self.img_size[1]):
        gt_image = gt_image.resize(self.img_size)
        # else:
        #     gt_image = T.RandomCrop(self.img_size)(gt_image)
        
        # augment the gt image
        gt_image = T.ColorJitter(hue=(-0.3,0.3),
                                contrast=0.3, 
                                saturation=0.3, 
                                brightness=0.3)(gt_image)
        # gt_image = T.ColorJitter(brightness=(.8,1))(gt_image)
        # gt_image = T.ColorJitter(contrast=(0.7,1))(gt_image)
        
        # Add the fence to the image
        fenced_image, fence_mask = self.add_fence(gt_image, self.fence_list)
        
        # Transpose
        gt_image = np.asarray(gt_image) #np.transpose(np.asarray(gt_image),(2,0,1))
        fenced_image = np.asarray(fenced_image) #np.transpose(np.asarray(fenced_image),(2,0,1))
        fence_mask = np.asarray(fence_mask)
        
        # Convert to Tensors
        gt_image = self.transform(gt_image)
        fenced_image = self.transform(fenced_image)
        
        sample = {'gt_image': gt_image, 
                  'fenced_image': fenced_image, 
                  'fence_mask:':fence_mask}
        
#         transform = torch.nn.ModuleList([
#             T.RandomAdjustSharpness(sharpness_factor=2),
#             T.RandomEqualize(),
#             T.RandomAutocontrast(),
#             T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ])

#         sample = transform(sample)

        return sample
    
    
    def add_fence(self, gt_img, fence_list):
                
        fence_sample = random.sample(fence_list,1)
        

        # Load the fence mask
        fence_img = Image.open(fence_sample[0][0])
        fence_mask = Image.open(fence_sample[0][1])


        # random crop and resize it
        # state = torch.get_rng_state()         # THIS WAS COMMENTED OUT 3/2/22
        # fence_img = T.RandomCrop(gt_img.size)(fence_img)
        # torch.set_rng_state(state)
        # fence_mask = T.RandomCrop(gt_img.size)(fence_mask)
        fence_img = fence_img.resize(gt_img.size)
        fence_mask = fence_mask.resize(gt_img.size)

        #apply transformations on the fence and mask
        transforms = T.RandomApply(torch.nn.ModuleList([
                    T.RandomRotation((0,45)),
                    T.RandomAffine(scale=(0.5,3),degrees=0),
                    T.RandomAffine(translate=(0.2, 0.2),degrees=0),
                    T.RandomHorizontalFlip(p=0.7),
                    T.RandomVerticalFlip(p=0.7),
                    T.RandomPerspective(distortion_scale=0.2, p=0.7)
        ]), p=0.7)

        
        # Fence transformations
        state = torch.get_rng_state()
        fence_img = transforms(fence_img)
        fence_img = T.GaussianBlur(3)(fence_img)
        fence_img = T.ColorJitter(hue=(-0.5,0.5),contrast=0.5, saturation=0.5, brightness=0.5)(fence_img)
        # Mask transformations
        torch.set_rng_state(state)
        fence_mask = transforms(fence_mask)
        fence_mask = T.GaussianBlur(3)(fence_mask)

        
        # state = torch.get_rng_state()
        generated_image = gt_img.copy()
        # torch.set_rng_state(state)
        generated_image.paste(fence_img,(0,0),mask=fence_mask)
        
        
        return generated_image, fence_mask

    
class TestFenceDataset(Dataset):
    """Test Fence dataset."""

    def __init__(self, test_fence_list, img_size = (256,256), transform=None):
        """
        Args:
            fence_list (list): list of fence names
        """

        self.fence_list = test_fence_list
        
        self.img_size = img_size

        self.transform = T.Compose([
                                    T.ToTensor(),
#                                     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ])
        
    def __len__(self):
        return len(self.fence_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.fence_list[idx]

        gt_image = Image.open(img_name)
                
        gt_image = gt_image.resize(self.img_size)
        
        gt_image = np.asarray(gt_image) #np.transpose(np.asarray(gt_image),(2,0,1))
        
        # Convert to Tensors
        gt_image = self.transform(gt_image)
        
#         sample = {'gt_image': gt_image, 
#                   'fenced_image': gt_image}


        return gt_image