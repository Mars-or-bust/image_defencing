import json
import glob
from torch.utils.tensorboard import SummaryWriter

from kornia.losses import ssim_loss
from kornia.losses import psnr_loss

from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse

from utils.networks import *
from utils.DataLoaders import *


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',
                    default=32,
                    dest='batch_size',
                    help='Batch size for training',
                    type=int
                    )
parser.add_argument('--n_workers',
                    default=32,
                    dest='n_workers',
                    help='Number of workers for data loading for training',
                    type=int
                    )
parser.add_argument('--load',
                    action='store_true',
                    dest='load',
                    help='Load a previous save'
                    )
parser.add_argument('-lr',
                    default=1e-3,
                    dest='learning_rate',
                    help='Learning Rate',
                    type=float
                    )
parser.add_argument('--start_epoch',
                    default=1,
                    dest='start_epoch',
                    help='Epoch to resume training from if load is true',
                    type=int
                    )
parser.add_argument('--test_name',
                    default="TEST",
                    dest='test_name',
                    help='Save name for the test',
                    type=str
                    )
parser.add_argument('--features',
                    default=128,
                    dest='features',
                    help='Number of features for the autoencoders',
                    type=int
                    )

args = parser.parse_args()


batch_size = args.batch_size
workers = args.n_workers

num_epochs = 500
learning_rate = args.learning_rate
load = args.load
last_epoch=args.start_epoch
model_dir = "/ocean/projects/iri180005p/bvw546/tmp/fence_models"
log_dir = "/ocean/projects/iri180005p/bvw546/tmp/log"

model_name =  args.test_name #"RED10"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


coco_root = "/ocean/datasets/community/COCO"

coco_annotations = "/Dataset_2017/annotations/instances_train2017.json"
coco_image_folder = "/Dataset_2017/train2017"

fence_path = "/ocean/projects/iri180005p/bvw546/data/De-fencing-master/dataset"
fence_testing = "/Test Set/Test_Images"
fence_training = "/Training Set"


#######################################################################
# # # # # # # # # # # # # # Load the Data # # # # # # # # # # # # # # # 
#######################################################################

# TRAINING LIST
fence_file_list = []
for file in glob.glob(fence_path + fence_training + '/Training_Images/*'):
    fence_file_list.append((file, file.replace("Training_Images","Training_Labels").replace("jpg","png")))
fence_file_list[:1]

# TEST LIST
test_list = []
for file in glob.glob(fence_path + fence_testing + '/*'):
    test_list.append(file)
test_list[:1]

# load in the json data
with open(coco_root + coco_annotations) as f:
  coco_data = json.load(f)

image_size = (128,128)
# Create the dataloader
train_dataloader = torch.utils.data.DataLoader(NewFenceDataset(coco_root + coco_image_folder, 
                                                               coco_data, 
                                                               fence_file_list,
                                                              img_size = image_size),
                                         batch_size=batch_size,
                                         shuffle=False, num_workers=workers)

test_dataloader = torch.utils.data.DataLoader(TestFenceDataset(test_list,
                                                              img_size = image_size),
                                         batch_size=100,
                                         shuffle=False, num_workers=workers)


########################################################################
# # # # # # # # # # # # # # Create the Model # # # # # # # # # # # # # #
########################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = REDNet8(num_features=args.features)
model= nn.DataParallel(model)

if load:
    checkpoint = torch.load(os.path.join(model_dir, model_name + '_epoch-{0:0>3}.pth'.format(last_epoch)))
    model.load_state_dict(checkpoint)
    loss_list = list(np.load(log_dir + '/' + model_name + '_LossList.npy'))
#     intermediate_test_images = list(np.load(log_dir + '/intermediate_imgs.npy'))
else:
    last_epoch=0
    loss_list = []

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#######################################################################
# # # # # # # # # # # # # # Train the Model # # # # # # # # # # # # # #
#######################################################################

writer = SummaryWriter()

# Load in the test data for examples
test_dataset = test_dataloader

if not load:
    for i, data in enumerate(test_dataset):
        np.save(log_dir + '/intermediate_imgs_START', data.detach().cpu().numpy())

# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    loop = tqdm(train_dataloader)
    for i, data in enumerate(loop):
        
        # Send the images to the device
        fenced_img = data['fenced_image'].to(device)
        gt_img = data['gt_image'].to(device)

        output = model(fenced_img)
        
        #calculate the loss
        loss = ssim_loss(gt_img,output,5) + nn.MSELoss()(gt_img,output) #nn.L1Loss(gt_img,output)) ## SSIM and L1 for regularization

        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
        
        # Update the progress bar
        loop.set_description(f"Epoch [{last_epoch+epoch}/{last_epoch+num_epochs}]")
        loop.set_postfix(loss=loss.item())
        
    torch.save(model.state_dict(), os.path.join(model_dir, model_name + '_epoch-{0:0>3}.pth'.format(last_epoch+epoch)))
    
    # Save the loss
    loss_list.append((last_epoch+epoch,loss.item()))
    np.save(log_dir + '/' + model_name + '_LossList.npy', np.array(loss_list))

    with torch.no_grad():
        for i, data in enumerate(test_dataset):
            output = model(data.to(device))
            #intermediate_test_images.append(output.cpu().numpy())
            #save the individual epoch results
            #np.save(log_dir + '/intermediate_imgs', np.array(intermediate_test_images))
            np.save(log_dir + '/' + model_name + '_intermediate_imgs_{0:0>3}'.format(last_epoch+epoch), output.cpu().numpy())
            print('saved!')
            
              
    # Update the Callbacks
    writer.add_scalar('Loss/train', loss.item(), last_epoch+epoch)
#         writer.add_scalar('Loss/test', np.random.random(), n_iter)
        