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
parser.add_argument('--lr',
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
                    default="TEST_ADV",
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
parser.add_argument('--num_samples',
                    default=10000,
                    dest='num_samples',
                    help='Number of samples the data loader will load',
                    type=int
                    )

parser.add_argument('--loss_weights)',
                    default=(1,1,1),
                    dest='loss_weights',
                    help='Weight values for the loss',
                    type=float,
                    nargs="+"
                    )
parser.add_argument('--dlr)',
                    default=0.1,
                    dest='dlr',
                    help='Discriminator Learning Rate Ratio compared to the LR',
                    type=float
                    )
parser.add_argument('--Dfeatures)',
                    default=32,
                    dest='Dfeatures',
                    help='Discriminator Features',
                    type=int
                    )
parser.add_argument('--img_size)',
                    default=256,
                    dest='img_size',
                    help='Image input size',
                    type=int
                    )


args = parser.parse_args()


batch_size = args.batch_size
workers = args.n_workers
loss_weights = tuple(args.loss_weights)


model_name =  args.test_name #"RED10"

num_epochs = 500
learning_rate = args.learning_rate
load = args.load
last_epoch=args.start_epoch
model_dir = "/ocean/projects/iri180005p/bvw546/tmp/fence_models" + "/" + model_name
log_dir = "/ocean/projects/iri180005p/bvw546/tmp/log" + "/" + model_name


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

image_size = (args.img_size,args.img_size)
# Create the dataloader
train_dataloader = torch.utils.data.DataLoader(NewFenceDataset(coco_root + coco_image_folder, 
                                                               coco_data, 
                                                               fence_file_list,
                                                              img_size = image_size, 
                                                              num_samples=args.num_samples),
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

generator = REDNet20(num_features=args.features)
generator = nn.DataParallel(generator)

discriminator = Discriminator(num_features=args.Dfeatures)
discriminator = nn.DataParallel(discriminator)

if load:
    checkpoint = torch.load(os.path.join(model_dir, model_name + '_epoch-{0:0>3}.pth'.format(last_epoch)))
    generator.load_state_dict(checkpoint)
    gen_loss_list = list(np.load(log_dir + '/GENERATOR_LossList.npy'))
    disc_loss_list = list(np.load(log_dir + '/DISCRIMINATOR_LossList.npy'))
#     intermediate_test_images = list(np.load(log_dir + '/intermediate_imgs.npy'))
else:
    last_epoch=0
    gen_loss_list = []
    disc_loss_list = []

generator.to(device)
discriminator.to(device)

optimizerG = torch.optim.Adam(generator.parameters(), lr=learning_rate)
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=learning_rate * args.dlr)

real_label = 1.
fake_label = 0.

criterion = nn.BCELoss()

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
        ############################################
        ### TRAIN THE DESCRIMINATOR ###
        ############################################
        ## Train with the all real batch 
        discriminator.zero_grad()
        # Get the array of true labels
        label = torch.full((gt_img.shape[0],), real_label, dtype=torch.float, device=device)
        # Get the output for the true images
        output = discriminator(gt_img).view(-1)
        # Calculate the loss and get the gradients
        errD_real = criterion(output, label)
        errD_real.backward(retain_graph=True)
        D_x = output.mean().item()

        ## Train with the fake batch 
        fake = generator(fenced_img)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = discriminator(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward(retain_graph=True)
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()



        ############################################
        ### TRAIN THE Generator ###
        ############################################
        optimizerG.zero_grad()
        # SKIP output = generator(fenced_img)

        # Calculate the adverasarial loss
        label.fill_(real_label)
        D_prediction = discriminator(fake).view(-1)
        
        #calculate the loss
        adv_loss = criterion(D_prediction, label)
        sim_loss = loss_weights[0] * ssim_loss(gt_img,fake,3) + loss_weights[1] * nn.MSELoss()(gt_img,fake) 

        loss = sim_loss + 0.05 * adv_loss * loss_weights[2]
                            #nn.L1Loss(gt_img,output)) ## SSIM and L1 for regularization


        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizerG.step()
        
        # Update the progress bar
        loop.set_description(f"Epoch [{last_epoch+epoch}/{last_epoch+num_epochs}]")
        loop.set_postfix(loss=loss.item())
        
    torch.save(generator.state_dict(), os.path.join(model_dir, model_name + 'GENERATOR_epoch-{0:0>3}.pth'.format(last_epoch+epoch)))
    torch.save(discriminator.state_dict(), os.path.join(model_dir, model_name + 'DISCRIMINATOR_epoch-{0:0>3}.pth'.format(last_epoch+epoch)))

    # Save the loss
    gen_loss_list.append((last_epoch+epoch,(sim_loss.item(),adv_loss.item())))
    disc_loss_list.append((last_epoch+epoch,errD.item()))

    np.save(log_dir + '/GENERATOR_LossList.npy', np.array(gen_loss_list))
    np.save(log_dir + '/DISCRIMINATOR_LossList.npy', np.array(disc_loss_list))

    with torch.no_grad():
        for i, data in enumerate(test_dataset):
            output = generator(data.to(device))
            #intermediate_test_images.append(output.cpu().numpy())
            #save the individual epoch results
            #np.save(log_dir + '/intermediate_imgs', np.array(intermediate_test_images))
            np.save(log_dir + '/ADV_intermediate_imgs_{0:0>3}'.format(last_epoch+epoch), output.cpu().numpy())
            print('saved!')
            
              
    # Update the Callbacks
    writer.add_scalar('Loss/train', loss.item(), last_epoch+epoch)
#         writer.add_scalar('Loss/test', np.random.random(), n_iter)