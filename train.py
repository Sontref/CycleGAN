import os
import time
import datetime
import itertools
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.utils import save_image, make_grid

import model
from model import Generator, Discriminator
from datasets import ImageDataset, FakeImageBuffer


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


parser = argparse.ArgumentParser()
parser.add_argument("--start_from", type=int, default=-1, help="epoch number to start from; -1 for training from scratch")
parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs")
parser.add_argument("--dataset_name", type=str, default="horse2zebra", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=10, help="number of samples in batch")
parser.add_argument("--img_height", type=int, default=256, help="image height in pixels")
parser.add_argument("--img_width", type=int, default=256, help="image width in pixels")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--num_residual", type=int, default=9, help="number of residual blocks")
parser.add_argument("--lambda_cycle", type=float, default=10.0, help="cycle loss weight")
args = parser.parse_args()
print(args)


#############################
# Dataloaders; fake buffers #
#############################
# TODO: make path_to_data exactly path_to_data.
# Now script requires to be launched only from location of *dataset_name* dir.
train_loader = DataLoader(
    ImageDataset(path_to_data='datasets/%s' % args.dataset_name, size=(args.img_height,args.img_width), mode='train'),
    batch_size=args.batch_size,
    num_workers=2,
    shuffle=True
)

test_loader = DataLoader(
    ImageDataset(path_to_data='datasets/%s' % args.dataset_name, size=(args.img_height,args.img_width), mode='test'),
    batch_size=5,
    num_workers=2,
    shuffle=True
)

fake_A_buffer = FakeImageBuffer()
fake_B_buffer = FakeImageBuffer()

##########
# Models #
##########
G_AB = Generator(in_channels=args.channels, num_residual=args.num_residual).to(device=device)
G_BA = Generator(in_channels=args.channels, num_residual=args.num_residual).to(device=device)
D_A = Discriminator().to(device=device)
D_B = Discriminator().to(device=device)

if args.start_from == -1:
    G_AB.apply(model.init_weights_func)
    G_BA.apply(model.init_weights_func)
    D_A.apply(model.init_weights_func)
    D_B.apply(model.init_weights_func)
else:
    G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (args.dataset_name, args.start_from)))
    G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (args.dataset_name, args.start_from)))
    D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (args.dataset_name, args.start_from)))
    D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (args.dataset_name, args.start_from)))


##############
# Criterions #
##############
criterion_adv = nn.MSELoss()
criterion_cycle = nn.L1Loss()

lambda_cycle = args.lambda_cycle

##############
# Optimizers #
##############
G_optimizer = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), 2e-4)
D_A_optimizer = torch.optim.Adam(D_A.parameters(), 2e-4)
D_B_optimizer = torch.optim.Adam(D_B.parameters(), 2e-4)

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(G_optimizer, lr_lambda=lambda epoch: 2 - epoch/100)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(D_A_optimizer, lr_lambda=lambda epoch: 2 - epoch/100)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(D_B_optimizer, lr_lambda=lambda epoch: 2 - epoch/100)

############################################
# Generating and saving images             #
# From https://github.com/eriklindernoren/ #
############################################
def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(test_loader))
    G_AB.eval()
    G_BA.eval()
    real_A = imgs["A"].permute(0, 3, 1, 2).to(device=device).float()
    fake_B = G_AB(real_A)
    real_B = imgs["B"].permute(0, 3, 1, 2).to(device=device).float()
    fake_A = G_BA(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "generated_images/%s/%s.png" % (args.dataset_name, batches_done), normalize=False)


############
# Training #
############
batch_time = time.time()
for epoch in range(args.start_from, args.num_epochs):
    for i, batch in enumerate(train_loader):

        real_A = batch['A'].permute(0, 3, 1, 2).to(device=device).float()
        real_B = batch['B'].permute(0, 3, 1, 2).to(device=device).float()

        #######################
        # Training generators #
        #######################
        G_AB.train()
        G_BA.train()

        G_optimizer.zero_grad()

        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)
        rec_A = G_BA(fake_B)
        rec_B = G_AB(fake_A)
        
        # Adversarial Loss
        out_DA = D_A(fake_A)     # TODO: need to avoid this; but how to get output shape of discriminator?
        out_DB = D_B(fake_B)     # TODO: need to avoid this; but how to get output shape of discriminator?
        loss_adv_AB = criterion_adv(out_DB, torch.ones(out_DB.shape, device=device))
        loss_adv_BA = criterion_adv(out_DA, torch.ones(out_DA.shape, device=device))
        loss_adv = (loss_adv_AB + loss_adv_BA) / 2
        # Cycle Loss
        loss_cycle_A = criterion_cycle(real_A, rec_A)
        loss_cycle_B = criterion_cycle(real_B, rec_B)  
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
        # Total generator loss
        loss_gen = loss_adv + lambda_cycle * loss_cycle
        
        loss_gen.backward()
        G_optimizer.step()


        ###########################
        # Training discriminators #
        ###########################
        D_A_optimizer.zero_grad()

        preds_real = D_A(real_A)
        loss_real = criterion_adv(preds_real, torch.ones(preds_real.shape, device=device))
        fake = fake_A_buffer.push_and_pop(fake_A)
        preds_fake = D_A(fake.detach())
        loss_fake = criterion_adv(preds_fake, torch.zeros(preds_fake.shape, device=device))
        
        loss_D_A = (loss_real + loss_fake) / 2
        loss_D_A.backward()
        D_A_optimizer.step()
        

        D_B_optimizer.zero_grad()

        preds_real = D_B(real_B)
        loss_real = criterion_adv(preds_real, torch.ones(preds_real.shape, device=device))
        fake = fake_B_buffer.push_and_pop(fake_B)
        preds_fake = D_B(fake.detach())
        loss_fake = criterion_adv(preds_fake, torch.zeros(preds_fake.shape, device=device))
        
        loss_D_B = (loss_real + loss_fake) / 2
        loss_D_B.backward()
        D_B_optimizer.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        ###########
        # Logging #
        ########### 
        batches_done = epoch * len(train_loader) + i
        batches_left = args.num_epochs * len(train_loader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - batch_time))
        batch_time = time.time()

        print("\r[Epoch %d/%d] [Iteration %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f] ETA: %s" 
            % (
                epoch, args.num_epochs, i, len(train_loader), 
                loss_D.item(), loss_gen.item(), loss_adv.item(), loss_cycle.item(),
                time_left
                )
        )

        # If at sample interval save image
        if batches_done > 0 and batches_done % 100 == 0:
            sample_images(batches_done)

    if epoch >= 100:
        lr_scheduler_G.step(epoch)
        lr_scheduler_D_A.step(epoch)
        lr_scheduler_D_B.step(epoch)

    if epoch % 30 == 0:
        torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (args.dataset_name, epoch))
        torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (args.dataset_name, epoch))
        torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (args.dataset_name, epoch))
        torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (args.dataset_name, epoch))
