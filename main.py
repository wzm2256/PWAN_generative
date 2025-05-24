import argparse
import os
import pdb
import pickle

import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchvision import datasets
import torch
import models.networks as networks
from models.Grad import Grad_Penalty_w
from dataset import ConcatDataset, PartialDataset, ImageRGB, TensorDataset, ImageRGB_fid
from util import cal_dloss_inc
from torch.utils.tensorboard import SummaryWriter
import util
import tqdm
from FID import compute_fid


parser = argparse.ArgumentParser()
parser.add_argument("--d_iter", type=int, default=15, help="Discriminator update frequency")
parser.add_argument("--ratio", type=float, default=0.9, help="Point mass. Set to 1 to use WGAN.")
parser.add_argument("--outlier_ratio", type=float, default=0.1, help="")
parser.add_argument("--exp_name", type=str, default='base', help="default_exp")
parser.add_argument("--batch_size", type=int, default=256, help="")
parser.add_argument("--vis_batch_size", type=int, default=100, help="batch size for visualization")
parser.add_argument("--n_epochs", type=int, default=250, help="number of epochs of training")
parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--save_interval", type=int, default=20, help="Save every n epochs")


args = parser.parse_args()

img_shape = (args.channels, args.img_size, args.img_size)


# Configure data loader
os.makedirs("./data/mnist", exist_ok=True)
os.makedirs("./data/cifar", exist_ok=True)

transform = transforms.Compose([transforms.Resize(args.img_size),
								 transforms.ToTensor(),
								 transforms.Normalize([0.5], [0.5])])

dataset1_ = datasets.MNIST("data/mnist", train=True, download=True, transform=transform)
# convert greyscale image to RGB to that it can be mixed into Cifar
dataset1 = ImageRGB(dataset1_)
dataset2 = datasets.CIFAR10("data/cifar", train=True, download=True, transform=transform)



if not os.path.exists('statistics.pkl'):
# Collect statistics of Cifar and Mnist for FID computation if the statistics are not precomputed.
    transform_fid = transforms.Compose([transforms.Resize(args.img_size),
                                        transforms.ToTensor()])
    dataset1_fid_ = datasets.MNIST("data/mnist", train=True, download=True, transform=transform_fid)
    dataset1_fid = ImageRGB_fid(dataset1_, convert=True)

    dataset2_fid_ = datasets.CIFAR10("data/cifar", train=True, download=True, transform=transform_fid)
    dataset2_fid = ImageRGB_fid(dataset2_fid_, convert=False)

    minist_statistics = compute_fid.compute_statistics_of_dataset(dataset1_fid)
    cifar1_statistics = compute_fid.compute_statistics_of_dataset(dataset2_fid)

    Statistics = {'mnist': minist_statistics, 'cifar':cifar1_statistics}

    with open('statistics.pkl', 'wb') as f_tmp:
        pickle.dump(Statistics, f_tmp)

# Load the precomputed statistics
with open('statistics.pkl', 'rb') as f_statistics:
    Statistics = pickle.load(f_statistics)


keep_num = int(len(dataset2) * args.outlier_ratio)
dataset1_partial = PartialDataset(dataset1, keep_num)
dataset_mix = ConcatDataset([dataset2, dataset1_partial])

dataloader = torch.utils.data.DataLoader(dataset_mix, batch_size=args.batch_size, shuffle=True, num_workers=8)

# Initialize generator and discriminator
G = networks.Generator(args.img_size, args.latent_dim).to('cuda')
D = networks.Discriminator(args.img_size).to('cuda')
G_P = Grad_Penalty_w(1000, gamma=1)


# Optimizers
optimizer = torch.optim.RMSprop(G.parameters(), lr=args.lr)
optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr)


LOG_folder = os.path.join('LOG', args.exp_name + str(np.random.randint(100000)).zfill(8))
os.makedirs(LOG_folder, exist_ok=True)

tf_log = os.path.join(LOG_folder, 'log')
writer = SummaryWriter(tf_log)


for epoch in range(args.n_epochs):

    if epoch % args.save_interval == 0 and epoch > 0:
        # visualize
        vis_dataloader = iter(torch.utils.data.DataLoader(dataset_mix, batch_size=args.vis_batch_size, shuffle=True))
        Total_num = 1000
        All_sample_r = []
        All_sample_f = []
        All_labels = []
        print('Start visualiztion')

        for step in tqdm.tqdm(range(int(Total_num / args.vis_batch_size))):
            (image, label), type = next(vis_dataloader)
            All_sample_r.append(image)
            All_labels.append(type)

            z = torch.randn(args.vis_batch_size, args.latent_dim).to('cuda')

            with torch.no_grad():
                fake_imgs = G(z)

            All_sample_f.append(fake_imgs)

        All_sample_r = torch.cat(All_sample_r, dim=0).to('cpu')
        All_sample_f = torch.cat(All_sample_f, dim=0).to('cpu')
        All_labels = torch.cat(All_labels, dim=0).to('cpu')

        print('Visualizing the results....')
        util.vis(All_sample_r, All_sample_f, All_labels, LOG_folder, name=str(epoch))
        save_image(fake_imgs[:25], f"{LOG_folder}/{epoch}.png", nrow=5, normalize=True)
        save_image(image[:25], f"{LOG_folder}/{epoch}_real.png", nrow=5, normalize=True)


    ### compute fid score
    Total_fid_num = 50000
    All_sample_f = []
    print('Start computing fid')
    for step in tqdm.tqdm(range(int(Total_fid_num / args.batch_size))):
        z = torch.randn(args.batch_size, args.latent_dim).to('cuda')
        with torch.no_grad():
            fake_imgs = G(z)

        All_sample_f.append(fake_imgs)

    All_sample_f = torch.cat(All_sample_f, dim=0) * 0.5 + 0.5

    fake_dataset = TensorDataset(All_sample_f.to('cpu'))

    g_statistics = compute_fid.compute_statistics_of_dataset(fake_dataset)

    fid_cifar = compute_fid.calculate_frechet_distance(*Statistics['cifar'], *g_statistics)
    mni_cifar = compute_fid.calculate_frechet_distance(*Statistics['mnist'], *g_statistics)

    writer.add_scalar('Train/fid_cifar', fid_cifar, epoch)
    writer.add_scalar('Train/mni_cifar', mni_cifar, epoch)

    print(f'Training:\t Epoch{epoch} fid_cifar {fid_cifar} mni_cifar {mni_cifar}')


    # training
    for i, ((imgs, _), type) in enumerate(dataloader):
        # Sample noise as generator input
        z = torch.randn(args.batch_size, args.latent_dim).to('cuda')

        # Generate a batch of images
        fake_imgs = G(z)
        imgs = imgs.to('cuda')
        imgs.requires_grad_(True)

        fake_d = fake_imgs.detach().requires_grad_(True)
        for d in range(args.d_iter):
            potential_r = D(imgs)
            potential_f = D(fake_d)
            d_loss = cal_dloss_inc(potential_r, potential_f, args.ratio)

            gp_loss, M, source_norm, all_norm, grad_all = G_P(d_loss, [fake_d, imgs], args.ratio)
            writer.add_scalar('Train/M_grad', M, i)

            d_loss_all = d_loss + gp_loss

            optimizer_D.zero_grad()
            d_loss_all.backward()
            optimizer_D.step()

        imgs.requires_grad_(False)

        potential_r_g = D(imgs)
        potential_f_g = D(fake_imgs)

        transfer_loss = -cal_dloss_inc(potential_r_g, potential_f_g,  args.ratio)

        optimizer.zero_grad()
        transfer_loss.backward()
        optimizer.step()

        writer.add_scalar('Train/d_loss', transfer_loss, len(dataloader) * epoch + i)
        print(f'Training:\t Epoch{epoch} step:{i} transfer_loss:{transfer_loss.item():.2f}\t')