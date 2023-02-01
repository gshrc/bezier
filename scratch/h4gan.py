# h4gan.py, dcgan, utf-8 colab
"""
# deep convolutional GAN (DCGAN)
### requirements:
• Create deep convolutional GAN model on MNIST dataset with Pytorch. 
• Accept input image size of 28 x 28. 
• Start with 128 batch size, input noise of 100 dimension, Adam optimizer learning rate of 2e-4. 
• Hyperparameter tuned per performance.
• LeakyRelu on all linear activation layers, slope 0.2
• Tanh on last generator activation layer
• Sigmoid on last discriminator activation layer
• Use Pooling or Conv stride=2 for feature downsampling
• Conv2d, UpssamplingBilinear2d, ConvTransposed (for Generator linear layers)
### generative adversarial network (GAN) ref:
• min G max D L(D, G) = Ex∼pdata(x)[log D(x)] + Ez∼pz(z)[log(1 − D(G(z)))] 

"""
"""
# notes __
### input tensor is of form:  
[batchsize, 1, 28, 28]  
[batchsize, input_channels, height, width]
### output desired is of form:
[batchsize, 1]  

"""
# !pip install tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import numpy as np
from torch.optim.lr_scheduler import StepLR
import torchvision.utils as vutils
from torch.utils.data import DataLoader, TensorDataset
from scipy import linalg
from scipy.stats import entropy
import tqdm
import cv2

# image input size
image_size = 28

# Setting up transforms to resize and normalize 
transform = transforms.Compose([transforms.ToTensor(),])

# batchsize of dataset
batch_size = 100

# hyperparams
nz = 100     # set num vector for noise generator
nc = 1       # set num channels for test images
ngf = 128    # set generator feature map size
ndf = 128    # set discriminator feature map depth

# Load MNIST Dataset
gan_train_dataset = datasets.MNIST(root='./MNIST/', train=True, transform=transform, download=True)
gan_train_loader = torch.utils.data.DataLoader(dataset=gan_train_dataset, batch_size=batch_size, shuffle=True)

# Load CIFAR-10 Dataset
# gan_train_dataset = datasets.CIFAR10(root='./cifar10_data/', train=True, transform=transform, download=True)

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Spot checks, plot some training images
real_batch = next(iter(gan_train_loader))
plt.figure(figsize=(5,5))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=10, normalize=True).cpu(),(1,2,0)))

# Sanity check shape of loaded data # TODO.
print("MINST data set details:"), print(gan_train_dataset)
print("=======================")
print("data sample shape (3D):"), print(gan_train_dataset[0][0].shape)
print("batched shape (4D)"), print(real_batch[0].shape)




"""## Model Definition"""
class DCGAN_Generator(nn.Module):
    def __init__(self):
        super(DCGAN_Generator,self).__init__()
        ################################
        self.conv0 = torch.nn.Conv2d(kernel_size=1, stride=1, padding=0, in_channels=100, out_channels=128*7*7)
        self.up1 = torch.nn.UpsamplingBilinear2d(size=(14,14))
        self.up2 = torch.nn.UpsamplingBilinear2d(size=(28,28))
        self.conv1 = torch.nn.Conv2d(kernel_size=5, padding=2, in_channels=128, out_channels=64)
        self.conv2 = torch.nn.Conv2d(kernel_size=5, padding=2, in_channels=64, out_channels=1)

        self.relu0 = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu1 = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu2 = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, input):
        # input size: batchsize, 100, 7, 7
        out = self.relu0(self.conv0(input))
        out = out.view(out.size(0), 128, 7, 7)
        out = self.up1(out)
        out = self.relu1(self.conv1(out))
        out = self.up2(out)
        out = self.relu2(self.conv2(out))
        # output size
        return out
    
class DCGAN_Discriminator(nn.Module):
    def __init__(self):
        super(DCGAN_Discriminator, self).__init__()
        ################################
        self.conv1 = torch.nn.Conv2d(kernel_size=5, stride=2, padding=2, in_channels=1, out_channels=64)
        self.conv2 = torch.nn.Conv2d(kernel_size=5, stride=2, padding=2,in_channels=64, out_channels=128)
        self.relu1 = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu2 = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fc = torch.nn.Linear(128*7*7, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        out1 = self.relu1(self.conv1(input))
        out2 = self.relu2(self.conv2(out1))
        out2 = out2.view(out2.size(0), -1)
        out2 = self.fc(out2)
        out5 = self.sigmoid(out2)
        return out5

g=DCGAN_Generator()
batchsize=2
z=torch.zeros((batchsize, 100, 1, 1))
out = g(z)
print(out.size()) # expect size [batchsize, 1, 28, 28]

d=DCGAN_Discriminator()
x=torch.zeros((batchsize, 1, 28, 28))
out = d(x)
print(out.size()) # expect size [batchsize, 1]




"""## GAN loss """
import torch

def loss_discriminator(D, real, G, noise, Valid_label, Fake_label, criterion, optimizerD):
    '''
    #####################################################
    1. Forward real images into the discriminator
    2. Compute loss between Valid_label and dicriminator output on real images
    3. Forward noise into the generator to get fake images
    4. Forward fake images to the discriminator
    5. Compute loss between Fake_label and discriminator output on fake images 
       (remember to detach the gradient from the fake images using detach()!)
    6. sum real loss and fake loss as the loss_D
    7. we also need to output fake images generate by G(noise) for loss_generator computation
    '''
    r_disc_output = D(real).squeeze()
    f_loss = criterion(r_disc_output,Valid_label)
    fake_imgs = G(noise)
    d_disc_output = D(fake_imgs.detach()).squeeze()
    s_loss = criterion(d_disc_output,Fake_label)
    loss_D = f_loss + s_loss
    return loss_D, fake_imgs


def loss_generator(netD, netG, fake, Valid_label, criterion, optimizerG):
    '''
    #####################################################
    1. Forward fake images to the discriminator
    2. Compute loss between valid labels and discriminator output on fake images
    '''
    f_disc_output = netD(fake).squeeze()
    loss_G = criterion(f_disc_output,Valid_label)
    return loss_G


# Commented out IPython magic to ensure Python compatibility.
# %pip install torchsummary
import torchvision.utils as vutils
from torch.optim.lr_scheduler import StepLR
import pdb

# Prefer GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of channels
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

netG = DCGAN_Generator().to(device)
netD = DCGAN_Discriminator().to(device)

from torchsummary import summary
print(summary(netG,(100,1,1)))
print(summary(netD,(1, 28, 28)))



"""## TRAINING"""
# Commented out IPython magic to ensure Python compatibility.
import torchvision.utils as vutils
from torch.optim.lr_scheduler import StepLR
import pdb

# Preference cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Number of channels
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100

# Create the generator and discriminator
netG = DCGAN_Generator().to(device)
netD = DCGAN_Discriminator().to(device)

# Initialize BCELoss function
criterion = nn.BCELoss()
# Create latent vector to test the generator performance
fixed_noise = torch.randn(36, nz, 1, 1, device=device)
# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0
learning_rate = 0.0002
beta1 = 0.5

# Setup Adam optimizers for G and D
################################
# WIP. 
# custom weights initialization called on netG and netD
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)
# netG.apply(weights_init)
# netD.apply(weights_init)
#
##################################
optimizerD = optim.Adam(list(netD.parameters()), lr=learning_rate, betas=(beta1, 0.999))
optimizerG = optim.Adam(list(netG.parameters()), lr=learning_rate, betas=(beta1, 0.999))
print("optimizers D, G:"), print(optimizerD), print(optimizerG)
##################################

img_list = []
real_img_list = []
G_losses = []
D_losses = []
iters = 0
num_epochs = 30




"""## TRAINING LOOP"""
import pdb

def load_param(num_eps):
    model_saved = torch.load('/content/gan_{}.pt'.format(num_eps))
    netG.load_state_dict(model_saved['netG'])
    netD.load_state_dict(model_saved['netD'])
    
# GAN Training loop
for epoch in range(num_epochs):
    for i, data in enumerate(gan_train_loader, 0):
        ############################
        real = data[0].to(device)
        b_size = real.size(0)
        noise = torch.randn(b_size, nz, 1, 1, device=device)

        Valid_label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        Fake_label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ############################
        netD.zero_grad()
        loss_D, fake = loss_discriminator(netD, real, netG, noise, Valid_label, Fake_label,criterion,optimizerD)
        loss_D.backward()
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ############################
        # Compute Generator loss
        netG.zero_grad()
        loss_G = loss_generator(netD, netG, fake, Valid_label, criterion, optimizerG)
        loss_G.backward()
        optimizerG.step()

        # Calculate gradients for G:

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
										% (epoch, num_epochs, i, len(gan_train_loader),
                    loss_D.item(), loss_G.item()))

        # Save Losses for plotting later
        G_losses.append(loss_G.item())
        D_losses.append(loss_D.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(gan_train_loader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
				

        
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

checkpoint = {'netG': netG.state_dict(),
              'netD': netD.state_dict()}
torch.save(checkpoint, 'gan_{}.pt'.format(num_epochs))





"""Qualitative Visualisations"""

# Test GAN on a random sample and display on 6X6 grid
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())