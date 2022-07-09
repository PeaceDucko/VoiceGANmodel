#%%
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch
import os
from pathlib import Path
import glob
import torch.nn as nn
from itertools import chain
import torch.optim as optim
import time
from progressbar import ETA, Bar, Percentage, ProgressBar
import simpleaudio as sa
# from model_Adp import *
from model_Adp_custom import *
import scipy.io as sio
from torchvision.io import read_image
import torchvision.transforms as transforms
from torchvision.transforms.functional import pad
from torch.utils.data import Dataset
from torchvision.transforms import *
import torchvision
from collections import defaultdict
from IPython.display import display
from IPython.display import clear_output
import torchaudio
from torchvision.transforms.functional import pad
from matplotlib.lines import Line2D

import librosa
from tqdm import tqdm

from utils import audio_tools as audio
from utils.hparams import *

import logging
import sys

SR = hparams['sample_rate']
num_mels = hparams['num_mels']
IMAGE_SIZE = 512 # Generate melspectrogram image in 128 x 128

logging.getLogger('matplotlib.font_manager').disabled = True
#%%

"""
Custom logging
"""
root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)
"""
End
"""

logging.info("Running")

#%%

ngpu = 2
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
logging.info(device)
# %%

path_A = "../../../scratch/sram/CMU/A"
path_B = "../../../scratch/sram/CMU/B"

n = 10000

files_A = glob.glob(path_A+"/*.wav")[:n]
files_B = glob.glob(path_B+"/*.wav")[:n]

# %%

if n > max(len(files_A), len(files_B)):
    n = max(len(files_A), len(files_B))
    
print("n: ",str(n))

# %%

def as_np(data):
    return data.cpu().data.numpy()

# %%

# Image to npy array
spect_dir_A = "../../../scratch/sram/CMU/A_spect"
spect_dir_B = "../../../scratch/sram/CMU/B_spect"

class TIMITDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_path = glob.glob(self.img_dir+"*.npy")

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        image = np.load(img_path)
        if self.transform:
            image = self.transform(image)  
            
        image = image.type(torch.FloatTensor)
    
        return image

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
    
# dataset = torchvision.datasets.ImageFolder(spect_dir, transform)
data_A = TIMITDataset(spect_dir_A, transform=transform)
data_B = TIMITDataset(spect_dir_B, transform=transform)

print(data_A.__len__())
print(data_B.__len__())

# %%
img = 10

print(data_A.__getitem__(img).mean())
print(data_A.__getitem__(img).var())
print(data_A.__getitem__(img).shape)
# %%

# Create the dataloader
batch_size = 8
num_workers = 1

data_A = torch.utils.data.DataLoader(data_A, 
                                     batch_size=batch_size, 
                                     shuffle=True, 
                                     num_workers=num_workers,
                                     drop_last=True)

data_B = torch.utils.data.DataLoader(data_B, 
                                     batch_size=batch_size, 
                                     shuffle=True, 
                                     num_workers=num_workers,
                                     drop_last=True)
# %%

# Display image.
train_features = next(iter(data_A))
img = train_features[0].squeeze()
plt.imshow(img, origin="lower", cmap=plt.get_cmap("magma"))
plt.show()
# %%

batch_tensor = next(iter(data_A))
grid_img = torchvision.utils.make_grid(batch_tensor, nrow=4, normalize=True)

plt.figure(figsize=(10, 10))
plt.title('Example speech')
plt.axis('off')
plt.imshow(grid_img.permute(1, 2, 0), origin="lower", cmap=plt.get_cmap("magma"))
# %%

n_batches = min( len(data_A), len(data_B) )

print(n_batches)

# %%
recon_criterion = nn.MSELoss() #nn.L1Loss()
gen_criterion = nn.BCELoss()
dis_criterion = nn.BCELoss()
feat_criterion = nn.HingeEmbeddingLoss()
stl_criterion = nn.CrossEntropyLoss()
# %%

## Change to 3 inputs 
def get_gan_loss(dis_real, dis_fake1, dis_fake2, gen_criterion, dis_criterion, topk, device):
    topk = int(topk)
    
    labels_dis_real = Variable(torch.ones( [dis_real.size()[0], 1] ))
    labels_dis_fake1 = Variable(torch.zeros([dis_fake1.size()[0], 1] ))
    labels_dis_fake2 = Variable(torch.zeros([dis_fake2.size()[0], 1] ))
    labels_gen1 = Variable(torch.ones([dis_fake1.size()[0], 1]))
    labels_gen2 = Variable(torch.ones([dis_fake2.size()[0], 1]))

    labels_dis_real = labels_dis_real.to(device)
    labels_dis_fake1 = labels_dis_fake1.to(device)
    labels_dis_fake2 = labels_dis_fake2.to(device)
    labels_gen1 = labels_gen1.to(device)
    labels_gen2 = labels_gen2.to(device)
    
    dis_real = torch.reshape(dis_real, (topk,1))
    dis_fake1 = torch.reshape(dis_fake1, (topk,1))
    dis_fake2 = torch.reshape(dis_fake2, (topk,1))
    
    dis_loss = dis_criterion( dis_real, labels_dis_real ) * 0.4 + dis_criterion( dis_fake1, labels_dis_fake1 ) * 0.3 + dis_criterion( dis_fake2, labels_dis_fake2 ) * 0.3
    gen_loss = gen_criterion( dis_fake1, labels_gen1 ) * 0.5 + gen_criterion( dis_fake2, labels_gen2 ) * 0.5

    return dis_loss, gen_loss
# %%

def get_fm_loss(real_feats, fake_feats, criterion, device):
    losses = 0
    for real_feat, fake_feat in zip(real_feats, fake_feats):
        # pdb.set_trace()
        l2 = (real_feat.mean(0) - fake_feat.mean(0)) * (real_feat.mean(0) - fake_feat.mean(0))
        loss = criterion( l2, Variable( torch.ones( l2.size() ) ).to(device) )
        losses += loss

    return losses
# %%

## Use CrossEntropyLoss: target should be N
def get_stl_loss(A_stl, A1_stl, A2_stl, B_stl, B1_stl, B2_stl, criterion, device):
    # for nn.CrossEntropyLoss, the target is class index.
    labels_A = Variable(torch.ones( A_stl.size()[0] )) # NLL/CE target N not Nx1
    labels_A.data =  labels_A.data.type(torch.LongTensor)

    labels_A1 = Variable(torch.ones( A1_stl.size()[0] )) # NLL/CE target N not Nx1
    labels_A1.data =  labels_A1.data.type(torch.LongTensor)

    labels_A2 = Variable(torch.ones( A2_stl.size()[0] )) # NLL/CE target N not Nx1
    labels_A2.data =  labels_A2.data.type(torch.LongTensor)
 
    labels_B = Variable(torch.zeros(B_stl.size()[0] ))
    labels_B.data =  labels_B.data.type(torch.LongTensor)

    labels_B1 = Variable(torch.zeros(B1_stl.size()[0] ))
    labels_B1.data =  labels_B1.data.type(torch.LongTensor)

    labels_B2 = Variable(torch.zeros(B2_stl.size()[0] ))
    labels_B2.data =  labels_B2.data.type(torch.LongTensor)
   
    labels_A = labels_A.to(device)
    labels_A1 = labels_A1.to(device)
    labels_A2 = labels_A2.to(device)
    labels_B = labels_B.to(device)
    labels_B1 = labels_B1.to(device)
    labels_B2 = labels_B2.to(device)

    A_stl = np.squeeze(A_stl)
    A1_stl = np.squeeze(A1_stl)
    A2_stl = np.squeeze(A2_stl)
    B_stl = np.squeeze(B_stl)
    B1_stl = np.squeeze(B1_stl)
    B2_stl = np.squeeze(B2_stl)

    stl_loss_A = criterion( A_stl, labels_A ) * 0.2 + criterion( A1_stl, labels_A1 ) * 0.15 + criterion( A2_stl, labels_A2 ) * 0.15
    stl_loss_B = criterion( B_stl, labels_B ) * 0.2 + criterion( B1_stl, labels_B1 ) * 0.15 + criterion( B2_stl, labels_B2 ) * 0.15
    stl_loss = stl_loss_A + stl_loss_B

    return stl_loss
# %%

def delta_regu(input_v, batch_size, criterion=nn.MSELoss()):
    losses = 0
    for i in range(batch_size):
        # pdb.set_trace()
        input_temp = np.squeeze(input_v.data[i,:,:,:])
        # no need to take mean among 3 channels since current input is 256x256 instead of 3x256x256
        input_temp = input_temp.cpu().numpy()
        input_delta = np.absolute(librosa.feature.delta(input_temp))
        b = input_delta.shape[1]
        delta_loss = criterion(Variable((torch.from_numpy(input_delta)).type(torch.DoubleTensor)), Variable((torch.zeros([256,b])).type(torch.DoubleTensor)))
        # delta_loss = criterion((torch.from_numpy(input_delta)), Variable((torch.zeros([256,256]))))
        losses += delta_loss

    delta_losses = losses/batch_size

    return delta_losses.type(torch.cuda.FloatTensor)  

# %%

generator_A = Generator(ngpu)
generator_B = Generator(ngpu)
discriminator_A = Discriminator(ngpu)
discriminator_B = Discriminator(ngpu)
discriminator_S = StyleDiscriminator(ngpu)

generator_A = generator_A.to(device)
generator_B = generator_B.to(device)
discriminator_A = discriminator_A.to(device)
discriminator_B = discriminator_B.to(device)
discriminator_S = discriminator_S.to(device)
    
if ngpu > 1:
    generator_A = nn.DataParallel(generator_A, device_ids = range(ngpu))
    generator_B = nn.DataParallel(generator_B, device_ids = range(ngpu))
    discriminator_A = nn.DataParallel(discriminator_A, device_ids = range(ngpu))
    discriminator_B = nn.DataParallel(discriminator_B, device_ids = range(ngpu))
    discriminator_S = nn.DataParallel(discriminator_S, device_ids = range(ngpu)) 
# %%

gen_params = chain(generator_A.parameters(), generator_B.parameters())
dis_params = chain(discriminator_A.parameters(), discriminator_B.parameters())
stl_params =  discriminator_S.parameters() 

# %%
learning_rate = 0.0002
lr_decay = 0

optim_gen = optim.AdamW( gen_params, lr=learning_rate, betas=(0.5,0.999))
optim_dis = optim.AdamW( dis_params, lr=learning_rate, betas=(0.5,0.999))
optim_stl = optim.AdamW( stl_params, lr=learning_rate, betas=(0.5,0.999))

# LR decay
if lr_decay == 1:
    sched_gen = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)
    sched_dis = optim.lr_scheduler.ExponentialLR(optim_dis, gamma=0.99)
    sched_stl = optim.lr_scheduler.ExponentialLR(optim_stl, gamma=0.99)

start = time.time()      

# %%

def plot_grad_flow(named_parameters, filename, path):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.cpu().abs().mean())
            max_grads.append(p.grad.cpu().abs().max())
    plt.figure(figsize=(10,5))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation=30)
    plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig( os.path.join(path, str( filename ) + ".jpg"))
    plt.show()
    plt.close()

# %%
epoch_size = 1000

topk = batch_size
gamma = 0.99
v = topk*0.5

# Strong GAN loss for certain period at the beginning
gan_curriculum = 1000
# Set the lambda weight between GAN loss and Recon loss during curriculum period at the beginning. We used the 0.01 weight
starting_rate = 0.01
# Set the lambda weight between GAN loss and Recon loss after curriculum period. We used the 0.5 weight
default_rate = 0.5
# choose among gan/recongan/discogan/spec_gan. gan - standard GAN, recongan - GAN with reconstruction, discogan - DiscoGAN, spec_gan - My modified GAN model for speech
model_arch = "spec_gan"
# Print loss values every log_interval iterations. The current code ensures that the logs are updated 10 times every epoch
log_interval = int(n_batches/10)
# Save test results every image_save_interval iterations
image_save_interval = 2000
# Save models every model_save_interval iterations
model_save_interval = 10000
# Set the path for trained models
task_name = "impersonation"

update_interval = 10

results_folder_name = "CMU_Adam_lr"+str(learning_rate)+"_lr_decay"+str(lr_decay)+"_starting_rate"+str(starting_rate)+"_default_rate"+str(default_rate)

path = "./results/"+results_folder_name
isExist = os.path.exists(path)

if not isExist:
    os.makedirs(path)
    os.makedirs(path+"/images")
    os.makedirs(path+"/gradients")
    os.makedirs(path+"/models")
    print("The new directories are created!")

model_path = path+"/models/"
images_path = path+"/images/"
gradients_path = path+"/gradients/"
    
model_path = os.path.join( model_path, task_name )
model_path = os.path.join( model_path, model_arch )

Path(model_path).mkdir(parents=True, exist_ok=True)

log_gen_loss = np.array([])
log_dis_loss = np.array([])
log_stl_loss = np.array([])
log_delta_A = np.array([])
log_delta_B = np.array([])
log_fm_loss_A = np.array([])
log_fm_loss_B = np.array([])
log_recon_loss_A = np.array([])
log_recon_loss_B = np.array([])
log_gen_loss_A = np.array([])
log_gen_loss_B = np.array([])
img_list = []

plt.figure(figsize=(10, 15))

for epoch in range(epoch_size):
    print("k:", topk)
    print("v:", v)
    print("gamma:", gamma)
    topk = max(gamma*topk,v)
    
    epoch_stats = defaultdict(lambda: 0)
    widgets = ['epoch #%d|' % epoch, Percentage(), Bar(), ETA()]
    pbar = ProgressBar(maxval=n_batches, widgets=widgets)
    pbar.start()
        
    for iters, (A,B) in enumerate(zip(data_A,data_B)):
        
        pbar.update(iters)

        generator_A.zero_grad()
        generator_B.zero_grad()
        discriminator_A.zero_grad()
        discriminator_B.zero_grad()
        discriminator_S.zero_grad()
           
        A = Variable( torch.FloatTensor( A ) ).to(device)
        B = Variable( torch.FloatTensor( B ) ).to(device)
    
        # A -> AB -> ABA
        # A = A.unsqueeze(1)
        AB, AL_feats, LAB_feats = generator_B(A)
        ABA, ABL_feats, ABLA_feats = generator_A(AB)
        # B -> BA -> BAB
        # B = B.unsqueeze(1)
        BA, BL_feats, LBA_feats = generator_A(B)
        BAB, BAL_feats, BALB_feats = generator_B(BA)
        
        recon_loss_BA = recon_criterion( BA, B)
        recon_loss_AB = recon_criterion( AB, A)
        recon_loss_ABA = recon_criterion( ABA, A)
        recon_loss_BAB = recon_criterion( BAB, B)
        
        recon_loss_A = 30 *recon_loss_AB + 70 * recon_loss_ABA
        recon_loss_B = 30 *recon_loss_BA + 70 * recon_loss_BAB 
        
        # Real/Fake GAN Loss (A)
        A_dis = discriminator_A( A )
        BA_dis = discriminator_A( BA )
        ABA_dis = discriminator_A( ABA )
            
        # New part based on Top-k Training of GANs: Improving GAN Performance by Throwing Away Bad Samples
        A_dis = torch.topk(A_dis , int(topk), dim=0)[0]
        BA_dis = torch.topk(BA_dis , int(topk), dim=0)[0]
        ABA_dis = torch.topk(ABA_dis , int(topk), dim=0)[0]
        
        # will be strange in one epoch
        dis_loss_A, gen_loss_A = get_gan_loss( A_dis, BA_dis, ABA_dis, gen_criterion, dis_criterion, topk, device)
                
        fm_loss_A1 = get_fm_loss(AL_feats, ABLA_feats, feat_criterion, device)
        fm_loss_A2 = get_fm_loss(LAB_feats, ABL_feats, feat_criterion, device)
        fm_loss_A = fm_loss_A1 + fm_loss_A2
        
        # Real/Fake GAN Loss (B)
        B_dis = discriminator_B( B )
        AB_dis = discriminator_B( AB )
        BAB_dis = discriminator_B( BAB )
        
        # New part based on Top-k Training of GANs: Improving GAN Performance by Throwing Away Bad Samples
        B_dis = torch.topk(B_dis , int(topk), dim=0)[0]
        AB_dis = torch.topk(AB_dis , int(topk), dim=0)[0]
        BAB_dis = torch.topk(BAB_dis , int(topk), dim=0)[0]

        dis_loss_B, gen_loss_B = get_gan_loss( B_dis, AB_dis, BAB_dis, gen_criterion, dis_criterion, topk, device)
        fm_loss_B1 = get_fm_loss( BL_feats, BALB_feats, feat_criterion, device)
        fm_loss_B2 = get_fm_loss( LBA_feats, BAL_feats, feat_criterion, device)
        fm_loss_B = fm_loss_B1 + fm_loss_B2
        
        # Style Discriminator Loss
        A_stl = discriminator_S(A)
        B_stl = discriminator_S(B)
        AB_stl = discriminator_S(AB) 
        BA_stl = discriminator_S(BA)
        ABA_stl = discriminator_S(ABA) 		
        BAB_stl = discriminator_S(BAB)
        
        stl_loss = get_stl_loss(A_stl, BA_stl, ABA_stl, B_stl, AB_stl, BAB_stl, stl_criterion, device)

        # Delta regularizer
        BA_delta = delta_regu(BA, batch_size)
        AB_delta = delta_regu(AB, batch_size)
        ABA_delta = delta_regu(ABA, batch_size)
        BAB_delta = delta_regu(BAB, batch_size)

        delta_A = BA_delta + ABA_delta
        delta_B = AB_delta + BAB_delta
        
        # Total Loss
        if iters < gan_curriculum:
            rate = starting_rate
        else:
            rate = default_rate
            
        gen_loss_A_total = (gen_loss_A) * (1.-rate) + (recon_loss_A*0.6 + fm_loss_A*0.3 + delta_A*0.1)*rate
        gen_loss_B_total = (gen_loss_B) * (1.-rate) + (recon_loss_B*0.6 + fm_loss_B*0.3 + delta_B*0.1)*rate
        
        if model_arch == 'discogan':
            gen_loss = gen_loss_A_total + gen_loss_B_total
            dis_loss = dis_loss_A + dis_loss_B
        elif model_arch == 'spec_gan':
            gen_loss = gen_loss_A_total + gen_loss_B_total
            dis_loss = dis_loss_A + dis_loss_B + stl_loss
        elif model_arch == 'recongan':
            gen_loss = gen_loss_A_total
            dis_loss = dis_loss_B
        elif model_arch == 'gan':
            gen_loss = (gen_loss_B*0.1 + fm_loss_B*0.9)
            dis_loss = dis_loss_B
            
        if iters % update_interval == 0:
            dis_loss.backward()
            optim_dis.step()
            optim_stl.step()
            
            # LR decay
            if lr_decay == 1:
                sched_dis.step()
                sched_stl.step()
                
        else:
            gen_loss.backward()
            optim_gen.step()
            
            # LR decay
            if lr_decay == 1:
                sched_gen.step()
            
        # if iters % 50 == 0: # Update every 50 batches
        if iters % log_interval == 0:
            print("---------------------")
            print("GEN Loss:", as_np(gen_loss_A.mean()), as_np(gen_loss_B.mean()))
            print("Feature Matching Loss:", as_np(fm_loss_A.mean()), as_np(fm_loss_B.mean()))
            print("RECON Loss:", as_np(recon_loss_A.mean()), as_np(recon_loss_B.mean()))
            print("DIS Loss:", as_np(dis_loss_A.mean()), as_np(dis_loss_B.mean()))
            print("Style Loss:", as_np(stl_loss.mean()))
            print("Delta Loss:", as_np(delta_A.mean()), as_np(delta_B.mean()))
            print("Time", (time.time()-start))
            start = time.time()     
                        
            log_gen_loss = np.hstack([log_gen_loss, as_np(gen_loss.mean())])
            log_dis_loss = np.hstack([log_dis_loss, as_np(dis_loss.mean())])
            log_stl_loss = np.hstack([log_stl_loss, as_np(stl_loss.mean())])
            log_delta_A = np.hstack([log_delta_A, as_np(delta_A.mean())])
            log_delta_B = np.hstack([log_delta_B, as_np(delta_B.mean())])
            log_fm_loss_A = np.hstack([log_fm_loss_A, as_np(fm_loss_A.mean())])
            log_recon_loss_A = np.hstack([log_recon_loss_A, as_np(recon_loss_A.mean())])
            log_fm_loss_B = np.hstack([log_fm_loss_B, as_np(fm_loss_B.mean())])
            log_recon_loss_B = np.hstack([log_recon_loss_B, as_np(recon_loss_B.mean())])
            log_gen_loss_A = np.hstack([log_gen_loss_A, as_np(gen_loss_A.mean())])
            log_gen_loss_B = np.hstack([log_gen_loss_B, as_np(gen_loss_B.mean())])
            
            epoch_stats['log_gen_loss'] = log_gen_loss
                
        if iters % model_save_interval == 0:
            torch.save( generator_A, os.path.join(model_path, 'model_gen_A-' + str( iters / model_save_interval )))
            torch.save( generator_B, os.path.join(model_path, 'model_gen_B-' + str( iters / model_save_interval )))
            torch.save( discriminator_A, os.path.join(model_path, 'model_dis_A-' + str( iters / model_save_interval )))
            torch.save( discriminator_B, os.path.join(model_path, 'model_dis_B-' + str( iters / model_save_interval )))
                    
        if epoch % 5 == 0 and iters % log_interval == 0:
            print('Epoch: '+str(epoch))
            for key, val in epoch_stats.items():
                print(' %s=%6.4f' % (key, val[-1] / iters), end='')

            images_for_plot = {
                'real_a': A, 'fake_ab': AB, 'cycle_aba': ABA,
                'real_b': B, 'fake_ba': BA, 'cycle_bab': BAB,
            }

            for z in range(3):
                for i, (im_title, im) in enumerate(images_for_plot.items()):
                    plt.subplot(3, 6, z * 6 + i + 1)
                    plt.imshow(im[z].detach().cpu().numpy().transpose(1, 2, 0) / 2 + 0.5)
                    if z == 0:
                        plt.title(im_title)
                    plt.axis('off')
                    
            plt.tight_layout()
            display(plt.gcf())
            clear_output(wait=True)
            plt.savefig( os.path.join(images_path, 'model-' + str( epoch ) + ".jpg"))
            
            plt.figure(figsize=(10,5))
            plt.title("Generator, Discriminator and StyleDiscriminator Loss During Training")
            plt.plot(log_gen_loss,label="Generator", color="m")
            plt.plot(log_dis_loss,label="Discriminator", color="c")
            plt.plot(log_stl_loss,label="StyleDiscriminator", color="r")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig( os.path.join(path, 'model-' + str( epoch ) + ".jpg"))
            plt.show()
            
            plot_grad_flow(discriminator_A.named_parameters(), 'discriminator_A-' + str( epoch ), gradients_path)
            plot_grad_flow(discriminator_B.named_parameters(), 'discriminator_B-' + str( epoch ), gradients_path)
            plot_grad_flow(discriminator_S.named_parameters(), 'discriminator_S-' + str( epoch ), gradients_path)
            plot_grad_flow(generator_A.named_parameters(), 'generator_A-' + str( epoch ), gradients_path)
            plot_grad_flow(generator_B.named_parameters(), 'generator_B-' + str( epoch ), gradients_path)
    
        plt.close()