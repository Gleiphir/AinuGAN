import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision

import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--checkpoint_dir', type=str, default='CKP_1v3_{}_{}_{}')
parser.add_argument('--model', type=str, default='std')

parser.add_argument('--d-lrs', type=float,nargs=3)

args = parser.parse_args()


loader = torch.utils.data.DataLoader(
datasets.ImageFolder('myDset', transform=transforms.Compose([
		transforms.RandomCrop(64),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
	batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

Z_dim = 128
#number of updates to discriminator for every update to generator 
disc_iters = 1

# discriminator = torch.nn.DataParallel(Discriminator()).cuda() # TODO: try out multi-gpu training

import model_cyclegan as model


# because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
# optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
# TODO: replace Parameters with buffers, which aren't returned from .parameters() method.


if args.d_lrs:
	d_lrs = [ x* 1e-4 for x in args.d_lrs]
else:
	d_lrs =  [2e-5]

d_wghs = [1.0 for d_lr in d_lrs]
#d_wghs = [1,1,1]


print("Disc lrs: ",d_lrs)


def mdf(it):
	return ''.join([str(t *1e4)+'-' for t in it])
ckp_mdf = args.checkpoint_dir.format(args.model,"ifuku",mdf(d_lrs))


discriminator = torch.nn.DataParallel(model.Discriminator()).cuda()
generator = model.Generator().cuda()


optim_disc = optim.Adam( filter(lambda p: p.requires_grad, discriminator.parameters()),
	 lr=2e-4, betas=(0.5,0.999)
	)

optim_gen  = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5,0.999))

# use an exponentially decaying learning rate
#scheduler_ds = [optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99) for optim_disc in optim_discs ]
#scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)

g_iter = 0

gen_times = 5

def train(epoch):
	global g_iter,start_t,last_t,now_t
	for batch_idx, (data, target) in enumerate(loader):
		if data.size()[0] != args.batch_size:
			continue
		data, target = Variable(data.cuda()), Variable(target.cuda())

		# update discriminator
		d_losses = []
	

		for _ in range(disc_iters):
			z = Variable(torch.randn(args.batch_size,Z_dim).cuda())
			optim_disc.zero_grad()
			#optim_gen.zero_grad()

			#disc_loss = nn.ReLU()(1.0 - discriminator(data)).mean() + nn.ReLU()(1.0 + discriminator(generator(z))).mean()
			loss_real = (1.0 - discriminator(data))
			loss_fake = (1.0 + discriminator(generator(z)))
			disc_loss = loss_real.mean() + loss_fake.mean()

			d_losses.append(disc_loss.data.item())

			disc_loss.backward()
			optim_disc.step()
		
		
		
		
		z = Variable(torch.randn(args.batch_size, Z_dim).cuda())

		#print(generator(z).size())# update generator
		#optim_disc.zero_grad()
		for _ in range(gen_times):
			optim_gen.zero_grad()
			#print(generator(z).size())
			losses = []

			gen_loss = - discriminator(generator(z)).mean()

			"""
			gen_loss = torch.cat(losses).mean()# 2 * 32 *... / 4 * 16 * ...
			
			gen_loss =gen_loss.mean()
			"""
			gen_loss.backward()

			optim_gen.step()

		g_iter += 1

		if batch_idx % 100 == 0:

			last_t = now_t
			now_t = time.time()
			print("##############################")
			print('\n')
			print("iter : %6d ------- time: %4d of %6d Sec"%(g_iter,now_t - last_t,now_t - start_t))
			print('disc loss(avg):R{},F{},gen loss:{}'.format(loss_real.mean(),loss_fake.mean(), gen_loss.data.item()))

		if g_iter % 10000 == 0:
			#torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(g_iter)))
			torch.save(generator.state_dict(), os.path.join(ckp_mdf, 'gen_{}'.format(g_iter)))#args.checkpoint_dir, 'gen_{}'.format(g_iter)))
		#for scheduler_d in scheduler_ds:
			#scheduler_d.step()
		#scheduler_g.step()


fixed_z = Variable(torch.randn(args.batch_size, Z_dim).cuda())

def evaluate(epoch):

	samples = generator(fixed_z).cpu().data.numpy()[:64]


	fig = plt.figure(figsize=(8, 8))
	gs = gridspec.GridSpec(8, 8)
	gs.update(wspace=0.05, hspace=0.05)

	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.transpose((1,2,0)) * 0.5 + 0.5)

	if not os.path.exists('out/'):
		os.makedirs('out/')

	plt.savefig('out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
	plt.close(fig)
if not os.path.exists(ckp_mdf): #args.checkpoint_dir):
	os.makedirs(ckp_mdf) #args.checkpoint_dir)#, exist_ok=True)

start_t = time.time()
last_t = start_t
now_t = start_t

if __name__ == "__main__":
	for epoch in range(10000):
		train(epoch)
		print("EPOCH # %d"%epoch)
		if epoch %100 ==0:
			evaluate(epoch)
			fixed_z = Variable(torch.randn(args.batch_size, Z_dim).cuda())
			fake_images = generator(fixed_z)
			torchvision.utils.save_image(fake_images.data,os.path.join("./out__", '{}_fake.png'.format(epoch)),normalize=True,padding=0)


