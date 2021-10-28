import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision
from model_category import Classifier
import time
import os

BATCH_SIZE = 1

CL = Classifier().cuda()

loader = torch.utils.data.DataLoader(
datasets.ImageFolder('/mnt/Dataset/JacobZh/AinuDset-ori', transform=transforms.Compose([
		transforms.RandomCrop(256),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
	batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)


optim = optim.Adam( filter(lambda p: p.requires_grad, CL.parameters()),
	 lr=2e-4, betas=(0.5,0.999)
	)
g_iter=0
start_t=0
last_t=0
now_t=0

if __name__ =='__main__':

	for batch_idx, (data, target) in enumerate(loader):
		if data.size()[0] != BATCH_SIZE:
			continue
		data, target = Variable(data.cuda()), Variable(target.cuda())

		# update discriminator
		d_losses = []

		optim.zero_grad()
		# print(generator(z).size())
		losses = []

		predict = CL(data)

		loss = nn.functional.cross_entropy(predict,target)

		"""
		gen_loss = torch.cat(losses).mean()# 2 * 32 *... / 4 * 16 * ...

		gen_loss =gen_loss.mean()
		"""
		loss.backward()

		optim.step()

		g_iter += 1

		if batch_idx % 100 == 0:
			last_t = now_t
			now_t = time.time()
			print("##############################")
			print('\n')
			print("iter : %6d ------- time: %4d of %6d Sec" % (g_iter, now_t - last_t, now_t - start_t))
			print('real:{},fake:{},loss:{}'.format(target.item(),predict.item(),loss.item()))
			if g_iter % 10000 == 0:
				# torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(g_iter)))
				torch.save(CL.state_dict(),
						   os.path.join(os.getcwd(),
										'gen_{}'.format(g_iter)))  # args.checkpoint_dir, 'gen_{}'.format(g_iter)))

# for scheduler_d in scheduler_ds:
# scheduler_d.step()
# scheduler_g.step()