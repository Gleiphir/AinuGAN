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

BATCH_SIZE = 8

CL = Classifier().cuda()

dset = datasets.ImageFolder('/mnt/Dataset/JacobZh/AinuDset-ori', transform=transforms.Compose([
		transforms.RandomCrop(256),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

loader = torch.utils.data.DataLoader(dset,
	batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

print(dset.class_to_idx)

N_CLASS = len(dset.classes)

optim = optim.Adam( filter(lambda p: p.requires_grad, CL.parameters()),
	 lr=2e-4, betas=(0.5,0.999)
	)
g_iter=0
start_t=0
last_t=0
now_t=0

OVER_ITER = 100000

over_flag = False

if __name__ =='__main__':
	start_t = time.time()
	while not over_flag:
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
			#print(target,data.size(1),data.size())
			target_onehot = F.one_hot(target,num_classes=N_CLASS)

			#print(predict.size(),target_onehot.size())
			loss_fn = nn.BCELoss()
			loss = loss_fn(predict, target_onehot.to(torch.float))

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

				confid = predict.cpu().detach()[torch.LongTensor(target.cpu())]

				print("##############################")
				print(target)
				print(predict.cpu().detach())
				print(predict.cpu().detach()[target])
				print(predict.cpu().detach()[:,target])
				print(predict.cpu().detach()[target,:])
				#print(torch.argmax(predict,dim=1))
				#print(confid)
				#print(confid.size())
				#print(predict[:,target].size())
				print('\n')
				print("iter : %6d ------- time: %4d of %6d Sec" % (g_iter, now_t - last_t, now_t - start_t))
				print('real:{},pred:{},confid:{}\n,acc:{:.4f},loss:{:.6f}'.format(
					target.tolist(),
					torch.argmax(predict,dim=1).tolist(),
					[''.join(["{:.2f}".format(x) for x in predict[target].tolist()])],
					predict[target].mean().item(),
					loss.item())
				)
			if g_iter % 10000 == 0:
				# torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, 'disc_{}'.format(g_iter)))
				torch.save(CL.state_dict(),'clsf_{}'.format(g_iter))  # args.checkpoint_dir, 'gen_{}'.format(g_iter)))
			if g_iter >= OVER_ITER:
				over_flag = True
# for scheduler_d in scheduler_ds:
# scheduler_d.step()
# scheduler_g.step()