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
import random


BATCH_SIZE = 8

CL = Classifier().cuda()

dset = datasets.ImageFolder('/mnt/Dataset/JacobZh/AinuDset-ori', transform=transforms.Compose([
		transforms.RandomCrop(256),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

loader = torch.utils.data.DataLoader(dset,
	batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

print(dset.class_to_idx)

N_CLASS = len(dset.classes)


num_tests = 1000
start_t=0
last_t=0
now_t=0

OVER_ITER = 100000

rand_idx = random.shuffle([x for x in range(len(loader))])[num_tests]



over_flag = False

if __name__ =='__main__':
	start_t = time.time()
	while not over_flag:
		for t,idx in enumerate(rand_idx):
			(data, target) = loader[idx]
			if data.size()[0] != BATCH_SIZE:
				continue
			data, target = Variable(data.cuda()), Variable(target.cuda())

			# update discriminator
			d_losses = []

			# print(generator(z).size())
			losses = []

			CL.load_state_dict(torch.load('clsf_{}'.format(40000)))

			predict = CL(data)
			#print(target,data.size(1),data.size())
			last_t = now_t
			now_t = time.time()

			confid = predict[torch.arange(BATCH_SIZE),target]

			ground_truth = target.tolist()
			pred_res = torch.argmax(predict,dim=1).tolist()

			hit = 0
			for i in range(len(ground_truth)):
				if ground_truth[i] == pred_res[i]:
					hit += 1

			print("##############################")

			#print(torch.argmax(predict,dim=1))
			#print(confid)
			#print(confid.size())
			#print(predict[:,target].size())
			#print('\n')
			print("batch : %4d ------- time: %4d of %6d Sec" % (t, now_t - last_t, now_t - start_t))
			print('real:{},pred:{}\nconfid: {}\n{} of {} hit, {:.1f}%'.format(
					ground_truth,
					pred_res,
					','.join( "{:.3f}".format(a) for a in  confid.tolist()),
					hit,
					BATCH_SIZE,
					hit / BATCH_SIZE * 100.0
				)
			)

# for scheduler_d in scheduler_ds:
# scheduler_d.step()
# scheduler_g.step()