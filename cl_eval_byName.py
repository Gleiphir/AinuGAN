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
from torch.utils.data.sampler import RandomSampler
import argparse

Aparser = argparse.ArgumentParser()

Aparser.add_argument("model",type=str)

args = Aparser.parse_args()

BATCH_SIZE = 1

CL = Classifier().cuda()

dset = datasets.ImageFolder('/mnt/Dataset/JacobZh/AinuDset-ori', transform=transforms.Compose([
		transforms.RandomCrop(256),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

loader = torch.utils.data.DataLoader(dset,
	batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

print(dset.class_to_idx)

N_CLASS = len(dset.classes)

SAMPLES_FOR_EACH_IMG = 5


num_tests = 100
start_t=0
last_t=0
now_t=0

it_total = 0

Results = { x:[] for x in dset.imgs }

over_flag = False

if __name__ =='__main__':
	start_t = time.time()
	hit_total = 0
	samp_total = 0
	for nSample in range(SAMPLES_FOR_EACH_IMG):
		for it,(data, target) in enumerate(loader):
			if data.size()[0] != BATCH_SIZE:
				continue
			data, target = Variable(data.cuda()), Variable(target.cuda())

			# update discriminator
			d_losses = []

			# print(generator(z).size())
			losses = []

			CL.load_state_dict(torch.load(args.model))

			predict = CL(data)
			#print(target,data.size(1),data.size())
			last_t = now_t
			now_t = time.time()

			confid = predict[torch.arange(BATCH_SIZE),target]

			ground_truth = target.tolist()
			pred_res = torch.argmax(predict,dim=1).tolist()

			hit = 0
			assert len(ground_truth) == 1

			for i in range(len(ground_truth)):
				if ground_truth[i] == pred_res[i]:
					hit += 1

			Results[ dset.imgs[it] ].append(confid.tolist()[0])

			hit_total += hit
			samp_total += BATCH_SIZE

			if it_total %100 ==0:
				print("##############################")

				#print(torch.argmax(predict,dim=1))
				#print(confid)
				#print(confid.size())
				#print(predict[:,target].size())
				#print('\n')
				print("batch : %4d ------- time: %4d of %6d Sec" % (it_total, now_t - last_t, now_t - start_t))
				print('real:{},pred:{}\nconfid: {}\n{} of {} hit, {:.1f}%\n{} of {} hit, {:.1f}% in all '.format(
						ground_truth,
						pred_res,
						','.join( "{:.3f}".format(a) for a in  confid.tolist()),
						hit,
						BATCH_SIZE,
						hit / BATCH_SIZE * 100.0,
						hit_total,
						samp_total,
						hit_total /samp_total *100.0
					)
				)
			it_total += 1
	sortedKeys = sorted(Results.items(),key=lambda item:sum(Results[item]) )
	with open("./PredByNameRec.json") as fp:
		for key in sortedKeys:
			print(key[0].split("/")[-1],key[1],":",sum(Results[key]) / SAMPLES_FOR_EACH_IMG ,file=fp)
	print("wrote to file")