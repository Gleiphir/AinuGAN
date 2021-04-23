import model_cyclegan as model
import torch
import torchvision
import argparse
from datetime import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--model', type=str)
args = parser.parse_args()

generator = model.Generator().cuda()

batch_size = args.batch_size
#mdlPath = args.model
mdlPath = "./gen_18000"
generator.load_state_dict(torch.load(mdlPath))

now = datetime.now()
date_time = now.strftime("%m-%d-%H-%M")
print("date and time:",date_time)



fixed_z = torch.randn(64, 128).cuda()
fake_images = generator(fixed_z)
torchvision.utils.save_image(fake_images.data, os.path.join("./out__", 'fake_{}.png'.format(date_time)), normalize=True,
                             padding=0)
