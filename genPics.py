import model_cyclegan_dropout_shallow as model
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

generator = model.Generator(0.5).cuda()

batch_size = args.batch_size
#mdlPath = args.model
#mdlPath = "./gen_12000.wgh"


parseInt = lambda s : int( "".join([ x for x in s if x in "0123456789" ]) )


def gen(mdlPath):
    generator.load_state_dict(torch.load(mdlPath))
    now = datetime.now()
    date_time = now.strftime("%m-%d-%H-%M")
    print("date and time:",date_time)



    fixed_z = torch.randn(64, 256).cuda()

    fake_images = generator(fixed_z)
    torchvision.utils.save_image(fake_images.data, os.path.join("./out__", 'fake_{}_{}.png'.format(date_time,parseInt(mdlPath))), normalize=True,
                                 padding=0)


for i in range(2000,18000,2000):
    gen("./gen_{}.wgh".format(i))