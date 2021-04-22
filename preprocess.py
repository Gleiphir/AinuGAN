from PIL import Image
from pathlib import Path
import os
raw_folder = Path(r"D:\Github\AinuGAN\ainuDset\Full Size Imgs\Ifuku")

output_folder = Path(r"D:\Github\AinuGAN\myDset-ifuku")

def centerCrop(im:Image,size):
    new_width,new_height = size[0],size[1]
    width, height = im.size   # Get dimensions

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    # Crop the center of the image
    return im.crop((left, top, right, bottom))


for fname in raw_folder.iterdir():
    new_name = os.path.splitext(fname.name)[0] + ".png"
    print(fname.absolute(),"->",new_name)

    im = Image.open(fname.absolute())
    im_cutted = centerCrop(im,(1488,1488))
    #im_cutted.show()
    im_resized = im_cutted.resize((256, 256), Image.BICUBIC)
    im_resized.save(output_folder / new_name )

