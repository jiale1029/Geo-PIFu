import argparse
import os
import lpips
import cv2
import numpy as np
import math
from tqdm import tqdm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, default='./imgs/ex_dir0')
parser.add_argument('-d1','--dir1', type=str, default='./imgs/ex_dir1')
parser.add_argument('-o','--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('-n', '--network', type=str, default="geopifu")
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
   
opt = parser.parse_args()
   
## Initializing the model
loss_fn = lpips.LPIPS(net='alex',version=opt.version)
if(opt.use_gpu):
    loss_fn.cuda()
   
# crawl directories
f = open(opt.out,'w')
files = sorted(os.listdir(opt.dir0))

# assert(len(files) == 21736*4)
# assert(len(os.listdir(opt.dir1)) == 21736*4)
print("dir0: ", len(files))
print("dir1: ", len(os.listdir(opt.dir1)))

def psnr(img1, img2):
    mse = np.mean( (img1-img2) **2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX/math.sqrt(mse))

total = 0
total_psnr = 0
count = 0
   
for file in tqdm(files):
    if "pix2pix" in opt.network:
        # output is in file1
        # gt is in file 2
        if "HD" in opt.network:
            file_2 = file.replace("gt_image", "synthesized_image")
            if file_2 == file:
                file_2 = file.replace("synthesized_image", "gt_image")
        else:
            file_2 = file.replace("fake", "real")
        if(os.path.exists(os.path.join(opt.dir1,file_2))):
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,file_2)))

            img_0 = cv2.imread(os.path.join(opt.dir0, file))
            img_1 = cv2.imread(os.path.join(opt.dir1, file_2))

            if(opt.use_gpu):
                img0 = img0.cuda()
                img1 = img1.cuda()

            # Compute distance
            dist01 = loss_fn.forward(img0,img1)
            d = psnr(img_0, img_1)
            total_psnr += d
            total += dist01.detach().cpu()
            count += 1
            print('%s,%s: %.3f'%(file,file_2,dist01))
            f.writelines('%s: %.6f, %.6f\n'%(file,dist01,d))
    else:
        if(os.path.exists(os.path.join(opt.dir1,file))):
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,file)))

            img_0 = cv2.imread(os.path.join(opt.dir0, file))
            img_1 = cv2.imread(os.path.join(opt.dir1, file))

            if(opt.use_gpu):
                img0 = img0.cuda()
                img1 = img1.cuda()

            # Compute distance
            dist01 = loss_fn.forward(img0,img1)
            d = psnr(img_0, img_1)
            total_psnr += d
            total += dist01.detach().cpu()
            count += 1
            print('%s: %.3f'%(file,dist01))
            f.writelines('%s: %.6f, %.6f\n'%(file,dist01,d))

print(f"Total: {total}")
print(f"Count: {count}")
print(f"Average perceptual similarity: {total/count}")
print(f"Average PSNR: {total_psnr/count}")
f.writelines(f"Total: {total}\n")
f.writelines(f"Count: {count}\n")
f.writelines(f"Average perceptual similarity: {total/count}\n")
f.writelines(f"Average PSNR: {total_psnr/count}")

f.close()
