import os
import cv2
import numpy as np
import math
from scipy.ndimage import gaussian_filter

out_path = './out/Video_Stripformer_TCL'
gt_path = './dataset/tcl/test/target'

totalo_psnr = 0
total_ssim = 0
count = 0

def calc_psnr(result, gt):
    mse = np.mean(np.power((result / 255. - gt / 255.), 2))
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def calc_ssim(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):
    # Processing input image
    img1 = np.array(img1, dtype=np.float32) / 255
    img1 = img1.transpose((2, 0, 1))

    # Processing gt image
    img2 = np.array(img2, dtype=np.float32) / 255
    img2 = img2.transpose((2, 0, 1))

    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2

    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))

    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = ssim_num / ssim_den
    return np.mean(ssim_map)

for video in os.listdir(out_path):
    out_video_dir = os.path.join(out_path, video)
    gt_video_dir = os.path.join(gt_path, video)
    for idx, image in enumerate(os.listdir(out_video_dir)):
        count += 1
        name = image.split('.')[0] + '.jpg'
        out_image_dir = os.path.join(out_video_dir, image)
        gt_image_dir = os.path.join(gt_video_dir, name)
        image = cv2.imread(out_image_dir).astype(np.float32)
        gt_image = cv2.imread(gt_image_dir).astype(np.float32)

        psnr = calc_psnr(image, gt_image)
        ssim = calc_ssim(image, gt_image)
        totalo_psnr += psnr
        total_ssim += ssim
        print(count, totalo_psnr / count, total_ssim / count)
