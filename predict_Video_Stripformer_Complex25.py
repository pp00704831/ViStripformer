import torch
from model.Video_Stripformer import Video_Stripformer
import torch.nn as nn
import cv2
import os
import time
import glob
import numpy as np
import math
import torchvision
cv2.setNumThreads(0)

# hyperparameters
frame_num = 9
data_path = './dataset/RainSynComplex25/test'
model_name = './weights/Derain/Video_Stripformer_Complex25.pth'
save_dir = './out/Video_Stripformer_RainSynComplex25'
print('Save at:', save_dir)
if not os.path.isdir('out'):
    os.mkdir('out')
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

# Model and optimizer
net = Video_Stripformer()
net.load_state_dict(torch.load(model_name))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

total_runtime = 0.
count = 0
clip_dir = [[0, 9], [7, 16], [14, 23], [21, 31]]
for _, video in enumerate(os.listdir(os.path.join(data_path, 'Rain'))):
    output_dir = os.path.join(save_dir, video)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    input_frames = sorted(glob.glob(os.path.join(data_path, 'Rain', video, "*")))

    video_length = len(input_frames)
    for idx, clip_idx in enumerate(clip_dir):
        if idx == 0 or idx == len(clip_dir) - 1:
            all = True
        else:
            all = False

        count += 1
        in_seq_dir = input_frames[clip_idx[0]:clip_idx[1]]

        in_seq = []
        for dir in in_seq_dir:
            image = cv2.imread(dir).astype(np.float32) / 255 - 0.5
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            H, W, C = np.shape(image)
            h = int(np.ceil(H/4)*4) - H
            w = int(np.ceil(W/4)*4) - W
            image = np.pad(image, ((0, h), (0, w), (0, 0)), mode='reflect').copy()
            in_seq.append(torch.from_numpy(image.transpose((2, 0, 1))))

        in_seq = torch.stack(in_seq, dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            out = net(in_seq, all).clamp(-0.5, 0.5)
            if idx == 0:
                out = out[:, :-1]
            if idx == len(clip_dir) - 1:
                out = out[:, 1:]
            torch.cuda.synchronize()
            stop = time.time()
            runtime = (stop-start) / out.shape[1]
            total_runtime += runtime

        name_list = []
        if idx == 0:
            name_start = clip_idx[0]
        else:
            name_start = clip_idx[0] + 1
        for i in range(out.shape[1]):
            out_name = os.path.split(input_frames[name_start + i])[-1]
            out_name = out_name.split('.')[0] + '.png'
            name_list.append(out_name)
            torchvision.utils.save_image(out[:, i, :, :H, :W] + 0.5, os.path.join(output_dir, out_name))

        print('Video:{} ({}), Frames:{} to {}, Runtime:{:.4f}, Avg Runtime:{:.4f}'.format(video, video_length, name_list[0], name_list[-1], runtime, total_runtime / count))

print('Save at:', save_dir)






