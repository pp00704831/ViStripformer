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
frame_num = 16
data_path = './dataset/tcl/test'
model_name = './weights/Demoireing/Video_Stripformer_TCL.pth'
save_dir = './out/Video_Stripformer_TCL'
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
total_psnr = 0
for _, video in enumerate(os.listdir(os.path.join(data_path, 'source'))):
    output_dir = os.path.join(save_dir, video)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    input_frames = sorted(glob.glob(os.path.join(data_path, 'source', video, "*")))

    video_length = len(input_frames)
    end_idx = math.ceil((video_length - frame_num) / (frame_num - 2))
    if ((video_length - frame_num) % (frame_num - 2)) == 0:
        end_idx += 1

    clip_dir = []
    for idx in range(end_idx):
        start_idx = idx+(idx * (frame_num-3))
        clip_idx = [start_idx, start_idx+frame_num]
        clip_dir.append(clip_idx)

    if (video_length - frame_num) % (frame_num - 2) > 0:
        end_idx = [video_length - frame_num, video_length]
        clip_dir.append(end_idx)


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
            gt_start = 0
        else:
            name_start = clip_idx[0] + 1
            gt_start = 1
        for i in range(out.shape[1]):
            out_name = os.path.split(input_frames[name_start + i])[-1]
            out_name = out_name.split('.')[0] + '.jpg'
            name_list.append(out_name)
            torchvision.utils.save_image(out[:, i] + 0.5, os.path.join(output_dir, out_name))

        print('Video:{} ({}), Frames:{} to {}, Runtime:{:.4f}, Avg Runtime:{:.4f}'.format(video, video_length, name_list[0], name_list[-1], runtime, total_runtime / count))

print('Save at:', save_dir)






