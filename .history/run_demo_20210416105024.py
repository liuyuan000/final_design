import argparse
import time

import cv2
import numpy as np
from torch import torch
from torch.nn import DataParallel
from tqdm import tqdm

from face_models.resnet import *
from retinaface import Retinaface
from face_utils.facealign import (Config, cosin_metric, getTransMatrix, img_floder,
                             load_image, process)

parser = argparse.ArgumentParser(description="模型参数设置")
parser.add_argument("--pic-dir", default='test/', help="存放照片的文件夹")
parser.add_argument("--video", default=0, help="视频地址")
parser.add_argument('--save-video',type=bool, default=False, help='是否保存视频')
parser.add_argument('--output', default='my_work/output2.mp4', help='输出的视频文件')
parser.add_argument("--frame-pass", type=int, default=1,help='跳帧检测')
parser.add_argument("--set-frame", type=int, default=0,help='设置开始的帧数')
parser.add_argument("--distance",default=0.5,help='距离阈值')
face_args = parser.parse_args()

# 'test/“机场吐槽达人”特朗普.mp4'
cap = cv2.VideoCapture(face_args.video)
fps = int(cap.get(cv2.CAP_PROP_FPS))
cap.set(cv2.CAP_PROP_POS_FRAMES, face_args.set_frame)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

if face_args.save_video:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(face_args.output, fourcc, fps, (width, height))

retinaface = Retinaface()

opt = Config()
if opt.backbone == 'resnet18':
    model = resnet_face18(opt.use_se)
elif opt.backbone == 'resnet34':
    model = resnet34()
elif opt.backbone == 'resnet50':
    model = resnet50()

model = DataParallel(model)
model.load_state_dict(torch.load(opt.test_model_path))
model.to(torch.device("cuda"))
model.eval()

dst_pts = np.array([[66, 75], [128, 75], \
                                [93, 90], [69, 130], \
                                [122, 130]],dtype=np.float32)/192*150


known_face_names, known_face_encodings = img_floder(face_args.pic_dir, retinaface, dst_pts, model)
print('发现以下人物:', ' ，'.join(known_face_names))

pbar = tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT)-face_args.set_frame)
frame_pass = face_args.frame_pass
frame_num = 0
t0 = time.time()
while True:
    ret, frame = cap.read()
    frame_num += 1
    pbar.update(1)
    if frame_num % frame_pass != 0:
        continue
    if ret:
        flag, feature, b1s = process(retinaface, frame,  dst_pts, model)
        if flag:
            result = np.zeros((known_face_encodings.shape[0], feature.shape[0]))
            match = np.zeros((known_face_encodings.shape[0], feature.shape[0]))
            for i in range(known_face_encodings.shape[0]):
                for j in range(feature.shape[0]):
                    cv2.rectangle(frame, (b1s[j][0], b1s[j][1]), (b1s[j][2], b1s[j][3]), (255,0,0), 2)

                    result[i][j] = cosin_metric(known_face_encodings[i], feature[j].T)

                    if result[i][j] > face_args.distance:
                        match[i][j] = 1

            after_sort = np.argsort(-result, axis=0)

            for i in range(after_sort.shape[1]):
                if int(match[after_sort[0][i]][i])==1:
                    cv2.putText(frame, known_face_names[after_sort[0][i]]+"%.2f"%(result[after_sort[0][i]][i]), (b1s[i][0], b1s[i][1]),\
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (20, 216, 48), thickness=2)


    else:
        break
    cv2.imshow('result', frame)
    if face_args.save_video:
        out.write(frame)
    if cv2.waitKey(3) == 27:
        break
if face_args.save_video:
    out.release()
cap.release()
pbar.close()
cv2.destroyAllWindows()
print((frame_num/frame_pass)/(time.time()-t0), 'fps')
