from __future__ import absolute_import

import argparse
import glob
import os
import os.path as osp
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from cv2 import cv2
from numpy import random
from torch import torch
from torch.backends import cudnn
from tqdm import tqdm

from demo.predictor import FeatureExtractionDemo
from init import distmat_calculte, init_fast, multi_rank

sys.path.append('/home/liuyuan/final_design/yolo_fast_reid/fastreid')
from fastreid.config import get_cfg
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import (check_img_size, check_imshow, increment_path,
                           non_max_suppression, scale_coords, set_logging,
                           xyxy2xywh)
from utils.plots import plot_one_box
from utils.torch_utils import load_classifier, select_device

# yolo
parser = argparse.ArgumentParser(description="Feature extraction with yolo models")
parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
# /home/liuyuan/final_design/目标检测/video/MOT16-08-raw.mp4
# /home/liuyuan/app/百度网盘/MDT2018I004行人再识别视频采集&标注数据库-3摄像头/sample/行人1/video/192.168.1.63_01_20180314140123215_1_000138_000146.mp4
parser.add_argument('--source', type=str, default='/home/liuyuan/final_design/目标检测/video/行人检测测试视频.mp4', help='source')  # file/folder, 0 for webcam
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', type=bool, default=True, help='display results')
parser.add_argument('--save-vedio', type=bool, default=False, help='存储视频')
parser.add_argument('--save-imgs', type=bool, default=True, help='存储每个人的图片')
parser.add_argument('--person-path', type=str, default='/home/liuyuan/final_design/目标检测/video/yolov5/', help='存储每个人的图片的位置')
parser.add_argument('--save-conf', type=bool, default=True, help='save confidences in --save-txt labels')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--frame-pass', type = int, default=10, help='隔帧检测')
parser.add_argument('--save-dir', type=str, default='logs')
parser.add_argument('--yolo-show', type=bool, default=True)
opt = parser.parse_args()
# print(opt)
# fast-reid
parser = argparse.ArgumentParser(description="Feature extraction with reid models")
parser.add_argument("--config-file", default='configs/MSMT17/AGW_S50.yml', metavar="FILE", help="path to config file")
# parser.add_argument("--config-file", default='logs/market1501/agw_R50/config.yaml', metavar="FILE", help="path to config file")
parser.add_argument("--parallel",default=True,help='If use multiprocess for feature extraction.')
# datasets/query/f2tl64_142br264_536.jpg datasets/query/f2tl281_147br418_525.jpg datasets/query/f70tl506_176br632_521.jpg
parser.add_argument("--input",default='/home/liuyuan/final_design/yolo_fast_reid/datasets/query/5.png',nargs="+",help="A list of space separated input images; ""or a single glob pattern such as 'directory/*.jpg'")
parser.add_argument("--output",default='logs/my_vis',help='path to save features')
parser.add_argument("--distance",default=0.2,help='距离')
args = parser.parse_args()
# print(args)

if __name__ == '__main__':
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    print('init fast reid')
    fast_reid, reid_img_size = init_fast(args)
    img = cv2.imread(args.input)
    img = img[:, :, ::-1]
    img = cv2.resize(img, reid_img_size, interpolation=cv2.INTER_CUBIC)
    img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))[None]
    test_feat = fast_reid.my_run_on_image(img)
    del img
    print('fast_reid is already')

    source, weights, view_img, imgsz, save_img, yolo_show = opt.source, opt.weights, opt.view_img, opt.img_size, opt.save_vedio, opt.yolo_show
    # if opt.save_imgs:
    #     from utils.my_save_imgs import save_local_person
    #     os.makedirs(opt.person_path, exist_ok = True)

    # Initialize
    set_logging()
    device = select_device(opt.device)

    # Load model
    yolomodel = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(yolomodel.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    # Get names and colors
    names = yolomodel.module.names if hasattr(yolomodel, 'module') else yolomodel.names
    colors = [[0, 255, 0], [0, 0, 255]]
    # Run inference
    if device.type != 'cpu':
        yolomodel(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(yolomodel.parameters())))  # run once
    yolomodel.eval()
    print('yolo is already')

    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    save_dir = opt.save_dir + '/np'
    if osp.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok = True)
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    t0 = time.time()
    frame_num = 0
    pbar = tqdm(total=dataset.nframes)
    for path, img, im0s, vid_cap in dataset:
        frame_num += 1
        pbar.update(1)

        fps = vid_cap.get(cv2.CAP_PROP_FPS)

        if opt.frame_pass != 1 and frame_num%opt.frame_pass != 1:

            continue
        t1 = time.time()
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = yolomodel(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            s += '%gx%g ' % img.shape[2:]  # print string
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                person_location = None
                crop_image_flag = True
                image_numpy = None
                for *xyxy, conf, cls in reversed(det):
                    if cls.item() == 0:
                        if yolo_show:
                            # label = f'{conf:.2f}'
                            plot_one_box(xyxy, im0, label=None, color=colors[1], line_thickness=2)
                        crop_image = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :]
                        crop_image = crop_image[:, :, ::-1]
                        crop_image = cv2.resize(crop_image, reid_img_size, interpolation=cv2.INTER_CUBIC)
                        crop_image = torch.as_tensor(crop_image.astype("float32").transpose(2, 0, 1))[None]
                        xyxy = [[int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]]


                        if crop_image_flag:
                            images_tensor = crop_image
                            person_location = xyxy
                            crop_image_flag = False
                        else:
                            person_location = np.r_[person_location, xyxy]
                            images_tensor = torch.cat((images_tensor, crop_image), 0)
                del crop_image
                images_tensor = images_tensor.cuda()
                feat = fast_reid.my_run_on_image(images_tensor)
                feat = feat.cpu().numpy()
                image_numpy = np.hstack((person_location,feat))
                np.save(save_dir + '/{}'.format(frame_num), image_numpy)
    pbar.close()
    del yolomodel
    del fast_reid
    frame_find = np.zeros((frame_num))
    distance_mini = 100
    distmat_mini = np.array([0, 0, 0, 0, 0]) # distance frame location
    cap = cv2.VideoCapture(source)
    fourcc = 'mp4v'
    save_path = opt.save_dir + '/result1.mp4'
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (200, 512))
    pbar = tqdm(total=frame_num)
    for i in range(1, frame_num, opt.frame_pass):
        pbar.update(opt.frame_pass)
        feature = np.load(save_dir + '/{}.npy'.format(i))
        feat_tensor = torch.tensor(feature[:, 4:].astype("float32"))
        distmat, after_sort = distmat_calculte(test_feat, feat_tensor)
        if distmat[0, after_sort[0, 0]] < 0.08:
            cap.set(cv2.CAP_PROP_POS_FRAMES,i-1)
            ret, frame = cap.read()
            if ret:
                frame = frame[int(feature[after_sort[0,0], 1]): int(feature[after_sort[0,0], 3]), int(feature[after_sort[0,0], 0]):int(feature[after_sort[0,0], 2]), :]
                frame = cv2.resize(frame, (200, 512), interpolation=cv2.INTER_NEAREST)
                cv2.putText(frame, str(distmat[0, after_sort[0, 0]]), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), thickness=2)
                vid_writer.write(frame)
                cv2.imshow('result', frame)
                cv2.waitKey(30)
        if distmat[0, after_sort[0, 0]] < distance_mini:
            distance_mini = distmat[0, after_sort[0, 0]]
            distmat_mini[0] = i - 1
            distmat_mini[1] = int(feature[after_sort[0,0], 1])
            distmat_mini[2] = int(feature[after_sort[0,0], 3])
            distmat_mini[3] = int(feature[after_sort[0,0], 0])
            distmat_mini[4] = int(feature[after_sort[0,0], 2])
    pbar.close()
    vid_writer.release()
    print(distance_mini, distmat_mini)
    print(frame_num / (time.time()-t0), 'fps, time:', (time.time()-t0))
    cap.set(cv2.CAP_PROP_POS_FRAMES,distmat_mini[0])
    ret, frame = cap.read()
    frame = frame[distmat_mini[1]:distmat_mini[2], distmat_mini[3]:distmat_mini[4],:]
    cv2.imshow('mini', frame)
    cv2.waitKey(10000)
            # print(distmat[0, after_sort[0, 0]], i, feature[after_sort[0,0], 0:4])
    shutil.rmtree(save_dir)
    cv2.destroyAllWindows()
