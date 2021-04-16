from __future__ import absolute_import

import argparse
import datetime
import os
import os.path as osp
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from IPython import embed
from numpy import random
from torch import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from tqdm import tqdm

from demo.predictor import FeatureExtractionDemo
from face_models.resnet import *
from face_utils.facealign import (Config, cosin_metric, getTransMatrix,
                                  img_floder, load_image, process)
# sys.path.append('/home/liuyuan/final_design/yolo_fast_reid/fastreid')
from fastreid.config import get_cfg
from init import distmat_calculte, init_fast, multi_rank
from models.experimental import attempt_load
from retinaface import Retinaface
from utils.datasets import LoadImages, LoadStreams
from utils.general import (check_img_size, check_imshow, increment_path,
                           non_max_suppression, scale_coords, set_logging,
                           xyxy2xywh)
from utils.plots import plot_one_box
from utils.torch_utils import load_classifier, select_device

# yolo
parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='model/yolov5s.pt', help='model.pt path(s)')
# /home/liuyuan/final_design/目标检测/video/MOT16-08-raw.mp4  /home/liuyuan/app/百度网盘/MDT2018I004行人再识别视频采集&标注数据库-3摄像头/sample/行人6/video
parser.add_argument('--source', type=str, default='/home/liuyuan/app/百度网盘/MDT2018I004行人再识别视频采集&标注数据库-3摄像头/sample/行人3/video', help='source')  # file/folder, 0 for webcam
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS') # 0.4
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', type=bool, default=True, help='display results')
parser.add_argument('--save-vedio', type=bool, default=False, help='存储视频')
parser.add_argument('--person-path', type=str, default='/home/liuyuan/final_design/目标检测/video/yolov5/', help='存储每个人的图片的位置')
parser.add_argument('--save-conf', type=bool, default=True, help='save confidences in --save-txt labels')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--frame-pass', type = int, default=10, help='隔帧检测')
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--yolo-show', type=bool, default=True)
opt = parser.parse_args()
print(opt)
# fast-reid
parser = argparse.ArgumentParser(description="Feature extraction with reid models")
parser.add_argument("--config-file", default='configs/MSMT17/AGW_S50.yml', metavar="FILE", help="path to config file")
# parser.add_argument("--config-file", default='logs/market1501/agw_R50/config.yaml', metavar="FILE", help="path to config file")
parser.add_argument("--parallel",default=True,help='If use multiprocess for feature extraction.')
# datasets/query/f2tl64_142br264_536.jpg datasets/query/f2tl281_147br418_525.jpg datasets/query/f70tl506_176br632_521.jpg
parser.add_argument("--input",default='/home/liuyuan/app/百度网盘/MDT2018I004行人再识别视频采集&标注数据库-3摄像头/sample/行人3/picture',nargs="+",help="A list of space separated input images; ""or a single glob pattern such as 'directory/*.jpg'")
parser.add_argument("--distance",default=0.1,help='距离')
args = parser.parse_args()
print(args)
# face
parser = argparse.ArgumentParser(description="模型参数设置")
parser.add_argument("--pic-dir", default='test/', help="存放照片的文件夹")
parser.add_argument("--video", default=0, help="视频地址")
parser.add_argument('--save-video',type=bool, default=False, help='是否保存视频')
parser.add_argument('--output', default='my_work/output2.mp4', help='输出的视频文件')
parser.add_argument("--frame-pass", type=int, default=1,help='跳帧检测')
parser.add_argument("--set-frame", type=int, default=0,help='设置开始的帧数')
parser.add_argument("--distance",default=0.5,help='距离阈值')
face_args = parser.parse_args()

if __name__ == '__main__':
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True

    print('init fast reid')
    fast_reid, reid_img_size = init_fast(args)

    img_formats = ['.jpg', '.jpeg', '.png', '.tif']

    p = Path(args.input)
    images=p.rglob('*.*')
    images=[x for x in images if str(x)[-4:] in img_formats]
    img_paths=[str(x) for x in images]   # 得到所有图片路径组成
    assert len(img_paths)>0, '文件夹里没有图片'
    print('共发现{}张图片'.format(len(img_paths)))
    test_feat = None
    test_flag = True
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]
        img = cv2.resize(img, reid_img_size, interpolation=cv2.INTER_CUBIC)
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))[None]
        feat = fast_reid.my_run_on_image(img)
        if test_flag:
            test_feat = feat
            test_flag = False
        else:
            test_feat = torch.cat((test_feat, feat), dim=0)
    del img
    print('fast_reid is already')

    source, weights, view_img, imgsz, save_img, yolo_show = opt.source, opt.weights, opt.view_img, opt.img_size, opt.save_vedio, opt.yolo_show

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
