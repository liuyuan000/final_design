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

# fast-reid
parser = argparse.ArgumentParser(description="Feature extraction with reid models")
parser.add_argument("--config-file", default='configs/MSMT17/AGW_S50.yml', metavar="FILE", help="path to config file")
# parser.add_argument("--config-file", default='logs/market1501/agw_R50/config.yaml', metavar="FILE", help="path to config file")
parser.add_argument("--parallel",default=True,help='If use multiprocess for feature extraction.')
# datasets/query/f2tl64_142br264_536.jpg datasets/query/f2tl281_147br418_525.jpg datasets/query/f70tl506_176br632_521.jpg
parser.add_argument("--input",default='/home/liuyuan/app/百度网盘/MDT2018I004行人再识别视频采集&标注数据库-3摄像头/sample/行人3/picture',nargs="+",help="A list of space separated input images; ""or a single glob pattern such as 'directory/*.jpg'")
parser.add_argument("--distance",default=0.1,help='距离')
args = parser.parse_args()

# face
parser = argparse.ArgumentParser(description="模型参数设置")
parser.add_argument("--pic-dir", default='/home/liuyuan/app/百度网盘/MDT2018I004行人再识别视频采集&标注数据库-3摄像头/sample/行人3/', help="存放照片的文件夹")
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


    cap = cv2.VideoCapture(face_args.video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.set(cv2.CAP_PROP_POS_FRAMES, face_args.set_frame)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if face_args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(face_args.output, fourcc, fps, (width, height))

    retinaface = Retinaface()
    print('retinaface is already')

    if Config().backbone == 'resnet18':
        arcfacemodel = resnet_face18(Config().use_se)
    elif Config().backbone == 'resnet34':
        arcfacemodel = resnet34()
    elif Config().backbone == 'resnet50':
        arcfacemodel = resnet50()

    arcfacemodel = DataParallel(arcfacemodel)
    arcfacemodel.load_state_dict(torch.load(Config().test_model_path))
    arcfacemodel.to(torch.device("cuda"))
    arcfacemodel.eval()
    print('arcface is already')

    dst_pts = np.array([[66, 75], [128, 75], \
                                    [93, 90], [69, 130], \
                                    [122, 130]],dtype=np.float32)/192*150

    known_face_names, known_face_encodings = img_floder(face_args.pic_dir, retinaface, dst_pts, arcfacemodel)
    if len(known_face_names)!=0:
        print('发现以下人物:', ' ，'.join(known_face_names))
        process_face_flag = True
    else:
        process_face_flag =  False
        del arcfacemodel
        del retinaface
        print('没有发现人脸')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://'))

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    t0 = time.time()
    frame_num = 0
    xyxy_history= []
    flag = False
    label = 0
    confidence = []
    resize = 2
    pbar = tqdm(total=dataset.nframes)
    for path, img, im0s, vid_cap in dataset:
        frame_num += 1
        pbar.update(1)

        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        if frame_num == 1:
            imos_size = (int(im0s.shape[1]/resize), int(im0s.shape[0]/resize))

        im0s_reisze = cv2.resize(im0s, imos_size)
        if opt.frame_pass != 1 and frame_num%opt.frame_pass != 1:
            if view_img:
                if flag:
                    (success, boxes) = tracker.update(im0s_reisze)
                    (x, y, w, h) = [int(resize*v) for v in boxes]
                    plot_one_box([x,y,x+w,y+h], im0s, label=str(label), color=colors[0], line_thickness=2)
                cv2.imshow('result', im0s)
                cv2.waitKey(30)
            
            if save_img:
                vid_writer.write(im0s)
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
            save_path = str(save_dir / p.name)  # img.jpg
            os.makedirs(save_dir, exist_ok = True)
            s += '%gx%g ' % img.shape[2:]  # print string
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    if c.item() == 0:
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                images_tensor = None
                person_location = None
                person_location = []
                crop_image_flag = True
                for *xyxy, conf, cls in reversed(det):
                    if cls.item() == 0:
                        if yolo_show:
                            # label = f'{conf:.2f}'
                            plot_one_box(xyxy, im0, label=None, color=colors[1], line_thickness=2)
                        person_location.append(xyxy)
                        crop_image = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :]
                        crop_image = crop_image[:, :, ::-1]
                        crop_image = cv2.resize(crop_image, reid_img_size, interpolation=cv2.INTER_CUBIC)
                        crop_image = torch.as_tensor(crop_image.astype("float32").transpose(2, 0, 1))[None]
                        if crop_image_flag:
                            images_tensor = crop_image
                            crop_image_flag = False
                        else:
                            images_tensor = torch.cat((images_tensor, crop_image), 0)
                del crop_image
                images_tensor = images_tensor.cuda()
                feat = fast_reid.my_run_on_image(images_tensor)
                del images_tensor
                distmat, after_sort = distmat_calculte(test_feat, feat)
                reid_flag , xyxy, label = multi_rank(person_location, distmat, xyxy_history, distance=args.distance, frame_pass= opt.frame_pass)
                if reid_flag:
                    if process_face_flag:
                        crop_image = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :]
                        face_flag, feature, b1s = process(retinaface, crop_image, dst_pts, arcfacemodel)
                        if face_flag:
                            result = np.zeros((known_face_encodings.shape[0], feature.shape[0]))
                            match = np.zeros((known_face_encodings.shape[0], feature.shape[0]))
                            for i in range(known_face_encodings.shape[0]):
                                for j in range(feature.shape[0]):
                                    cv2.rectangle(crop_image, (b1s[j][0], b1s[j][1]), (b1s[j][2], b1s[j][3]), (255,0,0), 2)

                                    result[i][j] = cosin_metric(known_face_encodings[i], feature[j].T)

                                    if result[i][j] > face_args.distance:
                                        match[i][j] = 1

                            after_sort = np.argsort(-result, axis=0)

                            for i in range(after_sort.shape[1]):
                                if int(match[after_sort[0][i]][i])==1:
                                    cv2.putText(frame, known_face_names[after_sort[0][i]]+"%.2f"%(result[after_sort[0][i]][i]), (b1s[i][0], b1s[i][1]),\
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (20, 216, 48), thickness=2)

                        cv2.imshow('crop_image', crop_image)
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(im0s_reisze, (int(xyxy[0]/resize), int(xyxy[1]/resize), int((xyxy[2]-xyxy[0])/resize), int((xyxy[3]-xyxy[1])/resize)))
                    confidence.append([label])
                    if len(xyxy_history) > 5:
                        xyxy_history.pop(0)
                    area = (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1]))
                    center_x = 0.5*(int(xyxy[2]) + int(xyxy[0]))
                    center_y = 0.5*(int(xyxy[3]) + int(xyxy[1]))
                    xyxy_history.append([area, center_x, center_y])
                    s += 'Matched!!!'
                    plot_one_box(xyxy, im0, label=str(label), color=colors[0], line_thickness=2)
                else:
                    if len(xyxy_history) > 0:
                        xyxy_history.pop(0)


                if view_img:
                    cv2.imshow('result', im0)
                    if cv2.waitKey(3) == 27:
                        break

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            fourcc = 'mp4v'  # output video codec
                            
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        vid_writer.write(im0)

                print(s, f'Done. ({time.time() - t1:.3f}s)')

    print("总用时:", time.time()-t0, ' ', dataset.frame/(time.time()-t0), 'fps')
    print(confidence)
    print('平均值:', np.mean(confidence))
    print('方差：', np.var(confidence))
    print('标准差', np.std(confidence))
    if save_img:
        vid_writer.release()
