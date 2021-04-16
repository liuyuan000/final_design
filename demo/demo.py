# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import tqdm
import torch
from torch.backends import cudnn

sys.path.append('/home/liuyuan/final_design/fast-reid')

from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.utils.file_io import PathManager

from predictor import FeatureExtractionDemo
# import some modules added in project like this below
# from projects.PartialReID.partialreid import *

cudnn.benchmark = True

setup_logger(name="fastreid")


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.MODEL.WEIGHTS = 'model/market_agw_R50.pth'
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        default='logs/market1501/agw_R50/config.yaml',
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--parallel",
        default=True,
        # action='store_true',
        help='If use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--input",
        default='datasets/query/',
        nargs="+",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='logs/my_vis',
        help='path to save features'
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)

    PathManager.mkdirs(args.output)
    if args.input:
        if PathManager.isdir(args.input[0]):
            # args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        paths = glob.glob(os.path.join(args.input, '*.jpg'))

        i = 0
        for path in tqdm.tqdm(paths):
            img = cv2.imread(path)
            img = img[:, :, ::-1]
            img = cv2.resize(img, tuple(cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_CUBIC)
            img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))[None]
            if i == 0:
                images = img
                i = 1
            else:
                images = torch.cat((images, img), 0)
        begin_time = time.time()
        feats_flag = True
        print(images.shape)
        batch_size = 15
        for i in range(0, int(images.shape[0]/batch_size)):
            images_batch = images[i*batch_size:(i+1)*batch_size, ...]
            feat = demo.my_run_on_image(images_batch)
            if feats_flag:
                feats = feat
                feats_flag = False
            else:
                feats = torch.cat((feats, feat), dim=0)
        if images.shape[0]%batch_size != 0:
            images_batch = images[(i+1)*batch_size:, ...]
            feat = demo.my_run_on_image(images_batch)
            if feats_flag:
                feats = feat
                feats_flag = False
            else:
                feats = torch.cat((feats, feat), dim=0)
        del images
        test_f = feats[0:1,...]
        data_f = feats
        del feats
        test_f = 1. * test_f / (torch.norm(test_f, 2, dim = -1, keepdim=True).expand_as(test_f) + 1e-12)
        data_f = 1. * data_f / (torch.norm(data_f, 2, dim = -1, keepdim=True).expand_as(data_f) + 1e-12)
        m, n = test_f.size(0), data_f.size(0)
        """
        计算query特征与gallery特征的距离矩阵
        全局特征计算欧氏距离，矩阵A,B欧氏距离等于√(A^2 + (B^T)^2 - 2A(B^T))
        局部特征使用局部对齐最小距离算法计算距离
        """
        # 计算A^2 + (B^T)^2
        distmat = torch.pow(test_f, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                torch.pow(data_f, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # #计算A^2 + (B^T)^2 - 2A(B^T)
        distmat.addmm_(test_f, data_f.t(), beta=1, alpha=-2)
        distmat = distmat.numpy()
        distmat = np.argsort(distmat)
        print(distmat)
        # feats = feats.numpy()
        # np.save(os.path.join(args.output, path.replace('.jpg', '.npy').split('/')[-1]), feats)
        print('speed:', len(paths)/(time.time()-begin_time), ' fps')
