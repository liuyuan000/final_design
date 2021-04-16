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


from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.utils.file_io import PathManager

from demo.predictor import FeatureExtractionDemo
# import some modules added in project like this below
# from projects.PartialReID.partialreid import *


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.MODEL.WEIGHTS = 'model/msmt_agw_S50.pth'
    # cfg.MODEL.WEIGHTS = 'model/market_agw_R50.pth'
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg



def init_fast(args):
    setup_logger(name="fastreid")
    args = args
    cfg = setup_cfg(args)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)
    return demo, tuple(cfg.INPUT.SIZE_TEST[::-1])

def distmat_calculte(test_feat, feat):
    '''
    >>> test_feat: 要搜索的人物的特征;
    >>> feat: 视频帧的特征;
    '''
    test_f = test_feat
    data_f = feat

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
    after_sort = np.argsort(distmat).astype(np.int32)
    return distmat, after_sort

def multi_rank(xyxy, distmat, xyxy_history, distance, frame_pass=1):
    distmat = distmat[distmat.min(axis=1).argmin()]
    if min(distmat) > distance:
        return False, None, None
    after_sort = np.argsort(distmat).astype(np.int32)
    area_history = 0
    center_history_x = 0
    center_history_y = 0
    totle_score = distmat * 50
    if len(xyxy_history) != 0:
        for i in range(len(xyxy_history)):
            area_history = xyxy_history[i][0] + area_history
            center_history_x = xyxy_history[i][1] + center_history_x
            center_history_y = xyxy_history[i][2] + center_history_y
        area_history = area_history / len(xyxy_history)
        center_history_x = center_history_x / len(xyxy_history)
        center_history_y = center_history_y / len(xyxy_history)

        for i in range(len(xyxy)):
            area = (int(xyxy[i][2]) - int(xyxy[i][0])) * (int(xyxy[i][3]) - int(xyxy[i][1]))
            center_x = 0.5*(int(xyxy[i][2]) + int(xyxy[i][0]))
            center_y = 0.5*(int(xyxy[i][3]) + int(xyxy[i][1]))
            totle_score[i] = totle_score[i] + abs(area - area_history)/area_history + abs(center_history_x- center_x)/center_history_x + \
                abs(center_history_y - center_y)/center_history_y

    after_sort = np.argsort(totle_score).astype(np.int32)
    if totle_score[after_sort[0]] > (4.5 + frame_pass/50):
        return False, None, None
    # print('distmat', min(distmat))
    # print('score:', totle_score[after_sort[0]])
    return True, xyxy[after_sort[0]], totle_score[after_sort[0]]
