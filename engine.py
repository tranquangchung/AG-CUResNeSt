import math
import sys
import time
import torch
import cv2
import utils
from evaluation import *
import os
from configs import *
import numpy as np
import copy
import torch.nn.functional as F


@torch.no_grad()
def evaluation_in_medical_cunet(model, data_loader, device, couple_unet=False):
    torch.set_num_threads(1)
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    acc, recall, precision, tnr, f1, js, dc, = 0,0,0,0,0,0,0 
    threshold = threshold_segmentation
    n_data = len(data_loader)
    tp_all, fn_all, fp_all, mean_f1, mean_iou = 0,0,0,0,0
    for images, targets, name_img in metric_logger.log_every(data_loader, 100, header):
        images = images.to(device)
        targets = targets.to(device)
        _, _, h_gt, w_gt = targets.shape
        model_time = time.time()
        outputs = model(images)
        model_time = time.time() - model_time

        evaluator_time = time.time()
        if couple_unet:
            predict = outputs["U2"]
            predict_U1 = outputs["U1"]
        else:
            predict = outputs["U1"]
        predict = F.upsample(predict, size=(h_gt, w_gt), mode='bilinear', align_corners=False)
        gt = targets
        tp_, fn_, fp_ = get_TP_FN_FP(predict, gt, threshold=threshold)
        tp_all += tp_
        fn_all += fn_
        fp_all += fp_
        acc_img = get_accuracy(predict, gt, threshold=threshold)
        recall_img = get_sensitivity(predict, gt, threshold=threshold)
        precision_img = get_precision(predict, gt, threshold=threshold)
        tnr_img = get_specificity(predict, gt, threshold=threshold)
        f1_img = get_F1(predict, gt, threshold=threshold)
        js_img = get_JS(predict, gt, threshold=threshold)
        dc_img = get_DC(predict, gt, threshold=threshold)

        mask_final = torch.sigmoid(predict).squeeze().cpu().detach().numpy()
        acc += acc_img
        recall += recall_img
        precision += precision_img
        tnr += tnr_img
        f1 += f1_img
        js += js_img
        dc += dc_img
        
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    recall_micro = tp_all/(tp_all+fn_all)
    precision_micro = tp_all/(tp_all+fp_all)
    f1_micro = (2*recall_micro*precision_micro)/(recall_micro+precision_micro)
    jaccard_micro = (recall_micro*precision_micro)/((recall_micro+precision_micro)-(recall_micro*precision_micro))

    print("n_data", n_data)
    print("recall_macro", recall/n_data)
    print("precision_macro", precision/n_data)
    print("f1(mDICE)_macro", f1/n_data)
    print("jaccard(mIOU)_macro", js/n_data)
    print("*"*20)
    print("recall_micro", recall_micro)
    print("precision_micro", precision_micro)
    print("f1(mDICE)_micro", f1_micro)
    print("jaccard(mIOU)_micro", jaccard_micro)
