import numpy as np
import argparse
from osgeo import gdal
import os,sys

def generate_baselist(file_path, suffix):
    suffix_length = len(suffix)
    basename_list = []
    listfile = os.listdir(file_path)
    listfile.sort()
    for basename in listfile:
        if basename[(-suffix_length):] != suffix:
            continue
        basename_list.append(basename)
    return basename_list
def calc_weighted_average(weights, values, n_classes, with_background):
    mean_value = 0.0
    index_start = 0
    if with_background == 0:
        index_start = 1
    for i in range(index_start, n_classes):
        elem = weights[i] * values[i]
        mean_value += elem
    
    return mean_value

def f_measure(TP, FP, FN, f_beta):
    F_score_up = (1 + f_beta ** 2) * TP
    F_score_down = (1 + f_beta ** 2) * TP + (f_beta ** 2) * FN + FP
    F_score = np.true_divide(F_score_up, F_score_down)
    return F_score


pred_dir = '/opt/data2/study_area/shanxi2010dl/'
pred_suffix = '.tif'
gt_dir = '/opt/data2/study_area/gt/shanxigt2010/'
gt_suffix = '.tif'
accurate_file = os.path.join(pred_dir, pred_dir.split('/')[1]+ '_whole' +'.txt')
pred_list = generate_baselist(pred_dir, pred_suffix)
gt_list = generate_baselist(gt_dir, gt_suffix)


num = np.array([0, 1, 2, 3, 4, 5, 6, 7])
num = num.astype(int)
n_classes = len(num)
f_beta = 1
with_background=0
intersection = np.zeros(n_classes)
union = np.zeros(n_classes)
right = np.zeros(n_classes)
IoU = np.zeros(n_classes)

TP = np.zeros(n_classes)

FP = np.zeros(n_classes)
FN = np.zeros(n_classes)
F_score = np.zeros(n_classes)
weights = np.zeros(n_classes, dtype = np.float64)
real_elem_count = 0

for idx in range(len(gt_list)):
    print(idx)
    pred_name = os.path.join(pred_dir, pred_list[idx])
    pred = gdal.Open(pred_name)
    pred = pred.ReadAsArray()
    # import ipdb
    # ipdb.set_trace()
    gt_name = os.path.join(gt_dir, gt_list[idx])
    gt = gdal.Open(gt_name)
    gt = gt.ReadAsArray()

   
    for i in range(n_classes):
        weights[i] += np.sum(gt == num[i])
        gt_copy = gt.copy()
        pred_copy = pred.copy()

        gt_copy[gt != num[i]] = 0
        gt_copy[gt == num[i]] = 1
        pred_copy[pred != num[i]] = 0
        pred_copy[pred == num[i]] = 2
        gt_pred = gt_copy + pred_copy
        
        intersection[i] += np.sum(gt_pred == 3)
        union[i] += np.sum(gt_pred != 0)
        right[i] += np.sum(gt_copy == 1)

        TP[i] += np.sum(gt_pred == 3)
        FN[i] += np.sum(gt_pred == 1)
        FP[i] += np.sum(gt_pred == 2)

    if len(gt.shape)==3:
        elem_count = gt.shape[0] * gt.shape[1] * gt.shape[2]
    if len(gt.shape)==2:
        elem_count = gt.shape[0] * gt.shape[1]
    real_elem_count += float(elem_count)

acc = np.true_divide(intersection, right)
acc[np.isnan(acc)] = 1.0 
overall_acc = intersection.sum()/real_elem_count
if with_background == 0:
    real_elem_count -= weights[0]
    real_intersection = intersection[1:] 
    overall_acc = real_intersection.sum()/real_elem_count
weights /= real_elem_count

IoU = np.true_divide(intersection, union)
IoU[np.isnan(IoU)] = 1.0
mIoU = calc_weighted_average(weights, IoU, n_classes, with_background)

F_score = f_measure(TP, FP, FN, f_beta)
F_score[np.isnan(F_score)] = 1.0
mF_score = calc_weighted_average(weights, F_score, n_classes, with_background)


if with_background == 0:
    real_IoU = IoU[1:]
    real_F_score = F_score[1:]
else:
    real_IoU = IoU
    real_F_score = F_score

with open(accurate_file, 'a') as f:
    f.write('real_IoU: ' + str(real_IoU) + '\n')
    f.write('mIoU: ' + str(mIoU) + '\n')
    f.write('real_F_score: ' + str(real_F_score) + '\n')
    f.write('mF_score: ' + str(mF_score) + '\n')
    f.write('acc: ' + str(acc) + '\n')
    f.write('overall_acc:' + str(overall_acc) + '\n')
print(overall_acc)
