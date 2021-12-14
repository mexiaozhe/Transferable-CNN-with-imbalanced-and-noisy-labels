import numpy as np
import argparse
from osgeo import gdal

def evaluate(gt_img, pred_img, n_classes):
    with_background = 1
    f_beta = 1
    intersection = np.zeros(n_classes)
    union = np.zeros(n_classes)
    right = np.zeros(n_classes)
    IoU = np.zeros(n_classes)

    TP = np.zeros(n_classes)
    #TN = np.zeros(n_classes)
    FP = np.zeros(n_classes)
    FN = np.zeros(n_classes)
    F_score = np.zeros(n_classes)
    weights = np.zeros(n_classes, dtype = np.float64)

    for i in range(n_classes):
        weights[i] = np.sum(gt_img == i)
        gt_copy = gt_img.copy()
        pred_copy = pred_img.copy()

        gt_copy[gt_img != i] = 0
        gt_copy[gt_img == i] = 1
        pred_copy[pred_img != i] = 0
        pred_copy[pred_img == i] = 2
        gt_pred = gt_copy + pred_copy
        
        intersection[i] = np.sum(gt_pred == 3)
        union[i] = np.sum(gt_pred != 0)
        right[i] = np.sum(gt_copy == 1)

        TP[i] = np.sum(gt_pred == 3)
        FN[i] = np.sum(gt_pred == 1)
        FP[i] = np.sum(gt_pred == 2)
        #TN[i] = np.sum(gt_pred == 0)
    if len(gt_img.shape)==3:
        elem_count = gt_img.shape[0] * gt_img.shape[1] * gt_img.shape[2]
    if len(gt_img.shape)==2:
        elem_count = gt_img.shape[0] * gt_img.shape[1]
    real_elem_count = float(elem_count)
    
    acc = np.true_divide(intersection, right)
    acc[np.isnan(acc)] = 1.0 

    if with_background == 0:
        real_elem_count -= weights[0]
        overall_acc = ((gt_img==pred_img).sum()-weights[0])/real_elem_count
    weights /= real_elem_count

    IoU = np.true_divide(intersection, union)
    IoU[np.isnan(IoU)] = 1.0
    mIoU = calc_weighted_average(weights, IoU, n_classes, with_background)

    F_score = f_measure(TP, FP, FN, f_beta)
    F_score[np.isnan(F_score)] = 1.0
    mF_score = calc_weighted_average(weights, F_score, n_classes, with_background)
    overall_acc = (gt_img==pred_img).sum()/real_elem_count
    
    if with_background == 0:
        real_IoU = IoU[1:]
        real_F_score = F_score[1:]
    else:
        real_IoU = IoU
        real_F_score = F_score
    
    return real_IoU, mIoU, real_F_score, mF_score, acc, overall_acc

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

def read(args):
    gt_img = gdal.Open(args.gt)
    gt_img = gt_img.ReadAsArray()
    pred_img = gdal.Open(args.pred)
    pred_img = pred_img.ReadAsArray()
    return gt_img, pred_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--n_classes', nargs='?', type=int, default=6,
                        help='Number of classes')
    parser.add_argument('gt', nargs='?', type=str, default='/opt/data0/hebei/GT/hebei_3_img.tif',
                        help='ground_truth')
    parser.add_argument('pred', nargs='?', type=str, default='./out_hebei/hebei_3_img_copy.tif',
                        help='inferenced image')
    parser.add_argument('accurate_file', nargs='?', type=str, default='./out_hebei/accurate_file.txt',
                        help='accurate_file')
    args = parser.parse_args()

    gt_img, pred_img = read(args)
    real_IoU, mIoU, real_F_score, mF_score, acc, overall_acc = evaluate(gt_img, pred_img, args.n_classes)
    print('real_IoU: ' + real_IoU)
    print('mIoU: ' + mIoU)
    print('real_F_score: ' + real_F_score)
    print('mF_score: ' + mF_score)
    print('acc: ' + acc)
    print('overall_acc:' + overall_acc)
    with open(args.accurate_file, 'a') as f:
        f.write('epoch ' + str(epoch + 1) + ' loss = ' + str(epoch_loss/i/iteration) + '\n')
        f.write('real_IoU: ' + str(real_IoU_b/i_val) + '\n')
        f.write('mIoU: ' + str(mIoU_b/i_val) + '\n')
        f.write('real_F_score: ' + str(real_F_score_b/i_val) + '\n')
        f.write('mF_score: ' + str(mF_score_b/i_val) + '\n')
        f.write('acc: ' + str(acc_b/i_val) + '\n')
        f.write('overall_acc:' + str(overall_acc_b/i_val) + '\n')