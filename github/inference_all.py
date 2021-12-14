import torch 
import argparse
import os, sys
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm 
import torchvision.models as models
import shutil
import loss_function as LossFunc
import RSData as RSD
import pspnet_ori as net_model
import evaluate as Eval
import scipy.misc as misc
import numpy as np
import cv2

def train(args):
    datadir = args.datadir
    dirlist = os.listdir(datadir)
    dirlist.sort()

    for idx in range(len(dirlist)):
        testdir = os.path.join(datadir, dirlist[idx])
        mean = readtxt(os.path.join(args.traindir,'mean_value.txt'))
        std = readtxt(os.path.join(args.traindir,'std_value.txt'))
        t_file = RSD.RSData(args.datadir, mode='test',suffix=args.suffix, mean=mean,std=std,test_base=testdir)
        t_loader = data.DataLoader(t_file, batch_size=args.batch_size, num_workers=8)
        model = net_model.pspnet(args.n_classes)
        model = torch.nn.DataParallel(model, device_ids=args.gpu_id)
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state'])    
        model.cuda(args.gpu_id[0])
        model.eval()
        torch.cuda.empty_cache()
        out_path = os.path.join(args.out_path,dirlist[idx])
        check_path(out_path)

        for i, img in tqdm(enumerate(t_loader)):
            with torch.no_grad():
                img = img.cuda(args.gpu_id[0])
                outputs = model(img)
                pred = outputs.data.max(1)[1].cpu().numpy()

                print('Classes found: ', np.unique(pred))
                for j in range(pred.shape[0]):
                    k = i*args.batch_size + j
                    name = t_file.images['test'][k].split('/')[-1]
                    cv2.imwrite(os.path.join(out_path, name), pred[j,:,:])
                    # misc.imsave(os.path.join(out_path, name[:-4] +'.png'), pred[j,:,:])
                    print('Classified image saved at: {}'.format(out_path + '/' + name))            
def readtxt(file):
    mean = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            mean.append(float(line))
    return mean

def check_path(pathname):
    if not os.path.exists(pathname):
        os.makedirs(pathname)
        print(pathname + ' has been created!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--n_classes', nargs='?', type=int, default=8,
                        help='Number of classes')
    parser.add_argument('--gpu_id',nargs='?', type=int, default = [0,1,2,3],
                        help='ID of GPU')
    parser.add_argument('--traindir',nargs = '?', type=str, default='/opt/data1/jingjinji/for_samples/samples_8classes_50/',
                        help='dir of trainig samples')
    parser.add_argument('--datadir', nargs='?', type=str, default='/opt/data2/2005_clip_256/',
                        help='datadir')
    parser.add_argument('--suffix', nargs='?', type=str, default='.tif',
                        help='suffix')
    parser.add_argument('--batch_size', nargs='?', type=int, default=12,
                        help='Batch Size')
    parser.add_argument('--model_path', nargs='?', type=str, default='./ori/iter_best.pkl', 
                        help='Path to previous saved model to restart from')
    parser.add_argument('--out_path', nargs='?', type=str, default='../infer_ori_to_2005_256/',
                        help='File to save the out images')
    args = parser.parse_args()
    train(args)