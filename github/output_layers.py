import torch 
import torch.nn.functional as F
import argparse
import os, sys
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm 
import torchvision.models as models
import shutil
import loss_function as LossFunc
import RSData as RSD
import pspnet_dl as net_model
import evaluate as Eval
import scipy.misc as misc
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

def train(args):
    mean = readtxt(os.path.join(args.train_dir,'mean_value.txt'))
    std = readtxt(os.path.join(args.train_dir,'std_value.txt'))
    t_file = RSD.RSData(args.img_dir, mode='test',suffix=args.suffix, mean=mean,std=std,test_base=args.img_dir)
    t_loader = data.DataLoader(t_file, batch_size=args.batch_size, num_workers=8)
    model = net_model.pspnet(args.n_classes)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state'])    
    model.cuda()

    

    for i, img in tqdm(enumerate(t_loader)):
        with torch.no_grad():
            img = img.cuda()
            ndvi,ndwi,ndbi,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,y1,y2,y3,y4,y5,y6,y7,y8,y9  = model(img)
            import ipdb
            ipdb.set_trace()
            step_size = 99
            
            norm_ndvi =  torch.round((ndvi-torch.min(ndvi))/(torch.max(ndvi)-torch.min(ndvi))*step_size)
            freq = torch.zeros(step_size).cuda()
            for index in range(step_size):
                freq[index] = torch.sum(norm_ndvi==index)
            freq = freq / torch.sum(freq)
            entropy_ndvi = -torch.sum(freq*torch.log(freq+0.0001))
            
            norm_ndwi =  torch.round((ndwi-torch.min(ndwi))/(torch.max(ndwi)-torch.min(ndwi))*step_size)
            freq = torch.zeros(step_size).cuda()
            for index in range(step_size):
                freq[index] = torch.sum(norm_ndwi==index)
            freq = freq / torch.sum(freq)
            entropy_ndwi = -torch.sum(freq*torch.log(freq+0.0001))
            
            norm_ndbi =  torch.round((ndbi-torch.min(ndbi))/(torch.max(ndbi)-torch.min(ndbi))*step_size)
            freq = torch.zeros(step_size).cuda()
            for index in range(step_size):
                freq[index] = torch.sum(norm_ndbi==index)
            freq = freq / torch.sum(freq)
            entropy_ndbi = -torch.sum(freq*torch.log(freq+0.0001))
            
            
            
            
            norm_x1 =  torch.round((x1-torch.min(x1))/(torch.max(x1)-torch.min(x1))*step_size)
            freq = torch.zeros(step_size).cuda()
            for index in range(step_size):
                freq[index] = torch.sum(norm_x1==index)
            freq = freq / torch.sum(freq)
            entropy_x1 = -torch.sum(freq*torch.log(freq+0.0001))

            norm_x2 =  torch.round((x2-torch.min(x2))/(torch.max(x2)-torch.min(x2))*step_size)
            freq = torch.zeros(step_size).cuda()
            for index in range(step_size):
                freq[index] = torch.sum(norm_x2==index)
            freq = freq / torch.sum(freq)
            entropy_x2 = -torch.sum(freq*torch.log(freq+0.0001))
            
            norm_x3 =  torch.round((x3-torch.min(x3))/(torch.max(x3)-torch.min(x3))*step_size)
            freq = torch.zeros(step_size).cuda()
            for index in range(step_size):
                freq[index] = torch.sum(norm_x3==index)
            freq = freq / torch.sum(freq)
            entropy_x3 = -torch.sum(freq*torch.log(freq+0.0001))
            
            norm_x4 =  torch.round((x4-torch.min(x4))/(torch.max(x4)-torch.min(x4))*step_size)
            freq = torch.zeros(step_size).cuda()
            for index in range(step_size):
                freq[index] = torch.sum(norm_x4==index)
            freq = freq / torch.sum(freq)
            entropy_x4 = -torch.sum(freq*torch.log(freq+0.0001))
            
            norm_x5 =  torch.round((x5-torch.min(x5))/(torch.max(x5)-torch.min(x5))*step_size)
            freq = torch.zeros(step_size).cuda()
            for index in range(step_size):
                freq[index] = torch.sum(norm_x5==index)
            freq = freq / torch.sum(freq)
            entropy_x5 = -torch.sum(freq*torch.log(freq+0.0001))
            
            norm_x6 =  torch.round((x6-torch.min(x6))/(torch.max(x6)-torch.min(x6))*step_size)
            freq = torch.zeros(step_size).cuda()
            for index in range(step_size):
                freq[index] = torch.sum(norm_x6==index)
            freq = freq / torch.sum(freq)
            entropy_x6 = -torch.sum(freq*torch.log(freq+0.0001))
            
            norm_x7 =  torch.round((x7-torch.min(x7))/(torch.max(x7)-torch.min(x7))*step_size)
            freq = torch.zeros(step_size).cuda()
            for index in range(step_size):
                freq[index] = torch.sum(norm_x7==index)
            freq = freq / torch.sum(freq)
            entropy_x7 = -torch.sum(freq*torch.log(freq+0.0001))
            
            norm_x8 =  torch.round((x8-torch.min(x8))/(torch.max(x8)-torch.min(x8))*step_size)
            freq = torch.zeros(step_size).cuda()
            for index in range(step_size):
                freq[index] = torch.sum(norm_x8==index)
            freq = freq / torch.sum(freq)
            entropy_x8 = -torch.sum(freq*torch.log(freq+0.0001))
            
            norm_x9 =  torch.round((x9-torch.min(x9))/(torch.max(x9)-torch.min(x9))*step_size)
            freq = torch.zeros(step_size).cuda()
            for index in range(step_size):
                freq[index] = torch.sum(norm_x9==index)
            freq = freq / torch.sum(freq)
            entropy_x9 = -torch.sum(freq*torch.log(freq+0.0001))
            
            norm_x10 =  torch.round((x10-torch.min(x10))/(torch.max(x10)-torch.min(x10))*step_size)
            freq = torch.zeros(step_size).cuda()
            for index in range(step_size):
                freq[index] = torch.sum(norm_x10==index)
            freq = freq / torch.sum(freq)
            entropy_x10 = -torch.sum(freq*torch.log(freq+0.0001))
            
            norm_x11 =  torch.round((x11-torch.min(x1))/(torch.max(x11)-torch.min(x11))*step_size)
            freq = torch.zeros(step_size).cuda()
            for index in range(step_size):
                freq[index] = torch.sum(norm_x11==index)
            freq = freq / torch.sum(freq)
            entropy_x11 = -torch.sum(freq*torch.log(freq+0.0001))             
            
            
            

            temp_path = os.path.join(args.out_path,'x1')
            check_path(temp_path)
            layer_data = x1.data.cpu().numpy()
            for k in range(x1.shape[1]):
                name = 'x1' + str(k) + '.png'
                name = os.path.join(temp_path, name)
                out_img = layer_data[0,k,:,:]
                plt.imsave(name,out_img,cmap = 'hot')

            temp_path = os.path.join(args.out_path,'x2')
            check_path(temp_path)
            layer_data = x2.data.cpu().numpy()
            for k in range(x2.shape[1]):
                name = 'x2' + str(k) + '.png'
                name = os.path.join(temp_path, name)
                out_img = layer_data[0,k,:,:]
                plt.imsave(name,out_img,cmap = 'hot')
            name = 'infer' + '.png'
            name = os.path.join(temp_path, name)
            pred = x2.data.max(1)[1].cpu().numpy()
            misc.imsave(name, pred[0,:,:])

            temp_path = os.path.join(args.out_path,'x3')
            check_path(temp_path)
            layer_data = x3.data.cpu().numpy()
            for k in range(x3.shape[1]):
                name = 'x3' + str(k) + '.png'
                name = os.path.join(temp_path, name)
                out_img = layer_data[0,k,:,:]
                plt.imsave(name,out_img,cmap = 'hot')

            temp_path = os.path.join(args.out_path,'x4')
            check_path(temp_path)
            layer_data = x4.data.cpu().numpy()
            for k in range(x4.shape[1]):
                name = 'x4' + str(k) + '.png'
                name = os.path.join(temp_path, name)
                out_img = layer_data[0,k,:,:]
                plt.imsave(name,out_img,cmap = 'hot')
            name = 'infer' + '.png'
            name = os.path.join(temp_path, name)
            pred = x4.data.max(1)[1].cpu().numpy()
            misc.imsave(name, pred[0,:,:])

            temp_path = os.path.join(args.out_path,'x5')
            check_path(temp_path)
            layer_data = x5.data.cpu().numpy()
            for k in range(x5.shape[1]):
                name = 'x5' + str(k) + '.png'
                name = os.path.join(temp_path, name)
                out_img = layer_data[0,k,:,:]
                plt.imsave(name,out_img,cmap = 'hot')
            name = 'infer' + '.png'
            name = os.path.join(temp_path, name)
            pred = x5.data.max(1)[1].cpu().numpy()
            misc.imsave(name, pred[0,:,:])

            temp_path = os.path.join(args.out_path,'x6')
            check_path(temp_path)
            layer_data = x6.data.cpu().numpy()
            for k in range(x6.shape[1]):
                name = 'x6' + str(k) + '.png'
                name = os.path.join(temp_path, name)
                out_img = layer_data[0,k,:,:]
                plt.imsave(name,out_img,cmap = 'hot')
            name = 'infer' + '.png'
            name = os.path.join(temp_path, name)
            pred = x6.data.max(1)[1].cpu().numpy()
            misc.imsave(name, pred[0,:,:])

            temp_path = os.path.join(args.out_path,'x7')
            check_path(temp_path)
            layer_data = x7.data.cpu().numpy()
            for k in range(x7.shape[1]):
                name = 'x7' + str(k) + '.png'
                name = os.path.join(temp_path, name)
                out_img = layer_data[0,k,:,:]
                plt.imsave(name,out_img,cmap = 'hot')

            temp_path = os.path.join(args.out_path,'x8')
            check_path(temp_path)
            layer_data =x8.data.cpu().numpy()
            for k in range(x8.shape[1]):
                name = 'x8' + str(k) + '.png'
                name = os.path.join(temp_path, name)
                out_img = layer_data[0,k,:,:]
                plt.imsave(name,out_img,cmap = 'hot')
            name = 'infer' + '.png'
            name = os.path.join(temp_path, name)
            pred = x8.data.max(1)[1].cpu().numpy()
            misc.imsave(name, pred[0,:,:])

            temp_path = os.path.join(args.out_path,'x9')
            check_path(temp_path)
            layer_data = x9.data.cpu().numpy()
            for k in range(x9.shape[1]):
                name = 'x9' + str(k) + '.png'
                name = os.path.join(temp_path, name)
                out_img = layer_data[0,k,:,:]
                plt.imsave(name,out_img,cmap = 'hot')

            temp_path = os.path.join(args.out_path,'x10')
            check_path(temp_path)
            layer_data = x10.data.cpu().numpy()
            for k in range(x10.shape[1]):
                name = 'x103' + str(k) + '.png'
                name = os.path.join(temp_path, name)
                out_img = layer_data[0,k,:,:]
                plt.imsave(name,out_img,cmap = 'hot')
            name = 'infer' + '.png'
            name = os.path.join(temp_path, name)
            pred = x10.data.max(1)[1].cpu().numpy()
            misc.imsave(name, pred[0,:,:])

            temp_path = os.path.join(args.out_path,'x11')
            check_path(temp_path)
            layer_data = x11.data.cpu().numpy()
            for k in range(x11.shape[1]):
                name = 'x11' + str(k) + '.png'
                name = os.path.join(temp_path, name)
                out_img = layer_data[0,k,:,:]
                plt.imsave(name,out_img,cmap = 'hot')

           

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
    parser.add_argument('--train_dir',nargs='?',type=str, default='/opt/data1/jingjinji/for_samples/samples_8classes_50/',
                        help='dir of training set')
    parser.add_argument('--img_dir', nargs='?', type=str, default='/data/zxm/NDVI/50_psp/6bands/test_img/',
                        help='image dir')
    parser.add_argument('--suffix', nargs='?', type=str, default='.tif',
                        help='suffix')
    parser.add_argument('--batch_size', nargs='?', type=int, default=1,
                        help='Batch Size')
    parser.add_argument('--model_path', nargs='?', type=str, default='./dl/iter_best.pkl', 
                        help='Path to previous saved model to restart from')
    parser.add_argument('--out_path', nargs='?', type=str, default='/data/zxm/NDVI/50_psp/6bands/test_img_out/',
                        help='File to save the out images')

    args = parser.parse_args()
    train(args)