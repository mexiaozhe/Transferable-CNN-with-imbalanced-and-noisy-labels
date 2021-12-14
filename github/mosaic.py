import os, sys
import argparse
from osgeo import gdal
import shutil
import cv2
import torch
import numpy as np
from torch.autograd import Variable
import scipy.misc as misc

def check_path(pathname):
    if not os.path.exists(pathname):
        os.makedirs(pathname)
        print(pathname + ' has been created!')
    else:
        shutil.rmtree(pathname)
        os.makedirs(pathname)
        print(pathname + ' has been reset!')

def generate_baselist(file_path, suffix):
    suffix_length = len(suffix)
    basename_list = []
    listfile = os.listdir(file_path)
    listfile.sort()
    for basename in listfile:
        if basename[(-suffix_length):] != suffix:
            continue
        basename_list.append(basename[:(-suffix_length)])
    return basename_list

def generate_list(file_path, basename_list, suffix):
    filename_list = []
    for basename in basename_list:
        filename = file_path + '/' + basename + suffix
        filename_list.append(filename)
    return filename_list

def generate_suffix(driver_name):
    suffix = '.null'
    if driver_name == 'GTiff':
        suffix = '.tif'
    elif driver_name == 'ENVI':
        suffix = '.dat'
    elif driver_name == 'HFA':
        suffix = '.img'
    elif driver_name == 'bmp':
        suffix = '.bmp'
    else:
        suffix = '.undefined'
    return suffix

def check_path(pathname):
    if not os.path.exists(pathname):
        os.makedirs(pathname)
        print(pathname + ' has been created!')

def mosaic_image(args):
    sub_size=args.sub_size 
    overlap = args.overlap
    datadir = args.file_path
    dirlist = os.listdir(datadir)
    dirlist.sort()
    orilist = generate_baselist(args.ori_img_path, args.ori_suffix) 
    orilist.sort()

    for idx in range(len(dirlist)):
        print(idx)
        file_path = os.path.join(datadir,dirlist[idx])
        basename_list = generate_baselist(file_path, args.suffix)
        filename_list = generate_list(file_path, basename_list, args.suffix)
        filename_list.sort()        
        ori_img_path = os.path.join(args.ori_img_path, file_path.split('/')[-1]+args.ori_suffix)
        ori = gdal.Open(ori_img_path)
        ori_proj = ori.GetProjection()
        ori_coor = ori.GetGeoTransform() 
        col = ori.RasterXSize
        row = ori.RasterYSize
        band = ori.RasterCount

        file_count = len(filename_list)
        i_row=0
        i_col=0
        for name_num in range(file_count):
            filename=filename_list[name_num]
            i_row_temp = int(filename.split('_')[-2])
            i_col_temp = int(filename.split('_')[-1].split('.')[0])
            if i_row_temp > i_row:
                i_row = i_row_temp
            if i_col_temp > i_col:
                i_col = i_col_temp
        name_base = os.path.join(file_path,dirlist[idx])
        pad_len_row = sub_size-overlap-(row-sub_size) % (sub_size-overlap)
        pad_len_col = sub_size-overlap-(col-sub_size) % (sub_size-overlap)
        out_img = np.zeros((row+sub_size+pad_len_row, col+sub_size+pad_len_col))
        for i in range(i_row+1):
            for k in range(i_col+1):
                filename = name_base + '_' + str(i) + '_' + str(k) +  args.suffix
                img = gdal.Open(filename)
                img_data = img.ReadAsArray()

                if i==0:
                    if k==0:
                        out_img[:img_data.shape[0], 0:img_data.shape[1]]=img_data
                    else:
                        out_img[:img_data.shape[0], k*(sub_size-overlap)+int(overlap/2):(k+1)*(sub_size-overlap)+overlap]=img_data[:,int(overlap/2):]
                elif k==0:  
                    out_img[i*(sub_size-overlap)+int(overlap/2):(i+1)*(sub_size-overlap)+overlap, 0:img_data.shape[1]]=img_data[int(overlap/2):, :]
                else:
                    out_img[i*(sub_size-overlap)+int(overlap/2):(i+1)*(sub_size-overlap)+overlap, k*(sub_size-overlap)+int(overlap/2):(k+1)*(sub_size-overlap)+overlap] = img_data[int(overlap/2):, int(overlap/2):]

        out_img = out_img[int(sub_size/2):out_img.shape[0]-pad_len_row-int(sub_size/2), int(sub_size/2):out_img.shape[1]-pad_len_col-int(sub_size/2)]
        out_img = out_img.reshape(1,out_img.shape[0], out_img.shape[1])

        colormap = readtxt(args.colortable)

        check_path(args.out_path)
        ct = gdal.ColorTable()
        color_count = len(colormap)
        for i in range(color_count):
            ct.SetColorEntry(i, colormap[i])
        out_filename = args.out_path + '/' + ori_img_path.split('/')[-1][:-len(args.ori_suffix)] + '.tif'
        datatype = gdal.GDT_Byte
        driver = 'GTiff'
        driver = gdal.GetDriverByName(driver)
        dataset = driver.Create(out_filename, out_img.shape[2], out_img.shape[1], out_img.shape[0], datatype)
        dataset.SetProjection(ori_proj)
        dataset.SetGeoTransform(ori_coor)
        dataset.GetRasterBand(1).SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
        dataset.GetRasterBand(1).SetColorTable(ct)
        dataset.GetRasterBand(1).WriteArray(out_img[0])
        del dataset

def readtxt(file):
    mean = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            mean.append(tuple([int(i) for i in line.strip().split('#')[0].split('/')]))
    return mean

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--file_path', nargs='?', type=str, default='../infer_ori_to_2005_256/',
                        help='path of images waiting to be mosaic')
    parser.add_argument('--suffix', nargs='?', type=str, default='.tif',
                        help='suffix')
    parser.add_argument('--ori_suffix', nargs='?',type=str, default = '.tif',
                        help='suffix of the orignal image')
    parser.add_argument('--ori_img_path', nargs='?', type=str, default='/opt/data2/2005_ori/',
                        help='path of images waiting to be mosaic')
    parser.add_argument('--colortable',nargs='?',type=str, default='/opt/data1/dataprepare/color_table/colortable_wbf_8.txt',
                        help='path of the colortable')
    parser.add_argument('--sub_size', nargs='?', type=int, default=1024,
                        help='size of sub_img')
    parser.add_argument('--overlap', nargs='?', type=int, default=256,
                        help='size to pad on the left and right sides of the image') 
    parser.add_argument('--out_path', nargs='?', type=str, default='../mosaic_ori_to_2005/',
                        help='path of images waiting to be mosaic')
    args = parser.parse_args()
    mosaic_image(args)