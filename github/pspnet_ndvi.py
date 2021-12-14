import torch as t
import numpy as np
import torch.nn as nn

from math import ceil
from torch.autograd import Variable
from torch.nn import functional as F

class pspnet(nn.Module):

    def __init__(self, n_classes=11):
        super(pspnet, self).__init__()
        self.block_config=t.Tensor([3, 4, 6, 3]).int()
        self.psp_pool=t.Tensor([8,6,3,2]).int()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, dilation=1, bias=False) #, trackrunningstats=True
        self.bn1 = nn.BatchNorm2d(64, momentum=0.05)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(in_channels=64, mid_channels=64, 
                                        block_config=self.block_config[0], stride=1, padding=1, dilation=1)
        self.layer2 = self._make_layer(in_channels=128, mid_channels=256, 
                                        block_config=self.block_config[1], stride=2, padding=1, dilation=1)
        self.layer3 = self._make_layer(in_channels=256, mid_channels=512, 
                                        block_config=self.block_config[2], stride=1, padding=2, dilation=2)
        self.layer4 = self._make_layer(in_channels=512, mid_channels=1024, 
                                        block_config=self.block_config[3], stride=1, padding=4, dilation=4)

        self.pyramid_pooling = pyramidPooling(2048, self.psp_pool)
        self.cbr_1 = nn.Sequential(
                nn.Conv2d(4096, 512, 3, 1, 1, 1, 1, False),
                nn.BatchNorm2d(512, momentum=0.05),
                nn.ReLU()
                )
        self.cbr_2 = nn.Sequential(
                nn.Conv2d(513, 256, 3, 1, 1, 1, 1, False),
                nn.BatchNorm2d(256, momentum=0.05),
                nn.ReLU()
                )
        self.cbr_3 = nn.Sequential(
                nn.Conv2d(256, 128, 3, 1, 1, 1, 1, False),
                nn.BatchNorm2d(128, momentum=0.05),
                nn.ReLU()
                )
                
        # Final conv layers
        self.dropout = nn.Dropout2d(p=0.1,inplace=True)
        self.classification = nn.Conv2d(128, n_classes, 1, 1, 0, 1)


        #NDVI、NDBI、NDWI
        self.block1 = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=5,stride=4,padding=2))
        
        self.block2 = nn.Sequential(nn.Conv2d(64, 256, 3, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),)

        self.block3 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

        self.block4 = nn.Sequential(nn.Conv2d(512,1024,3,padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(inplace=True))

        self.block5 = nn.Sequential(nn.Conv2d(1024, 2048, 3, padding=1),
                                    nn.BatchNorm2d(2048),
                                    nn.ReLU(inplace=True))  
        self.ndvi_pool = nn.MaxPool2d(8,8)

    def _make_layer(self, in_channels, mid_channels, block_config, stride=1, padding=0, dilation=1):
        downsample = None
        if stride != 1 or mid_channels != in_channels * 4:
            downsample = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels * 4,
                          kernel_size=1, stride=stride, padding=0, bias=False, dilation=1),
                nn.BatchNorm2d(in_channels * 4, momentum=0.05),
            )
        layers = []
        layers.append(Bottleneck(mid_channels, in_channels, stride, downsample, dilation=1))
        for i in range(1, block_config):
            layers.append(Bottleneck(in_channels*4, in_channels, padding=0, dilation=1))
        return nn.Sequential(*layers)


    def forward(self, x):
        inp_shape = x.shape[2:]
        img = np.zeros((x.shape[0],3,x.shape[2],x.shape[3]))
        x0 = x.cpu().numpy()
        img[:,0,:,:] = x0[:,3,:,:]
        img[:,1,:,:] = x0[:,4,:,:]
        img[:,2,:,:] = x0[:,0,:,:]
        
        ndvi = (x0[:,3,:,:] - x0[:,2,:,:]) / (x0[:,3,:,:] + x0[:,2,:,:])
        ndvi = t.tensor(ndvi.reshape(1, ndvi.shape[0], ndvi.shape[1], ndvi.shape[2])).cuda().transpose(0,1)
       
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pyramid_pooling(x)
        x = self.cbr_1(x)
        x = F.upsample(x, size=inp_shape, mode='bilinear', align_corners=False)
        x = t.cat([x,ndvi],1)
        x = self.cbr_2(x)
        x = self.maxpool(x)
        x = self.cbr_3(x)
        x = self.dropout(x)
        x = self.classification(x)
        
        x = F.upsample(x, size=inp_shape, mode='bilinear', align_corners=False)
        return x

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, padding=0, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.05)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.05)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, momentum=0.05)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class pyramidPooling(nn.Module):

    def __init__(self, in_channels, pool_sizes):
        super(pyramidPooling, self).__init__()

        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(conv2DBatchNormRelu(in_channels, int(in_channels / len(pool_sizes)), 1, 1, 0, bias=False))

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes

    def forward(self, x):
        output_slices = [x]
        h, w = x.shape[2:]

        for module, pool_size in zip(self.path_module_list, self.pool_sizes):
            pool_size = pool_size.float()
            out = F.avg_pool2d(x, int(h/pool_size), int(h/pool_size), 0)
            out = module(out)
            out = F.upsample(out, size=(h,w), mode='bilinear', align_corners=False)
            output_slices.append(out)

        return t.cat(output_slices, dim=1)


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1):
        super(conv2DBatchNormRelu, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        self.cbr_unit = nn.Sequential(conv_mod,
                                      nn.BatchNorm2d(int(n_filters), momentum=0.05),
                                      nn.ReLU(),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


# For Testing Purposes only
if __name__ == '__main__':
    cd = 0
    from torch.autograd import Variable
    import matplotlib.pyplot as plt
    import scipy.misc as m
    from ptsemseg.loader.cityscapes_loader import cityscapesLoader as cl
    psp = pspnet(version='ade20k')

    # Just need to do this one time
    psp.load_pretrained_model(model_path='/home/meet/models/pspnet50_ADE20K.caffemodel')

    psp.float()
    psp.cuda(cd)
    psp.eval()

    dst = cl(root='/home/meet/datasets/cityscapes/')
    img = m.imread('/home/meet/seg/leftImg8bit/demoVideo/stuttgart_00/stuttgart_00_000000_000010_leftImg8bit.png')
    m.imsave('cropped.png', img)
    orig_size = img.shape[:-1]
    img = img.transpose(2, 0, 1)
    img = img.astype(np.float64)
    img -= np.array([123.68, 116.779, 103.939])[:, None, None]
    img = np.copy(img[::-1, :, :])
    flp = np.copy(img[:, :, ::-1])

    out = psp.tile_predict(img)
    pred = np.argmax(out, axis=0)
    m.imsave('ade20k_sttutgart_tiled.png', pred)

    torch.save(psp.state_dict(), "psp_ade20k.pth")
    print("Output Shape {} \t Input Shape {}".format(out.shape, img.shape))
