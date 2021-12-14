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
import pspnet_pfirst as net_model
import evaluate as Eval
import numpy as np

def train(args):
    weight = torch.ones(args.n_classes)
    weight[0] = 0
    weight = weight.cuda(device=args.gpu_id[0])
    mean = readtxt(os.path.join(args.datadir,'mean_value.txt'))
    std = readtxt(os.path.join(args.datadir,'std_value.txt'))
    t_file = RSD.RSData(args.datadir, mode='train',suffix=args.suffix, mean=mean,std=std)
    v_file = RSD.RSData(args.datadir, mode='validation',suffix=args.suffix, mean=mean,std=std)
    trainloader = data.DataLoader(t_file, batch_size=args.batch_size, num_workers=8, shuffle=True)
    valloader = data.DataLoader(v_file, batch_size=args.batch_size, num_workers=8)
    # Setup Model
    resnet50 = models.resnet50(pretrained=True)
    resnet50_names = []
    for name, param in resnet50.named_parameters():
        resnet50_names.append(name)
    conv1_weight = resnet50.state_dict()[resnet50_names[0]].numpy()
    for i in range(conv1_weight.shape[1]+1,args.n_bands+1):
        conv1_weight = np.append(conv1_weight, conv1_weight[:,i%3:i%3+1,:,:], axis=1)
    model = net_model.pspnet(args.n_classes)
    model_names = []
    for name, pram in model.named_parameters():
        model_names.append(name)   
    for name, pram in resnet50.named_parameters():
        if name == resnet50_names[0]:
            model_dict = model.state_dict()[model_names[0]].data.copy_(torch.from_numpy(conv1_weight))
        elif name in model.state_dict():
            model_dict = model.state_dict()[name].data.copy_(pram)
    
    for i in range(len(model_names)-17,len(model_names)):
        if '0.weight' in model_names[i]:
            param = torch.empty(model.state_dict()[model_names[i]].shape)
            torch.nn.init.xavier_normal_(param)
            model.state_dict()[model_names[i]].data.copy_(param)
    model.cuda(device = args.gpu_id[0])
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)
    


    optimizer = torch.optim.Adam(model.parameters(), lr=args.l_rate, weight_decay=5e-4)
    loss_model = LossFunc.cross_entropy(weight)
    pre_epoch = 0
    mIoU_e = 0
    overall_acc_e = 0    
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            pre_epoch = checkpoint['epoch']-1
        else:
            print("No checkpoint found at '{}'".format(args.resume))
    check_path(args.model_path)
    for epoch in range(pre_epoch, args.n_epoch+1):
        adjust_lr(optimizer,epoch)
        model.train() 
        epoch_loss = 0.0
        iteration = 0
        overall_best = 0.0
        print('epoch:' + str(epoch))
        model.train()
        for i, (images, labels) in tqdm(enumerate(trainloader)):
            images = images.cuda(args.gpu_id[0])
            labels = labels.cuda(args.gpu_id[0])
            optimizer.zero_grad()
            outputs = model(images)  
            loss = loss_model(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if os.path.exists(args.debug_file):
                import ipdb
                ipdb.set_trace()

        real_IoU_b = 0
        mIoU_b = 0
        real_F_score_b = 0
        mF_score_b = 0
        acc_b = 0
        overall_acc_b = 0
        model.eval()
        for i_val, (img_val, lbl_val) in tqdm(enumerate(valloader)):
            with torch.no_grad():
                img_val = img_val.cuda(args.gpu_id[0])
                lbl_val = lbl_val.cuda(args.gpu_id[0])
                outputs = model(img_val)
                pred = outputs.data.max(1)[1].cpu().numpy()
                gt = lbl_val.data.cpu().numpy()

                real_IoU, mIoU, real_F_score, mF_score, acc, overall_acc = Eval.evaluate(gt, pred, args.n_classes)
                real_IoU_b += real_IoU
                mIoU_b += mIoU
                real_F_score_b += real_F_score
                mF_score_b += mF_score
                acc_b += acc
                overall_acc_b += overall_acc 
        iteration += 1  
        i=i+1
        i_val=i_val+1
        weight = 1/acc

        print('epoch ' + str(epoch) + ' loss = ' + str(epoch_loss/i/iteration) + '\n')
        print('real_IoU: ' + str(real_IoU_b/i_val) + '\n')
        print('mIoU: ' + str(mIoU_b/i_val) + '\n')
        print('real_F_score: ' + str(real_F_score_b/i_val) + '\n')
        print('mF_score: ' + str(mF_score_b/i_val) + '\n')
        print('acc: ' + str(acc_b/i_val) + '\n')
        print('overall_acc:' + str(overall_acc_b/i_val) + '\n')
        with open(args.accurate_file, 'a') as f:
            f.write('epoch ' + str(epoch + 1) + ' loss = ' + str(epoch_loss/i/iteration) + '\n')
            f.write('real_IoU: ' + str(real_IoU_b/i_val) + '\n')
            f.write('mIoU: ' + str(mIoU_b/i_val) + '\n')
            f.write('real_F_score: ' + str(real_F_score_b/i_val) + '\n')
            f.write('mF_score: ' + str(mF_score_b/i_val) + '\n')
            f.write('acc: ' + str(acc_b/i_val) + '\n')
            f.write('overall_acc:' + str(overall_acc_b/i_val) + '\n')
        print('Writing Result File: ', args.accurate_file)

        if (epoch) % 5 ==0:
            state = {'epoch': epoch+1,
                'model_state': model.state_dict(),
                'optimizer_state' : optimizer.state_dict()}
            torch.save(state, args.model_path+'iter_{}.pkl'.format(epoch))
        if overall_best < overall_acc_b/i_val:
            state = {'epoch': epoch+1,
                'model_state': model.state_dict(),
                'optimizer_state' : optimizer.state_dict()}
            torch.save(state, args.model_path+'iter_best.pkl')
            overall_best = overall_acc_b/i_val

def adjust_lr(optimizer, epoch):
    if epoch < 20:
        lr = args.l_rate * 0.01
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if 20 <= epoch < 200:
        lr = args.l_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    if epoch > 200:
        lr = args.l_rate * (1.0**(epoch//30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
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
    parser.add_argument('--datadir', nargs='?', type=str, default='/opt/data1/jingjinji/for_samples/samples_8classes_50/',
                        help='datadir')
    parser.add_argument('--suffix', nargs='?', type=str, default='.tif',
                        help='suffix')
    parser.add_argument('-n_bands', nargs='?', type=int, default=9,
                        help='number of bands')
    parser.add_argument('--img_size', nargs='?', type=int, default=512,
                        help='Size of the input image')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=300,
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=28,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-4,
                        help='Learning Rate')
    parser.add_argument('--model_path', nargs='?', type=str, default='./pfirst/',
                        help='Path where the trained models are saved.')
    parser.add_argument('--resume', nargs='?', type=str, default=None,#'./ndviwi/iter_best.pkl',
                        help='Path to previous saved model to restart from')
    parser.add_argument('--accurate_file', nargs='?', type=str, default='./pfirst/accurate_file.txt',
                        help='File to save the evaluation results')
    parser.add_argument('--debug_file', nargs='?', type=str, default='/tmp/debug')
    args = parser.parse_args()
    train(args)