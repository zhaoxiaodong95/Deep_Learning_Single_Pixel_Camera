import os
import pytorch_ssim

import sys
import json
import numpy as np
import torch
import time
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from utils import Logger
import torch.nn.functional as f
from data_cluster_loader import UCF101, UCF101_val

from skimage import measure
from conv_LSTM_fixed import ConvLSTM
# from model_Conv import SPC_LSTM
# from model import SPC_LSTM
from utils import AverageMeter
from pytorch_modelsize import SizeEstimator
import matplotlib.pyplot as plt
import torchvision.utils as v_image
import torchvision
from tensorboardX import SummaryWriter
from sklearn.preprocessing import normalize
from math import log10



import os
import torch
import time
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.optim import lr_scheduler
from utils import Logger
from data_cluster_loader_128 import UCF101, UCF101_val
from conv_LSTM_fixed_128 import ConvLSTM
from utils import AverageMeter
import torchvision.utils as v_image
from math import log10
from skimage import measure
import torch.nn.functional as f


## For GPU use
USE_CUDA = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Non-trainable Parameters
sample_size = 128
num_meas = 666      ### if we use more number of measurements, CPU memory can't handle it.(It's the key for memory usage)
hidden_size = 256
frame_size = [sample_size, sample_size]
sample_duration = 16                       ### sample duration should be around this value

learning_rate = 0.001
weight_decay = 1e-5
batch_size = 4
num_epochs = 100
lr_decay_freq = 30
lr_decay_factor = 0.5


n_threads = 8
checkpoint = 1                      ## after this epoch
attention_mechanism = "true"
conv_before_LSTM = True
resume_saved_model = False

## Path to the data, annotation, results(we don't need annotation)
resume_path = './results_raw/save_model_fixed_128.pth'
result_path = './results_raw'

hdf5_file  = './training_128.hdf5'
hdf5_file_validation  = './validation_128.hdf5'

validation_data = UCF101_val(hdf5_file_validation)
training_data = UCF101(hdf5_file)


train_loader = torch.utils.data.DataLoader(
    training_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=n_threads,
    pin_memory=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(
    validation_data,
    batch_size = batch_size,
    shuffle=False,
    num_workers=n_threads,
    pin_memory=True, drop_last=True)

## Loss Function
criterion = nn.MSELoss()
if not USE_CUDA:
    criterion = criterion.cuda()

if conv_before_LSTM:
    input_ch = 128*2
else:
    input_ch = 1


model= ConvLSTM(input_channels=input_ch, hidden_channels=[64], kernel_size=[9], frame_size=frame_size,
                n_filters=num_meas, hidden_size= hidden_size, device=device, step=sample_duration)
print(model)


## we need to put the parameters and buffers in the 1st GPU device
if USE_CUDA:
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0, 1])

arch = '{}-{}'.format(model, attention_mechanism)

optimizer = optim.Adam(model.parameters(),
                       lr=learning_rate,
                       weight_decay=weight_decay)

scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_freq, gamma=lr_decay_factor)

## For monitoring the training
train_epoch_logger = Logger(
    os.path.join(result_path, 'train_fixed_128.log'),
    ['epoch', 'loss'])

train_batch_logger = Logger(
    os.path.join(result_path, 'train_batch_fixed_128.log'),
    ['epoch', 'batch', 'loss', 'lr'])


val_logger = Logger(
    os.path.join(result_path, 'val.log'), ['epoch', 'loss', 'psnr'])

if resume_path and resume_saved_model:
    print('loading checkpoint {}'.format(resume_path))
    checkpoint = torch.load(resume_path)
    assert arch == checkpoint['arch']

    begin_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

phi_0 = torch.randn((num_meas, sample_size, sample_size))
phi_0 = phi_0[None, :, :, :]
phi_0 = phi_0.repeat((batch_size, 1, 1, 1)).cuda()

for epoch in range(num_epochs):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end_time = time.time()

    ## For Full Dataset
    for i, (inputs) in enumerate(train_loader):
        model.train()

        data_time.update(time.time() - end_time)
        inputs = inputs/255
        targets = inputs

        inputs = Variable(inputs)
        targets = Variable(targets)
        # GPU for targets and inputs
        if USE_CUDA:
            targets = targets.cuda(non_blocking=True)
            inputs = inputs.cuda(non_blocking=True)

        outputs, _ = model(inputs, phi_0)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        torch.autograd.set_detect_anomaly(True)

        ## Update the value of loss
        losses.update(loss.item(), inputs.size(0))

        ## Time for whole batch
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        train_batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'loss': losses.val,
            'lr': [param_g['lr'] for param_g in optimizer.param_groups]
        })

        train_epoch_logger.log({
            'epoch': epoch,
            'loss': losses.avg
        })

        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'lr {3}\t'.format(
                epoch,
                i + 1,
                len(train_loader),
                [param_g['lr'] for param_g in optimizer.param_groups],
                batch_time=batch_time,
                data_time=data_time,
                loss=losses))
    scheduler.step()
    ## If the training hits a new checkpoint
    if epoch % checkpoint == 0:
        save_file_path = os.path.join(result_path,
                                      'save_model_fixed_128.pth')
        states = {
            'epoch': epoch + 1,
            'arch': arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        torch.save(states, save_file_path)

        for j in range(batch_size):
            for k in range(1):
                v_image.save_image(inputs[j,k,:].data,'%s/x_bar_input_epoch_%04d_%04d.png' % ("./output_fixed_image_128", j, k))
                v_image.save_image(outputs[j,k,:].data,'%s/x_bar_epoch_%04d_%04d.png' % ("./output_fixed_image_128", j, k))

    ## Evaluate the model for validation
    model.eval()
    mse = nn.MSELoss(reduction='none')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    val_losses = AverageMeter()
    avg_psnr = AverageMeter()
    # ssim_value = 0
    end_time = time.time()

    # with torch.no_grad():
    for i, (inputs) in enumerate(val_loader):
        data_time.update(time.time() - end_time)

        inputs = inputs / 255
        targets = inputs

        if USE_CUDA:
            targets = targets.cuda()
            inputs = inputs.cuda()

        with torch.no_grad():
            inputs = Variable(inputs)
            targets = Variable(targets)

            outputs, _ = model(inputs, phi_0)

            mse_vals = mse(outputs, targets)
            mse_batch = torch.mean(mse_vals, dim=(1, 2, 3))
            val_loss = torch.mean(mse_vals)
            val_losses.update(val_loss.item(), inputs.size(0))

        ## Calculate the SSIM
        # ssim_value_inst = torch.zeros(batch_size)
        # for j in range(batch_size):
        #     ssim_value_inst[i] = measure.compare_ssim(outputs[j], targets[j])
        # ssim_value += torch.sum(ssim_value_inst) / batch_size


        ## Calculate the PSNR
        psnr = torch.mean(10 * torch.log10(1 / mse_batch))
        avg_psnr.update(psnr, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

    print('Validation Epoch: [{0}]\t'
          'Loss: {loss.avg:.4f}\t'
          'PSNR: {psnr.avg:.4f}\t'.format(
        epoch,
        loss=val_losses,
        psnr=avg_psnr))


    val_logger.log({'epoch': epoch, 'loss': losses.avg, 'psnr': avg_psnr.avg})



