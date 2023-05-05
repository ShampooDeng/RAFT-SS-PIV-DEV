import torch
from torch.utils.data import DataLoader
from tfrecord.torch.dataset import TFRecordDataset

from unliteflownet.model.models import estimate, device, Network
from unliteflownet.model.loss_functions import *
from unliteflownet.model.utils import realEPE

import numpy as np
from tqdm import tqdm
import cv2
import random
import math

import os
from subprocess import call
import argparse
# import time


# Data pre-processing
description = {"target":"byte",
               "label":"byte",
               "flow":"byte"}

TRAIN_QUEUE_SIZE = 19000
VAL_QUEUE_SIZE = 1000
to_fdi = True
cropOffset = 0
imgShape = [2,256,256]


def remap( img, x, y):
    x, y = np.float32(x), np.float32(y)
    out = cv2.remap(img, x, y, cv2.INTER_LINEAR)  # INTER_LANCZOS4 INTER_CUBIC INTER_LINEAR
    return out

def cdi2fdi( u_cdi, v_cdi, delta=1):
    assert u_cdi.shape == v_cdi.shape
    # shape[0]:height, shape[1]:width
    y, x = np.meshgrid(np.arange(u_cdi.shape[0]), np.arange(u_cdi.shape[1]),indexing='ij')
    u_c, v_c = u_cdi/delta, v_cdi/delta
    
    # init fdi field for later process
    u_f = remap(u_c, x+0.5*u_c, y+0.5*v_c)
    v_f = remap(v_c, x+0.5*u_c, y+0.5*v_c)
    
    # solve 'Vcdi[x + 1/2Vfdi(x)] = Vfdi(x)' via iteration
    for _ in range(5):
        u_f = remap(u_c, x+0.5*u_f, y+0.5*v_f)
        v_f = remap(v_c, x+0.5*u_f, y+0.5*v_f)
    u_f, v_f = delta*u_f, delta*v_f
    return u_f, v_f

def cropImg( img, offset=10):
    """
    img: CxHxW
    """
    _, height, width = img.shape
    return img[:, offset:height-offset, offset:width-offset]

def reshapeImage(img,imgShape,ifFlow=False):
    img = img.reshape(imgShape[0],imgShape[1],imgShape[2])
    
    if ifFlow and to_fdi:
        ucdi = img[0]
        vcdi = img[1]
        img = cdi2fdi(ucdi,vcdi)
        img = np.stack(img,axis=0)

    if cropOffset != 0:
        img = cropImg(img)
    return img
    
def decodeData(features):
    """
    https://stackoverflow.com/questions/40589499/what-do-the-signs-in-numpy-dtype-mean
    https://numpy.org/doc/stable/reference/generated/numpy.dtype.byteorder.html
    """
    features['target'] = reshapeImage(features['target'].view('<f4'),imgShape)
    features['flow'] = reshapeImage(features['flow'].view('<f4'),imgShape)
    features['label'] = features['label'].view('<f4')
    
    return features

###############################################################
def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True  # uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms
    torch.backends.cudnn.enabled = True

    return True

def train(args):
    set_seed(0)
    model_save_dir = args.save_path +args.name
    model_save_path = model_save_dir +'/' +'ckpt.tar'
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
        
    lowest_val_epe = np.inf
    
    # Prepare dataloader
    tfrecord2idx_script = "tfrecord2idx"
    if not os.path.isfile(args.train_tfrecord_idx):
        call([tfrecord2idx_script, args.train_tfrecord, args.train_tfrecord_idx])
    if not os.path.isfile(args.val_tfrecord_idx):
        call([tfrecord2idx_script, args.val_tfrecord, args.val_tfrecord_idx])
        
    trainDataset = TFRecordDataset(args.train_tfrecord,
                                   args.train_tfrecord_idx,
                                   description,
                                   transform=decodeData,
                                   shuffle_queue_size=TRAIN_QUEUE_SIZE)
    train_pii = DataLoader(trainDataset,batch_size=args.batch_size)
    valDataset = TFRecordDataset(args.val_tfrecord,
                                 args.val_tfrecord_idx,
                                 description,
                                 transform=decodeData,
                                 shuffle_queue_size=VAL_QUEUE_SIZE)
    val_pii = DataLoader(valDataset,batch_size=args.batch_size)
    
    # Prepare model
    model = Network().to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay,
                                 eps=args.eps,
                                 amsgrad=args.amsgrad)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=.2,
                                                           patience=5,
                                                           threshold=.001,
                                                           min_lr=8e-7)
    criterion_train = multiscaleUnsupervisorError
    criterion_val = realEPE
    
    if args.recover:
        ckpt = torch.load(model_save_path)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Recovered from epoch {ckpt['epoch']+1}")
    
    # train & validation
    for epoch in range(args.epochs):
        last_training_loss = .0
        last_validation_loss = .0
        train_loss, flow2_EPE =.0, .0
        validation_EPE = .0
        total_training_samples = 0
        total_validation_samples = 0

        # train
        model.train()
        train_loader_len = int(math.ceil(TRAIN_QUEUE_SIZE / args.batch_size))
        train_pbar = tqdm(enumerate(train_pii), total=train_loader_len,
                          desc='Epoch: [' + str(epoch + 1) + '/' + str(args.epochs) + '] Training',
                          postfix='loss: ' + str(last_training_loss), position=0, leave=False)
        
        for i, sample_batched in train_pbar:
            local_dict = sample_batched
            images = local_dict['target'].type(torch.FloatTensor).to(device) / 255
            flows = local_dict['flow'].type(torch.FloatTensor).to(device)
            image1 = images[:,0,...].unsqueeze(dim=1) # image1 to BxCxWxH
            image2 = images[:,1,...].unsqueeze(dim=1) # image2 to BxCxWxH
            assert len(image1.shape) == 4, 'wrong shape for image1, image2'
            
            optimizer.zero_grad()
            output_forward = estimate(image1,image2,model,train=True)
            output_backward = estimate(image2,image1,model,train=True)
            loss = criterion_train(output_forward, output_backward, image1, image2)
            real_timeEPE = realEPE(output_forward[-1], flows).item()
            flow2_EPE += real_timeEPE*image1.shape[0]

            loss.backward()
            train_loss += loss.item() * image1.shape[0]
            optimizer.step()
            
            total_training_samples += image1.shape[0]
            epoch_training_loss = train_loss / total_training_samples
            epoch_training_epe = flow2_EPE / total_training_samples
            
            train_pbar.set_postfix_str(
                'loss: ' + "{:10.6f}".format(epoch_training_loss) + \
                ' epe: ' + "{:10.6f}".format(epoch_training_epe))
            
        # validation
        model.eval()
        val_loader_len = int(math.ceil(VAL_QUEUE_SIZE / args.batch_size))
        val_pbar = tqdm(enumerate(val_pii), total=val_loader_len,
                    desc='Epoch: [' + str(epoch+1) + '/' + str(args.epochs) + '] Validation',
                    postfix='loss: ' + str(last_validation_loss), position=1, leave=False)

        with torch.no_grad():
            for i, sample_batched in val_pbar:
                local_dict = sample_batched
                images = local_dict['target'].type(torch.FloatTensor).to(device) / 255
                flows = local_dict['flow'].type(torch.FloatTensor).to(device)
                image1 = images[:,0,...].unsqueeze(dim=1)
                image2 = images[:,1,...].unsqueeze(dim=1)
                assert len(image1.shape) == 4, 'wrong shape for image1, image2'

                output_forward = estimate(image1, image2, model, train=True)
                loss = criterion_val(output_forward[-1],flows)
                validation_EPE += loss.item()*image1.shape[0]
                
                total_validation_samples += image1.shape[0]
                epoch_validation_EPE = validation_EPE / total_validation_samples
                val_pbar.set_postfix_str(
                    'epe: ' + "{:10.6f}".format(epoch_validation_EPE)
                )
        scheduler.step(epoch_validation_EPE)
        
        # Save model
        if epoch_validation_EPE < lowest_val_epe:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                model_save_path
            )
            lowest_val_epe = epoch_validation_EPE
            print(f"ckpt updated with val epe:{round(lowest_val_epe,5)}")

        # push train metrics to console
        print(
            "Epoch: ", epoch+1, ", Avg. Train EPE Loss: %1.3f" % epoch_training_epe,
            " Avg. Validation EPE Loss: %1.3f" % epoch_validation_EPE,
            # "Time used this epoch (seconds): %1.3f" % (end_time - start_time),
            # "Time remain(hrs) %1.3f" % (total_time / (epoch + 1) *
            #                             (n_epochs - epoch) / 3600))
        )

        # add writer to trian.log
        file = open(args.log_path, 'a')
        file.writelines('Epoch: '+ str(epoch + 1)+ ' ')
        file.writelines('training loss: '+ str(round(epoch_training_loss,5))+ ' ')
        # file.writelines('validation loss: '+ str(round(loss_metric[1].cpu().item(),5))+ ' ')
        file.writelines('train epe: '+ str(round(epoch_training_epe,5))+ ' ')
        file.writelines('val epe: '+ str(round(epoch_validation_EPE,5))+ ' ')
        file.writelines('lr: '+ str(optimizer.__getattribute__('param_groups')[0]['lr'])+ '\n')
        file.close()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO: add more argument
    parser.add_argument('--name',type=str,
                        default='UnLiteFlowNet-PIV_Problem2')
    parser.add_argument('--train_tfrecord', type=str,
                        default='./data/Training_Dataset_ProblemClass2_RAFT256-PIV.tfrecord-00000-of-00001')
    parser.add_argument('--train_tfrecord_idx', type=str,
                        default='./data/idx_files/training_dataset_ProbClass2_256px.idx')
    parser.add_argument('--val_tfrecord', type=str,
                        default='./data/Validation_Dataset_ProblemClass2_RAFT256-PIV.tfrecord-00000-of-00001')
    parser.add_argument('--val_tfrecord_idx', type=str,
                        default='./data/idx_files/validation_dataset_ProbClass2_256px.idx')
    parser.add_argument('--batch_size', type=int,
                        default=10)
    parser.add_argument('--lr', type=float,
                        default=0.0001)
    parser.add_argument('--weight_decay', type=float,
                        default=0.00005)
    parser.add_argument('--eps', type=float,
                        default=0.001)
    parser.add_argument('--amsgrad', type=bool,
                        default=True)
    parser.add_argument('--epochs', type=int,
                        default=25)
    parser.add_argument('--save_path', type=str,
                        default='./results/')
    parser.add_argument('--log_path', type=str,
                        default='./train.log')
    parser.add_argument('--recover', type=bool,
                        default=False)
    args = parser.parse_args()
    
    train(args)
