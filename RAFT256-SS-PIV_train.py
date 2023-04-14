'''
Copyright (c) 2020-2021, Christian Lagemann
'''

import os
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from tqdm import tqdm

import math
from RAFT.flowNetsRAFT256_SS import RAFT256_SS
# import socket
import cv2
import numpy as np
from torch.utils.data import DataLoader
from tfrecord.torch.dataset import TFRecordDataset

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch import DALIClassificationIterator as PyTorchIterator
import nvidia.dali.tfrecord as tfrec

from subprocess import call
import os.path

###############################################################################
# class TFRecordPipeline(Pipeline):
#     def __init__(self, batch_size, num_threads, device_id, num_gpus, shard_id, num_shards, tfrecord, tfrecord_idx, exec_pipelined=False, exec_async=False, is_shuffle=False, image_shape=[2,256,256], label_shape=[2,12]):
#         super(TFRecordPipeline, self).__init__(batch_size, num_threads, device_id, exec_pipelined=False, exec_async=False)
#         self.input = ops.TFRecordReader(path = tfrecord, 
#                                         index_path = tfrecord_idx,
#                                         random_shuffle=is_shuffle,
#                                         pad_last_batch = True,
#                                         shard_id=shard_id,
#                                         num_shards=num_shards,
#                                         features = {"target" : tfrec.FixedLenFeature([], tfrec.string, ""),
#                                                     "label": tfrec.FixedLenFeature([], tfrec.string,  ""),
#                                                     "flow" : tfrec.FixedLenFeature([], tfrec.string, ""),
#                                                    })
    
#         self.decode = ops.PythonFunction(function=self.extract_view, num_outputs=1)
#         self.reshape_image = ops.Reshape(shape=image_shape)
#         self.reshape_label = ops.Reshape(shape=label_shape)

#     def extract_view(self, data):
#         ext_data = data.view('<f4')
#         return ext_data

#     def define_graph(self):
#         inputs = self.input(name="Reader")
#         images = self.reshape_image(self.decode(inputs['target']))
#         labels = self.reshape_label(self.decode(inputs['label']))
#         flows = self.reshape_image(self.decode(inputs['flow']))

#         return images, labels, flows


# pathToTfrecord = './data/Test_Dataset_10Imgs_cylinder.tfrecord-00000-of-00001'
# indexPath = './data/idx_files/Test_Dataset_10Imgs_cylinder.idx'
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
    features['flow'] = reshapeImage(features['flow'].view('<f4'),imgShape,ifFlow=True)
    features['label'] = features['label'].view('<f4')
    
    return features
               
# testDataset = TFRecordDataset(pathToTfrecord, indexPath, description,
#                               transform=decodeData)
# testLoader = DataLoader(testDataset, batch_size=1,)

###############################################################################

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


### main method


def main():

    
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int,
                        help='number of compute nodes')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('--name', type=str, default='RAFT-PIV256_training',
                        help='name of experiment')
    parser.add_argument('--input_path_ckpt', type=str,
                        default='./precomputed_ckpts/RAFT256-PIV_ProbClass2/ckpt.tar',
                        help='path of already trained checkpoint')
    parser.add_argument('--recover', type=eval, default=False,
                        help='Wether to load an existing checkpoint')
    parser.add_argument('--output_dir_ckpt', type=str, default='./checkpoints/',
                        help='output directory of checkpoint')

    parser.add_argument('--train_tfrecord', type=str,
                        default='./data/minimal_training_dataset_ProbClass1_256px.tfrecord-00000-of-00001',
                        help='TFRECORD file for training')
    parser.add_argument('--train_tfrecord_idx', type=str,
                        default='./data/idx_files/minimal_training_dataset_ProbClass1_256px.idx',
                        help='corresponding idx file for train_tfrecord - if not existing, it is generated later')
    parser.add_argument('--val_tfrecord', type=str,
                        default='./data/minimal_validation_dataset_ProbClass1_256px.tfrecord-00000-of-00001',
                        help='TFRECORD file for validation')
    parser.add_argument('--val_tfrecord_idx', type=str,
                        default='./data/idx_files/minimal_validation_dataset_ProbClass1_256px.idx',
                        help='corresponding idx file for val_tfrecord - if not existing, it is generated later')

    parser.add_argument('--amp', type=eval, default=False, help='Wether to use auto mixed precision')
    parser.add_argument('-a', '--arch', type=str, default='RAFT256', choices=['RAFT256'],
                        help='Type of architecture to use')
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--offset', default=256, type=int,
                        help='interrogation window size')
    parser.add_argument('--shift', default=64, type=int,
                        help='shift of interrogation windon in px')

    parser.add_argument('--init_lr', default=0.0001, type=float,
                        help='initial learning rate')
    parser.add_argument('--reduce_factor', default=0.2, type=float,
                        help='reduce factor of ReduceLROnPlateau scheme')
    parser.add_argument('--patience_level', default=10, type=int,
                        help='patience level of ReduceLROnPlateau scheme')
    parser.add_argument('--min_lr', default=1e-8, type=float,
                        help='minimum learning rate')

    parser.add_argument('--iters', default=16, type=int,
                        help='number of update steps in ConvGRU')
    parser.add_argument('--upsample', type=str, default='convex',
                        choices=['convex', 'bicubic', 'bicubic8', 'lanczos4', 'lanczos4_8'],
                        help="""Type of upsampling method""")

    parser.add_argument('--log_path', type=str, default='./train.log')
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau')
    args = parser.parse_args()
    print('args parsed')
    
    print(args.init_lr, args.patience_level)

    # mp.spawn(train, nprocs=args.gpus, args=(args,))
    train(0,args)

def train(GPU,args):
    #init procedure
    torch.manual_seed(0)
    # print('getting master_addr', flush=True)
    # if "SLURM_JOBID" in os.environ:
    #     masterIP, _, _, _, _, _, _ = resolve_master_node(platform.node(), 8888)
    #     os.environ['MASTER_ADDR'] = masterIP
    # else:
    #     os.environ['MASTER_ADDR'] = '127.0.0.1'

    print('start GPU:' + str(GPU))
    
    # MASTER_PORT = int(os.environ.get("MASTER_PORT", 8738))
    # MASTER_ADDR = os.environ.get("MASTER_ADDR")
    N_NODES = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NNODES", 1)))
    # NODE_RANK = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    WORLD_SIZE = args.gpus * N_NODES
    # backend = 'nccl'
    # NODE_NAME = socket.gethostname()
    # NODE_IP = socket.gethostbyname(NODE_NAME)

    # print('node_name', NODE_NAME, 'node_ip', NODE_IP, 'rank', rank, 'node_rank', NODE_RANK, 'GPU', GPU,
    #       'master_IP', MASTER_ADDR, 'master_port', MASTER_PORT, 'world_size    ', WORLD_SIZE, flush=True)

    # tcp_store = dist.TCPStore(MASTER_ADDR, MASTER_PORT, WORLD_SIZE, rank == 0)
    # dist.init_process_group(backend,
    #                         store=tcp_store,
    #                         rank=rank,
    #                         world_size=WORLD_SIZE
    #                         )
    rank = 0

    int_GPU = GPU
    GPU = torch.device("cuda", GPU)
    torch.cuda.set_device(GPU)

    # print('synchronizing all processes', flush=True)
    # dist.barrier()
    # print('processes synchronized', flush=True)

    # create a second gloo group
    # print('creating second process group', flush=True)
    # list_ranks = [int(_) for _ in range(dist.get_world_size())]
    # gather_group = dist.new_group(ranks=list_ranks, backend='gloo')

    if args.arch == 'RAFT256':
        model = RAFT256_SS(args)
        print('Selected model: RAFT-SS - -', args.arch)
    else:
        raise ValueError('Selected model not supported: ', args.arch)

    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of trainable parameters: ', pytorch_trainable_params)

    checkpoint_dir = args.output_dir_ckpt + args.name
    checkpoint_path = checkpoint_dir + '/ckpt.tar'
    if rank == 0:
        if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
    
    if args.recover:
        print('recovering: ', args.input_path_ckpt)
        checkpoint = torch.load(args.input_path_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('model recovered')
    
    model.cuda(GPU)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr)
    if args.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.reduce_factor, patience=args.patience_level, min_lr=args.min_lr)
    elif args.scheduler == 'OneCycleLR':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,args.lr,epochs=args.epochs,steps_per_epoch=1900,pct_start=0.05,cycle_momentum=False,anneal_strategy='linear')
    scaler = GradScaler()
        
    if args.recover:
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']+1
        print("recovering at epoch: ", start_epoch)

    # model = DDP(model, device_ids=[GPU],find_unused_parameters=True)

    # DALI data loading
    tfrecord2idx_script = "tfrecord2idx"
    if not os.path.isfile(args.train_tfrecord_idx):
        call([tfrecord2idx_script, args.train_tfrecord, args.train_tfrecord_idx])
    if not os.path.isfile(args.val_tfrecord_idx):
        call([tfrecord2idx_script, args.val_tfrecord, args.val_tfrecord_idx])

    # train_pipe = TFRecordPipeline(batch_size=args.batch_size, num_threads=8, device_id=int_GPU, num_gpus=1,
    #                               tfrecord=args.train_tfrecord, tfrecord_idx=args.train_tfrecord_idx,
    #                               num_shards=WORLD_SIZE, shard_id=rank,
    #                               is_shuffle=True, image_shape=[2, args.offset, args.offset], label_shape=[12, ])
    # train_pipe.build()
    # train_pii = DALIGenericIterator(train_pipe, ['target', 'label', 'flow'],
    #                                 size=int(train_pipe.epoch_size("Reader") / WORLD_SIZE),
    #                                 last_batch_padded=True, fill_last_batch=False, auto_reset=True)

    # val_pipe = TFRecordPipeline(batch_size=args.batch_size, num_threads=8, device_id=int_GPU, num_gpus=1,
    #                             tfrecord=args.val_tfrecord, tfrecord_idx=args.val_tfrecord_idx,
    #                             num_shards=WORLD_SIZE, shard_id=rank,
    #                             is_shuffle=True, image_shape=[2, args.offset, args.offset], label_shape=[12, ])
    # val_pipe.build()
    # val_pii = DALIGenericIterator(val_pipe, ['target', 'label', 'flow'],
    #                               size=int(val_pipe.epoch_size("Reader") / WORLD_SIZE),
    #                               last_batch_padded=True, fill_last_batch=False, auto_reset=True)

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

    lowest_val_epe = 100000.0
    for epoch in range(args.epochs):
        model.train()
        last_training_loss, last_validation_loss = 0.0, 0.0
        sum_training_loss, sum_validation_loss = 0.0, 0.0
        total_training_samples, total_validation_samples = 0, 0
        sum_training_epe, sum_validation_epe = 0.0, 0.0

        train_loader_len = int(math.ceil(TRAIN_QUEUE_SIZE / args.batch_size))
        train_pbar = tqdm(enumerate(train_pii), total=train_loader_len,
                          desc='Epoch: [' + str(epoch + 1) + '/' + str(args.epochs) + '] Training',
                          postfix='loss: ' + str(last_training_loss), position=0, leave=False)

        #Training
        for i, sample_batched in train_pbar:
            # debug
            # print(type(sample_batched))
            # print(sample_batched.keys())
            # assert False

            # local_dict = sample_batched[0]
            local_dict = sample_batched
            images = local_dict['target'].type(torch.FloatTensor).cuda(GPU) / 255
            flows = local_dict['flow'].type(torch.FloatTensor).cuda(GPU)

            with autocast(enabled=args.amp):
                pred_flows = model(images, flows, args=args)

                training_loss, metrics = pred_flows[1]
                train_epe_loss = metrics['epe']

                sum_training_loss += training_loss.item() * images.shape[0]
                total_training_samples += images.shape[0]
                epoch_training_loss = sum_training_loss / total_training_samples
                sum_training_epe += train_epe_loss * images.shape[0]
                epoch_train_epe_loss = sum_training_epe / total_training_samples

                scaler.scale(training_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()

                train_pbar.set_postfix_str(
                    'loss: ' + "{:10.6f}".format(epoch_training_loss) + \
                    ' epe: ' + "{:10.6f}".format(epoch_train_epe_loss))

        # Validation
        with torch.set_grad_enabled(False):
            #set evaluation mode
            model.eval()
            is_training=False

            val_loader_len = int(math.ceil(VAL_QUEUE_SIZE / args.batch_size))
            val_pbar = tqdm(enumerate(val_pii), total=val_loader_len,
                        desc='Epoch: [' + str(epoch+1) + '/' + str(args.epochs) + '] Validation',
                        postfix='loss: ' + str(last_validation_loss), position=1, leave=False)

            for i, sample_batched in val_pbar:
                # local_dict = sample_batched[0]
                local_dict = sample_batched
                images = local_dict['target'].type(torch.FloatTensor).cuda(GPU) / 255
                flows = local_dict['flow'].type(torch.FloatTensor).cuda(GPU)

                with autocast(enabled=args.amp):
                    pred_flows = model(images, flows, args=args)

                    validation_loss, metrics = pred_flows[1]
                    val_epe_loss = metrics['epe']

                    sum_validation_loss += validation_loss.item() * images.shape[0]
                    total_validation_samples += images.shape[0]
                    epoch_validation_loss = sum_validation_loss / total_validation_samples

                    sum_validation_epe += val_epe_loss * images.shape[0]
                    epoch_val_epe_loss = sum_validation_epe / total_validation_samples

                    val_pbar.set_postfix_str(
                        'loss: ' + "{:10.6f}".format(epoch_validation_loss) + \
                        ' epe: ' + "{:10.6f}".format(epoch_val_epe_loss))

        scheduler.step(epoch_validation_loss)
        metrics_tensor = torch.tensor(
            [epoch_training_loss, epoch_validation_loss, epoch_train_epe_loss, epoch_val_epe_loss])
        # losses = [torch.ones_like(metrics_tensor) for _ in range(dist.get_world_size())]
        # if rank == 0:
        #     torch.distributed.gather(tensor=metrics_tensor, gather_list=losses, dst=0, group=gather_group)
        # else:
        #     torch.distributed.gather(tensor=metrics_tensor, dst=0, group=gather_group)

        loss_metric = metrics_tensor
        if rank == 0:
            # loss_metric = torch.mean(torch.stack(losses), dim=0)
            print('Epoch: ', epoch + 1, ' training loss: ', loss_metric[0], ' validation loss: ', \
                  loss_metric[1], ' train epe: ', loss_metric[2], ' val epe: ', loss_metric[3], \
                  ' lr: ', str(optimizer.__getattribute__('param_groups')[0]['lr']), flush=True)
            
            # add writer to trian.log
            if not os.path.isfile(args.log_path):
                pass
            file = open(args.log_path, 'a')
            file.writelines('Epoch: '+ str(epoch + 1)+ ' ')
            file.writelines('training loss: '+ str(round(loss_metric[0].cpu().item(),5))+ ' ')
            file.writelines('validation loss: '+ str(round(loss_metric[1].cpu().item(),5))+ ' ')
            file.writelines('train epe: '+ str(round(loss_metric[2].cpu().item(),5))+ ' ')
            file.writelines('val epe: '+ str(round(loss_metric[3].cpu().item(),5))+ ' ')
            file.writelines('lr: '+ str(optimizer.__getattribute__('param_groups')[0]['lr'])+ '\n')
            file.close()

            #save model
            if (loss_metric[3] < lowest_val_epe):
                lowest_val_epe = loss_metric[3]
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, checkpoint_path)
                print('model saved and lowest epe overwritten:', lowest_val_epe, flush=True)

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
