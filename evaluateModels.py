import torch

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import glob
import time
import os
import argparse

from RAFT.flowNetsRAFT256_SS import RAFT256_SS
from RAFT.flowNetsRAFT256 import RAFT256
import unliteflownet.model.models as UnLiteFlowNet
from deformpiv.deformpiv import DeformPIV

assert torch.cuda.is_available(), "cuda not avaiable!"
assert torch.cuda.device_count() == 1, "multi-gpu is not currently supported!"


PREFIX = './PIG/'
GPU_ID = 0
CASE_NAMES = ['lamb-oseen', 'sin']
MODEL_NAMES = ['RAFT-PIV','RAFT-SS-PIV','RAFT-SS-PIV-DEF','UnLiteFlowNet-PIV']


def getSamples(prefix:str, caseName:str):
    pathList = []
    sortKeyClipHead = 0
    sortKeyClipTail = 0

    if caseName == 'lamb-oseen':
        matchString = prefix + 'oseen*.npz'
        sortKeyClipHead, sortKeyClipTail = 12, 17
    elif caseName == 'sin':
        matchString = prefix + 'sin*.npz'
        sortKeyClipHead, sortKeyClipTail = 10, 12

    pathList = glob.glob(matchString)
    pathList.sort(
        key = lambda item: float(item[sortKeyClipHead:sortKeyClipTail])
        )
    return pathList


def computeEPE(gt:torch.Tensor, pred:torch.Tensor):
    """ gt: NxCxWxH or CxWxH
        pred: NxCxWxH or CxWxH
        return: float64 """
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
    if len(gt.shape) == 3:
        gt = gt.unsqueeze(0)

    epe = torch.sum(torch.square(pred-gt),dim=1).sqrt()
    return epe.mean().item()


def readCase(path):
    data = np.load(path)
    element = ['img1', 'img2', 'u', 'v']
    img1, img2, u, v = [data[x] for x in element]

    flowGt = np.stack((u,v),axis=1)
    flowGt = torch.from_numpy(flowGt).float()
    # flowGt = flowGt.unsqueeze(dim=0)

    img1 = torch.from_numpy(img1).float()
    img2 = torch.from_numpy(img2).float()
    # img1 = img1.repeat(1,1,1,1)
    # img2 = img2.repeat(1,1,1,1)

    # pass to gpu
    img1, img2, flowGt = [x.cuda(GPU_ID) for x in [img1,img2,flowGt]]
    
    # concate imgas & normalize image
    imgs = torch.stack((img1,img2),dim=1)
    # imgs = torch.squeeze(imgs, dim=2)
    imgs = imgs/255
    
    return imgs, flowGt


def predictFlowRAFTSS(args, model, imgs, gt):
    flow_iteration, eval_metrics = model(imgs, gt, args, evaluate=True)
    pred = flow_iteration[-1]
    return pred, eval_metrics


def predictFlowRAFTSSDeform(args, model, imgs, gt):
    flow_iteration, eval_metrics = model(imgs, gt, args, evaluate=False)
    pred = flow_iteration[-1]
    return pred, eval_metrics


def predictFlowRAFT(args, model, imgs, gt):
    flow_iteration, eval_metrics = model(imgs, gt, args)
    pred = flow_iteration[-1]
    return pred, eval_metrics


def predictFlowUnLiteFlowNet(_, model, imgs, gt):
    img1 = imgs[:,0,...].unsqueeze(dim=1)
    img2 = imgs[:,1,...].unsqueeze(dim=1)
    assert len(img1.shape) == 4, "wrong shape for img1"
    output = UnLiteFlowNet.estimate(img1,img2,model,train=True)
    nan = torch.as_tensor(torch.nan)
    return output[-1], (nan,nan)


def evaluateModel(args, modelName:str, path2Cases:list, storeDir:str,debug=False):
    case = []
    aveEPE = []
    preds = []
    gts = []
    # load model
    if modelName == 'RAFT-SS-PIV':
        model = RAFT256_SS(args).cuda(GPU_ID)
        path2Ckpt = './results/RAFT256-PIV_SS_ProbClass2/ckpt.tar'
        predictFlow = predictFlowRAFTSS
    elif modelName == 'RAFT-SS-PIV-DEF':
        model = RAFT256_SS(args).cuda(GPU_ID)
        path2Ckpt = './results/RAFT256-PIV_SS_ProbClass2/ckpt.tar'
        predictFlow = predictFlowRAFTSSDeform
    elif modelName == 'RAFT-PIV':
        model = RAFT256(args).cuda(GPU_ID)
        path2Ckpt = './results/ckpt.tar'
        predictFlow = predictFlowRAFT
    elif modelName == 'UnLiteFlowNet-PIV':
        model = UnLiteFlowNet.Network().cuda(GPU_ID)
        path2Ckpt = './results/UnLiteFlowNet-PIV_Problem2/ckpt.tar'
        predictFlow = predictFlowUnLiteFlowNet
        
    checkpoint = torch.load(path2Ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with torch.set_grad_enabled(False):
        # evaluate model on test cases
        evalProcessBar = tqdm(path2Cases)
        for path in evalProcessBar:
            evalProcessBar.set_description('')
            imgs, gt = readCase(path)
            # flow_iteration, eval_metrics = model(imgs, gt, args, evaluate=True)
            # flow_iteration, eval_metrics = model(imgs, gt, args)
            # pred = flow_iteration[-1]
            # loss, epe_dict = eval_metrics
            # epe = epe_dict['epe']

            pred, eval_metrics = predictFlow(args,model,imgs,gt)
            epe = computeEPE(gt,pred)
            loss, _ = eval_metrics

            evalProcessBar.set_postfix_str('loss:'+'{:10.6f}'.format(loss.item()) + ' ' + \
                                       'epe:'+'{:10.6f}'.format(epe))
            aveEPE.append(epe)
            preds.append(pred.cpu().numpy())
            gts.append(gt.cpu().numpy())
            
            if debug:
                print('loss:'+'{:10.6f}'.format(loss.item()) + ' ' + \
                                           'epe:'+'{:10.6f}'.format(epe))
                print(path)
        
        # save results
        if not os.path.exists(storeDir):
            os.makedirs(storeDir)
        np.savez(storeDir + modelName, data=np.asarray(aveEPE), preds=preds, gts=gts)
        

def evalWithDeformPIV(config, paht2Cases, storeDir, modelName='DiffeomorphicPIV', debug=False):
    # initialize DeformPIV
    dpiv = DeformPIV(config)
    
    # compute img pairs
    epeList = []
    preds = []
    gts = []
    evalProcessBar = tqdm(paht2Cases)
    for path in evalProcessBar:
        # data preprocessing
        data = np.load(path)
        element = ['img1', 'img2', 'u', 'v']
        img1, img2, u, v = [data[x] for x in element]
        gt = np.stack((u,v),axis=1)
        
        # compute flow
        predOfOneSampleSet = []
        for i in range(img1.shape[0]):
            _,_,predu,predv = dpiv.compute(img1[i], img2[i], u[i], v[i])
            pred = np.stack((predu, predv),axis=0)
            predOfOneSampleSet.append(pred)
            # # debug
            # print(pred.shape)
            # break
        predOfOneSampleSet = np.stack(predOfOneSampleSet,axis=0)
        epe = np.sqrt(np.sum(np.square(predOfOneSampleSet - gt), axis=1)).mean()

        epeList.append(epe)
        preds.append(predOfOneSampleSet.astype(np.float32))
        gts.append(gt.astype(np.float32))

    # debug
    if debug:
        print(epeList.__len__(), preds.__len__(), gts.__len__())
        print(preds[0].shape, gts[0].shape)

    # save model predicted epe resutls in the shape of NxBx2xWxH
    if not os.path.exists(storeDir):
        os.makedirs(storeDir)
    np.savez(storeDir + modelName, data=np.asarray(epeList), preds=preds, gts=gts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ------------------------- config for deeppiv methods -------------------------
    parser.add_argument('-n', '--nodes', default=1, type=int,
                        help='number of compute nodes')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('--name', type=str, default='RAFT-PIV256_test_test1',
                        help='name of experiment')
    parser.add_argument('--input_path_ckpt', type=str,
                        default='./precomputed_ckpts/RAFT256-PIV_ProbClass2/ckpt.tar',
                        help='path of already trained checkpoint')
    parser.add_argument('--recover', type=eval, default=True,
                        help='Wether to load an existing checkpoint')
    parser.add_argument('--output_dir_results', type=str, default='./test_results/',
                        help='output directory of test results')

    parser.add_argument('--test_dataset', type=str, default='cylinder',
                        choices=['backstep', 'cylinder', 'jhtdb', 'dns_turb', 'sqg', 'tbl', 'twcf'],
                        help='test dataset to evaluate')
    parser.add_argument('--plot_results', type=eval, default=True,
                        help="""Whether or not to plot predicted results.""")

    parser.add_argument('--amp', type=eval, default=False, help='Wether to use auto mixed precision')
    parser.add_argument('-a', '--arch', type=str, default='RAFT256', choices=['RAFT256'],
                        help='Type of architecture to use')
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--batch_size_test', default=1, type=int)
    parser.add_argument('--split_size', default=1, type=int)
    parser.add_argument('--offset', default=256, type=int,
                        help='interrogation window size')
    parser.add_argument('--shift', default=64, type=int,
                        help='shift of interrogation window in px')

    parser.add_argument('--iters', default=16, type=int,
                        help='number of update steps in ConvGRU')
    parser.add_argument('--upsample', type=str, default='convex',
                        choices=['convex', 'bicubic', 'bicubic8', 'lanczos4', 'lanczos4_8'],
                        help="""Type of upsampling method""")
    # ------------------------- config for deformpiv -------------------------
    parser.add_argument('--pivmethod', type=str,
                        default='deeppiv1')
    parser.add_argument('--deform', type=str,
                        default='FDDI')
    parser.add_argument('--runs', type=int,
                        default=15)
    args = parser.parse_args()
    
    # paths = getSamples(PREFIX,caseName='lamb-oseen')
    # paths = getSamples(PREFIX,caseName='sin')
    ## debug
    # print(paths, sep='\n')
    # imgs,gt = readCase(paths[0])
    # print(imgs.shape, gt.shape, sep='\n')
    
    ## evaluate later appended model or methods
    paths = getSamples(PREFIX, caseName='lamb-oseen')
    evalWithDeformPIV(args, paths, storeDir='./results/lamb-oseen/', debug=True)
    # paths = getSamples(PREFIX, caseName='sin')
    # evalWithDeformPIV(args, paths, storeDir='./results/sin/', debug=True)
    assert False
    
    for model in MODEL_NAMES:
        for case in CASE_NAMES:
            paths = getSamples(PREFIX, caseName=case)
            evaluateModel(args,modelName=model,path2Cases=paths,
                          storeDir='./results/'+case+'/',
                          debug=False)
            print(model, case, sep=' ')
