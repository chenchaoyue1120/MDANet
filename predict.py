import numpy as np
import torch
import cv2
from tqdm import tqdm
import torch.nn as nn
from model.mda_model import MDANet
from utils.dataset import FundusSeg_Loader
import copy
from sklearn.metrics import roc_auc_score
from eval_metrics import perform_metrics
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

dataset_name="drive"
model_path='./drive.pth'

if dataset_name == "drive":
    test_data_path = "./DRIVE/drive_test/"
    raw_height = 584
    raw_width = 565

save_path='./results/'

if __name__ == "__main__":
    with torch.no_grad():
        test_dataset = FundusSeg_Loader(test_data_path,0, dataset_name)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
        print('Testing images: %s' %len(test_loader))
        # 选择设备CUDA
        device = torch.device('cuda')
        # 加载网络，图片单通道，分类为1。
        net = MDANet(n_channels=3, n_classes=1)
        # 将网络拷贝到deivce中
        net.to(device=device)
        # 加载模型参数
        print(f'Loading model {model_path}')
        net.load_state_dict(torch.load(model_path, map_location=device))

        # 测试模式
        net.eval()
        pre_stack = []
        label_stack = []
        mask_stack = []

        for image, label, mask, filename in test_loader:
            image = image.cuda().float()
            label = label.cuda().float()
            mask = mask.cuda().float()

            image = image.to(device=device, dtype=torch.float32)
            s1, s2, s3, s4, pred = net(image)
            # Normalize to [0, 1]
            pred = torch.sigmoid(pred)
            pred = pred.cpu().numpy().astype(np.double)[0][0]  # CHASE: 1024*1024, DRIVE: 584*584
            label = label.cpu().numpy().astype(np.double)[0][0]# CHASE: 1024*1024, DRIVE: 584*584
            mask = mask.cpu().numpy().astype(np.double)[0]     # CHASE: 1024*1024, DRIVE: 584*584

            pred  = pred[:raw_height,:raw_width]  # CROP: height 960 , width 999
            label = label[:raw_height,:raw_width]
            mask  = mask[:raw_height,:raw_width]

            pre_stack.append(pred)
            label_stack.append(label)
            mask_stack.append(mask)
            # 保存图片
            pred = pred * 255
            save_filename = save_path + filename[0] + '_mda.png'
            cv2.imwrite(save_filename, pred)
            #print(f'{save_filename} done!')

        print('Evaluating...')
        label_stack = np.stack(label_stack, axis=0)
        pre_stack = np.stack(pre_stack, axis=0)
        mask_stack = np.stack(mask_stack, axis=0)
        label_stack = label_stack.reshape(-1)
        pre_stack = pre_stack.reshape(-1)
        mask_stack = mask_stack.reshape(-1)

        precision, sen, spec, f1, acc, roc_auc, pr_auc = perform_metrics(pre_stack, label_stack, mask_stack)
        print(f'Precision: {precision} Sen: {sen} Spec:{spec} F1-score: {f1} Acc: {acc} ROC_AUC: {roc_auc} PR_AUC: {pr_auc}')
