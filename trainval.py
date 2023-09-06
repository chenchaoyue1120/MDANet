from model.mda_model import MDANet
from utils.dataset import FundusSeg_Loader
from torch import optim
import torch.nn as nn
import torch
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


dataset_name = "drive" #

if dataset_name == "drive":
    train_data_path = "DRIVE/drive_train/"
    valid_data_path = "DRIVE/drive_test/"
    N_epochs = 250
    lr_decay_step = [180]
#    lr_decay_step = [20, 200]
    Init_lr = 0.001
    batch_size = 8

def train_net(net, device, epochs=N_epochs, batch_size=batch_size, lr=Init_lr):
    # 加载训练集
    train_dataset = FundusSeg_Loader(train_data_path, 1, dataset_name)
    valid_dataset = FundusSeg_Loader(valid_data_path, 0, dataset_name)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=4, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)
    print('Train images: %s' % len(train_loader))
    print('Valid  images: %s' % len(valid_loader))

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=lr_decay_step,gamma=0.1)
    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = nn.BCEWithLogitsLoss()
    criterion3 = nn.BCEWithLogitsLoss()
    criterion4 = nn.BCEWithLogitsLoss()
    criterion5 = nn.BCEWithLogitsLoss()
    # 训练epochs次
    # 求最小值，所以初始化为正无穷
    best_loss = float('inf')
    train_loss_list = []
    val_loss_list = []
    for epoch in range(epochs):
        # 训练模式
        net.train()
        train_loss = 0
        print(f'Epoch {epoch + 1}/{epochs}')
        # SGD
        with tqdm(total=train_loader.__len__()) as pbar:
            for i, (image, label, filename) in enumerate(train_loader):
                optimizer.zero_grad()
                # 将数据拷贝到device中
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                # 使用网络参数，输出预测结果
                s1, s2, s3, s4, pred = net(image)
                # 计算loss
                loss1 = criterion1(s1, label)
                loss2 = criterion2(s2, label)
                loss3 = criterion3(s3, label)
                loss4 = criterion4(s4, label)
                loss5 = criterion5(pred, label)
                loss = 0.1*loss1 + 0.1*loss2 + 0.1*loss3 + 0.1*loss4 + loss5
                train_loss = train_loss + loss.item()
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=float(loss.cpu()), epoch=epoch)
                pbar.update(1)

        train_loss_list.append(train_loss / i)

        # Validation
        net.eval()
        val_loss = 0
        for i, (image, label, mask, filename) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            s1, s2, s3, s4, pred = net(image)
            loss = criterion5(pred, label)
            val_loss = val_loss + loss.item()

        scheduler.step()
        
        # net.state_dict()就是用来保存模型参数的
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(), './snapshot.pth')
            print('saving model............................................')
        
        val_loss_list.append(val_loss / i)
        print('Loss/valid', val_loss / i)
        sys.stdout.flush()
    # torch.save(net.state_dict(), './tmp_cat.pth')


if __name__ == "__main__":
    # 选择设备cuda
    device = torch.device('cuda')
    # 加载网络，图片单通道1，分类为1。
    net = MDANet(n_channels=3, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 开始训练
    train_net(net, device)
