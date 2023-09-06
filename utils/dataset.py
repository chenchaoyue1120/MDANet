import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

class FundusSeg_Loader(Dataset):
    def __init__(self, data_path, is_train, dataset_name):
        # 初始化函数，读取所有data_path下的图片
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.tif'))
        self.labels_path = glob.glob(os.path.join(data_path, 'label/*.tif'))
        self.masks_path = glob.glob(os.path.join(data_path, 'mask/*.tif'))
        self.is_train = is_train

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        if self.is_train == 1:
            label_path = image_path.replace('image', 'label')
            if self.dataset_name == "drive":
                label_path = label_path.replace('training', 'manual1') # DRIVE
            if self.dataset_name == "chase":
                label_path = label_path.replace('.tif', '_1stHO.tif') #CHASE
            if self.dataset_name == "idrid":
                label_path = label_path.replace('.tif', '_EX.tif') # IDRID _EX
        else:
            if self.dataset_name == "drive":
                label_path = image_path.replace('image', 'label')
                label_path = label_path.replace('test.tif', 'manual1.tif') #DRIVE
                mask_path = image_path.replace('image', 'mask')
                mask_path = mask_path.replace('test.tif', 'test_mask.tif') # DRIVE
                mask = Image.open(mask_path)
            if self.dataset_name == "chase":
                label_path = image_path.replace('image', 'label')
                label_path = label_path.replace('.tif', '_1stHO.tif') #CHASE
                mask_path = image_path.replace('image', 'mask')
                mask = Image.open(mask_path)
            if self.dataset_name == "em":
                label_path = image_path.replace('image', 'label')
                mask = Image.new('L', (512, 512), color='white')
            if self.dataset_name == "stare":
                label_path = image_path.replace('image', 'label')
                mask_path = image_path.replace('image', 'mask')
                mask = Image.open(mask_path)
            if self.dataset_name == "idrid":
                label_path = image_path.replace('image', 'label')
                label_path = label_path.replace('.tif', '_EX.tif') # IDRID _EX
                mask = Image.new('L', (1072, 712), color='white')

        if self.dataset_name == "em":
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.open(image_path)
        label = Image.open(label_path)
        label = label.convert('L')

        # 这一步的目的是干啥
        if self.is_train == 0: # TEST
            if self.dataset_name == "drive":
                pad_to_h = 584
                pad_to_w = 584
                image, label, mask = self.padding_image(image, label, mask, pad_to_h, pad_to_w)
            if self.dataset_name == "chase":
                pad_to_h = 1024
                pad_to_w = 1024
                image, label, mask = self.padding_image(image, label, mask, pad_to_h, pad_to_w)
                #image = TF.crop(image, 256, 256, 512, 512)
                #label = TF.crop(label, 256, 256, 512, 512)
                #mask = TF.crop(mask, 256, 256, 512, 512)
            if self.dataset_name == "stare":
                pad_to_h = 700
                pad_to_w = 700
                image, label, mask = self.padding_image(image, label, mask, pad_to_h, pad_to_w)
            if self.dataset_name == "em":
                pad_to = 0 # Do nothing
            if self.dataset_name == "idrid":
                pad_to_h = 1072
                pad_to_w = 1072
                image, label, mask = self.padding_image(image, label, mask, pad_to_h, pad_to_w)

        if self.is_train == 1:
            image, label = self.randomRotation(image, label)


            if np.random.random_sample() <= 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)

            if np.random.random_sample() <= 1:
                # random.uniform(),从一个均匀分布[low,high)中随机采样
                crop_size = 256
                w = random.uniform(0, image.size[0]-crop_size)
                h = random.uniform(0, image.size[1]-crop_size)
                image = TF.crop(image, w, h, crop_size, crop_size)
                label = TF.crop(label, w, h, crop_size, crop_size)

        image = np.asarray(image)
        label = np.asarray(label)

        image = image.transpose(2, 0, 1)
        label = label.reshape(1, label.shape[0], label.shape[1])
        label = np.array(label)
        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label[label > 1] = 255
            label = label / 255

        sp = image_path.split('/')
        filename = sp[len(sp)-1]
        filename = filename[0:len(filename)-4] # del .tif

        if self.is_train == 1:
            return image, label, filename
        else:
            return image, label, np.array(mask), filename

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

    def randomRotation(self, image, label, mode=Image.BICUBIC):
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, mode), label.rotate(random_angle, Image.NEAREST)

    def padding_image(self,image, label, mask, pad_to_h, pad_to_w):
        #新建长宽608像素，背景色为（0, 0, 0）的画布对象,即背景为黑，RGB是彩色三通道图像，P是8位图像
        new_image = Image.new('RGB', (pad_to_w, pad_to_h), (0, 0, 0))
        new_label = Image.new('P', (pad_to_w, pad_to_h), (0, 0, 0))
        new_mask = Image.new('P', (pad_to_w, pad_to_h), (0, 0, 0))
        # 把新建的图像粘贴在原图上
        new_image.paste(image, (0, 0))
        new_label.paste(label, (0, 0))
        new_mask.paste(mask, (0, 0))
        return new_image, new_label, new_mask
