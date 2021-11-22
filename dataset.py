# -*- coding: utf-8 -*-
"""dataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FJJTY_-eUjyRTcSEfUz8p3Gye1yJHPuw
"""

from PIL import Image
from PIL import JpegImagePlugin
import os
import glob
import torch
import numpy as np


def read_q_table(file_name):
    jpg = JpegImagePlugin.JpegImageFile(file_name)
    qtable = JpegImagePlugin.convert_dict_qtables(jpg.quantization)
    Y_qtable = qtable[0]
    Y_qtable_2d = np.zeros((8, 8)) 

    qtable_idx = 0
    for i in range(0, 8):
        for j in range(0, 8):
            Y_qtable_2d[i, j] = Y_qtable[qtable_idx]
            qtable_idx = qtable_idx + 1

    return Y_qtable_2d


class SingleDoubleDataset(torch.utils.data.Dataset): 
    def __init__(self, data_path):
        train_dir_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

        file_list_double = list()
        for dir_idx in train_dir_list:
            double_path = os.path.join(data_path, 'double')
            file_list_double_ = glob.glob(double_path+'/'+dir_idx+'/*.jpg')
            file_list_double.extend(file_list_double_)

        double_label = [1]*len(file_list_double)
        
        file_list_single = list()
        for dir_idx in train_dir_list:
            single_path = os.path.join(data_path, 'single')
            file_list_single_ = glob.glob(single_path+'/'+dir_idx+'/*.jpg')
            file_list_single.extend(file_list_single_)

        single_label = [0]*len(file_list_single)
        
        self.file_list = file_list_double + file_list_single
        self.label_list = double_label + single_label

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        im = Image.open(self.file_list[idx])
        im = im.convert('YCbCr')
        
        Y = np.array(im)[:,:,0]

        q_table = read_q_table(self.file_list[idx])
        q_vector = q_table.flatten()
        label = self.label_list[idx]

        item = (Y, q_vector, label)
        return item


class SingleDoubleDatasetValid(SingleDoubleDataset):
    def __init__(self, data_path):
        valid_dir_list = ['13', '14', '15', '16']

        file_list_double = list()
        for dir_idx in valid_dir_list:
            double_path = os.path.join(data_path, 'double')
            file_list_double_ = glob.glob(double_path+'/'+dir_idx+'/*.jpg')
            file_list_double.extend(file_list_double_)

        double_label = [1]*len(file_list_double)
        
        file_list_single = list()
        for dir_idx in valid_dir_list:
            single_path = os.path.join(data_path, 'single')
            file_list_single_ = glob.glob(single_path+'/'+dir_idx+'/*.jpg')
            file_list_single.extend(file_list_single_)            

        single_label = [0]*len(file_list_single)
        
        self.file_list = file_list_double + file_list_single
        self.label_list = double_label + single_label


class SingleDoubleDatasetTest(SingleDoubleDataset):
    def __init__(self, data_path):
        test_dir_list = ['17', '18', '19', '20']

        file_list_double = list()
        for dir_idx in test_dir_list:
            double_path = os.path.join(data_path, 'double')
            file_list_double_ = glob.glob(double_path+'/'+dir_idx+'/*.jpg')
            file_list_double1.extend(file_list_double1_)

        double_label = [1]*len(file_list_double)
        
        file_list_single = list()
        for dir_idx in test_dir_list:
            single_path1 = os.path.join(data_path, 'single')
            file_list_single_ = glob.glob(single_path+'/'+dir_idx+'/*.jpg')
            file_list_single.extend(file_list_single_)

        single_label = [0]*len(file_list_single)
        
        self.file_list = file_list_double + file_list_single
        self.label_list = double_label + single_label