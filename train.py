# -*- coding: utf-8 -*-

import easydict
import os
import time
from network import TransformerWithCnnModel
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn

# only specify one of our four networks for simplicity.
from net1 import Net1
from dataset import SingleDoubleDataset, SingleDoubleDatasetValid, SingleDoubleDatasetTest


def train(dataloader, epoch):
    print('[Epoch %d]' % (epoch+1))
    
    criterion = nn.CrossEntropyLoss() 
    running_loss = 0.0
    
    start_time = time.time()
    for batch_idx, samples in enumerate(dataloader): 
        Ys, qvectors, labels = samples[0].to(device), samples[1].to(device), samples[2].to(device) 
        Ys = Ys.float() 
        Ys = torch.unsqueeze(Ys, axis=1) 
        qvectors = qvectors.float() 

        # zero the parameter gradients
        optimizer.zero_grad() 

        # forward + backward + optimize
        outputs = net(Ys, qvectors, None) 

        loss = criterion(outputs, labels) 
        loss.backward() 
        optimizer.step() 

        # print statistics
        running_loss += loss.item()
        if batch_idx % args.loss_interval == (args.loss_interval-1):
            elapsed = time.time() - start_time
            print('[%d, %5d] loss: %.6f in %.3f seconds' %
                (epoch + 1, batch_idx + 1, running_loss / args.loss_interval, elapsed))
            
            running_loss = 0.0
            start_time = time.time()

        if args.split_training:
            if batch_idx > 10*args.loss_interval:
                break
  
    return running_loss / args.loss_interval


def valid(dataloader, epoch):
    print('**Validation**')
    criterion = nn.CrossEntropyLoss()
    valid_loss = 0.0

    classes = ('single', 'double')
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    class_acc = list(0. for i in range(2))

    with torch.no_grad():
      for samples in dataloader:
        Ys, qvectors, labels = samples[0].to(device), samples[1].to(device), samples[2].to(device)
        Ys = Ys.float()
        Ys = torch.unsqueeze(Ys, axis=1)
        qvectors = qvectors.float()
        
        # feed forward
        outputs = net(Ys, qvectors, None)
        
        loss = criterion(outputs, labels)

        # print statistics
        valid_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        
        for i in range(args.batch_size):
          label = labels[i]
          class_correct[label] += c[i].item()
          class_total[label] += 1
    
      for i in range(2):
        class_acc[i] = 100 * class_correct[i] / class_total[i]
        print('Accuracy of %5s : %.2f %%' % ( classes[i], class_acc[i]))
        
      valid_loss = valid_loss / len(dataloader)
      total_acc = (class_acc[0]+class_acc[1])/2
      print('Validation Loss: %.6f' % valid_loss)
      print('Accuracy of %5s : %.2f %%' % ('total', total_acc))
      
      # calculate valid best
      if total_acc > valid_best['total_acc']:
        valid_best['total_acc'] = total_acc
        valid_best['single_acc'] = class_acc[0]
        valid_best['double_acc'] = class_acc[1]
        valid_best['epoch'] = epoch+1

        #save model
        os.makedirs('./model', exist_ok=True)
        torch.save(net.state_dict(), './model/net-best.pth')
      
      return valid_loss, total_acc


def test(dataloader):
    # print(dataloader)

    classes = ('single', 'double')
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    class_acc = list(0. for i in range(2))

    with torch.no_grad():
        for samples in dataloader:
            Ys, qvectors, labels = samples[0].to(device), samples[1].to(device), samples[2].to(device)
            Ys = Ys.float()
            Ys = torch.unsqueeze(Ys, axis=1)
            qvectors = qvectors.float()

            # feed forward
            outputs = net(Ys, qvectors, None)

            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            for i in range(args.batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    for i in range(2):
        class_acc[i] = 100 * class_correct[i] / class_total[i]
        print('Accuracy of %5s : %.2f %%' % ( classes[i], class_acc[i]))

    total_acc = (class_acc[0]+class_acc[1])/2
    print('Accuracy of %5s : %.2f %%' % ('total', total_acc))

    return total_acc


if __name__ == "__main__":
    args = easydict.EasyDict({
      "batch_size": 32,
      "epoch": 10,
      "loss_interval": 500,
      "split_training": False,
      "data_path": './jpeg_data',
    })

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = SingleDoubleDataset(args.data_path)
    valid_dataset = SingleDoubleDatasetValid(args.data_path)
    test_dataset = SingleDoubleDatasetTest(args.data_path)
  
    input_size = 120
    conv_output_size = 120
    n_head = 8
    en_hidden_size = 256
    en_n_layers = 3
    drop_out = 0.1

    net = TransformerWithCnnModel(input_size, conv_output_size, n_head, en_hidden_size, en_n_layers, drop_out)
    net.to(device)
    
    optimizer = torch.optim.Adam(net.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    valid_best = dict(epoch=0, single_acc=0, double_acc=0, total_acc=0)

    train_losses = []
    valid_losses = []
    valid_accuracy = []

    for epoch in range(0, args.epoch):
        epoch_start_time = time.time()
        net.train()
        loss = train(train_dataloader, epoch)
        
        net.eval()
        valid_loss, acc = valid(valid_dataloader, epoch)
        print("end of the epoch in %f seconds" % (time.time() - epoch_start_time))
        
        scheduler.step(valid_loss)

        valid_losses.append(valid_loss)
        valid_accuracy.append(acc)

        torch.save(net.state_dict(), './model/network-recent.pth')

    # print valid best
    print('[Best epoch: %d' % valid_best['epoch'])
    print('Accuracy of %5s: %.2f %%' % ('single', valid_best['single_acc']))
    print('Accuracy of %5s: %.2f %%' % ('double', valid_best['double_acc']))
    print('Accuracy of %5s: %.2f %%' % ('total', valid_best['total_acc']))
    print('Done')

    net.eval()
    test_accuracy = test(test_dataloader)

    print('[Test Set]')
    print('Total Accuracy: ', test_accuracy)

    # show result graphs
    fig = plt.figure()
    columns = 2
    rows = 1
    
    fig.add_subplot(rows, columns, 1)
    plt.title('train loss')
    plt.plot(train_losses)

    fig.add_subplot(rows, columns, 2)
    plt.title('valid loss')
    plt.plot(valid_losses)

    fig.add_subplot(rows, columns, 3)
    plt.title('valud accuracy')
    plt.plot(valid_accuracy)

    fig.add_subplot(rows, columns, 4)
    plt.title('test accuracy')
    plt.plot(test_accuracy)

    plt.show()