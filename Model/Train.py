#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/26 19:38
# @Author  : Tom SONG 
# @Mail    : xdmyssy@gmail.com
# @File    : Train.py
# @Software: PyCharm


import os
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import time
import pandas as pd

def train(net,
          max_epoch,
          optimizer,
          scheduler,
          device,
          loader_train,
          loader_valid,
          ckp_dir,
          log_interval,
          val_interval):



    if not os.path.exists(ckp_dir):
        os.mkdir(ckp_dir)

    model_path = os.path.join(ckp_dir, 'net.pth')
    loss_path = os.path.join(ckp_dir, 'loss.png')

    # net.initialize_weights()

    criterion = nn.CrossEntropyLoss()

    net.to(device)
    criterion.to(device)

    train_curve = list()
    valid_curve = list()

    start_time = time.time()

    train_time=0

    iteration=0

    train_iteration=list()
    train_error=list()
    train_loss=list()

    valid_itertation=list()
    valid_error=list()
    valid_loss=list()

    for epoch in range(max_epoch):

        loss_mean = 0.
        correct = 0.
        total = 0.

        net.train()

        for i, (points, vectors, labels) in enumerate(loader_train):

            train_start= time.time()

            # input configuration
            points = points.to(device)
            vectors = vectors.to(device)
            labels = labels.to(device)

            # forward
            outputs = net(points, vectors)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            # update weights
            optimizer.step()

            train_end=time.time()

            train_time+=(train_end-train_start)

            # analyze results
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().sum().cpu().numpy()

            # print results
            loss_mean += loss.item()
            train_curve.append(loss.item())

            iteration += 1

            if (i + 1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, max_epoch, i + 1, len(loader_train), loss_mean, correct / total))

                train_iteration.append(iteration)
                train_error.append(1-correct/total)
                train_loss.append(loss_mean)

                loss_mean = 0.





        scheduler.step()  # renew LR

        # validate the model
        if (epoch + 1) % val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            net.eval()
            with torch.no_grad():
                for j, (points, vectors,labels) in enumerate(loader_valid):
                    # input configuration
                    points = points.to(device)
                    vectors = vectors.to(device)
                    labels = labels.to(device)

                    outputs = net(points, vectors)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().sum().cpu().numpy()

                    loss_val += loss.item()

                loss_val_epoch = loss_val / len(loader_valid)
                valid_curve.append(loss_val_epoch)
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, max_epoch, j + 1, len(loader_valid), loss_val_epoch, correct_val / total_val))

                valid_itertation.append(iteration)
                valid_error.append(1-correct_val / total_val)
                valid_loss.append(loss_val_epoch)


    end_time = time.time()
    # save model
    torch.save(net.state_dict(), model_path)

    df_time = pd.DataFrame({
        'time': [end_time - start_time],
        'train_time': [train_time]
    })
    df_time.to_csv(os.path.join(ckp_dir, 'time.csv'))

    train_x = range(len(train_curve))
    train_y = train_curve

    train_iters = len(loader_train)
    valid_x = np.arange(1,
                        len(valid_curve) + 1) * train_iters * val_interval - 1  # valid records epochloss，need to be converted to iterations
    valid_y = valid_curve

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.legend(loc='upper right')
    plt.ylabel('loss value')
    plt.xlabel('Iteration')
    plt.savefig(loss_path)

    # save loss
    df1 = pd.DataFrame({
        'train_x': train_x,
        'train_y': train_y,

    })
    df1.to_csv(os.path.join(ckp_dir, 'loss_train.csv'))

    df2 = pd.DataFrame({
        'valid_x': valid_x,
        'valid_y': valid_y,
    })
    df2.to_csv(os.path.join(ckp_dir, 'loss_validation.csv'))


    df3=pd.DataFrame({
        'train_x':train_iteration,
        'train_y':train_loss,
        'train_error':train_error
    })

    df3.to_csv(os.path.join(ckp_dir, 'loss_train_{}_{}.csv'.format(max_epoch, log_interval)))

    df4 = pd.DataFrame({
        'valid_x': valid_itertation,
        'valid_y': valid_loss,
        'valid_error': valid_error
    })

    df4.to_csv(os.path.join(ckp_dir, 'loss_valid_{}_{}.csv'.format(max_epoch, val_interval)))





if __name__ == '__main__':
    pass
