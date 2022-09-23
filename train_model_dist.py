from __future__ import print_function, division
import os
import torch
from optparse import OptionParser
import torch
from torch import optim
import sys
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from unet_test5_dist import ProbabilisticUnet
from utils1 import l2_regularisation
from torchvision import transforms
from data_gen_cardiac_dist import ListDataset
from eval import eval_net
import torch.nn as nn
import  numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]='0'


print('==> Preparing data..')
'''
transform = transforms.Compose([transforms.Resize(224),
    transforms.ToTensor()
])

transform = transforms.Compose([#transforms.CenterCrop(550),
                                #transforms.RandomResizedCrop(210, scale=(0.9, 1.1), ratio=(1,1), interpolation=2),
    transforms.RandomRotation(5),
    transforms.Resize(224),
    transforms.ToTensor()
])
transform1 = transforms.Compose([#transforms.Resize(224),
    transforms.Resize(224),
    transforms.ToTensor()
])
'''
for k in range(10):
    trainset = ListDataset(root=['gt_img', 'index_img'],
                           list_file='open_cardic_1.txt', state='Train',  k=k)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=24, shuffle=True, num_workers=0)

    testset = ListDataset(root=['gt_img', 'index_img'],
                          list_file='open_cardic_1.txt', state='Valid',  k=k)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

    net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32, 64, 128, 256], latent_dim=32,
                            no_convs_fcomb=4, beta=10.0)
    net.cuda()
#############################################################################################
    optimizer = torch.optim.SGD(params=net.parameters(),  lr=0.1, momentum=0.9, dampening=0.5, weight_decay=0.0001, nesterov=False)
    #torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
    mx_val = 0
    epochs = 50
    T_max =5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)
    dir_checkpoint = 'my_method/'
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)
    for epoch in range(epochs):
        for step, (patch, mask,bdm) in enumerate(trainloader):
            #patch = patch[:, 1, :, :].unsqueeze(1)
            true_masks_1 = np.array(mask)
            true_masks = true_masks_1#[:, :1, :, :]
            true_masks = np.where(true_masks == 2/ 255, 1, 0)  # t_masks[t_masks != v+1] = 0
            # np.savetxt("a.txt",t_masks[0,:,:], fmt="%f", delimiter=" ")
            true_masks = torch.from_numpy(true_masks.astype(np.float32))
            true_bdm_1 = np.array(bdm)
            true_bdm = torch.from_numpy(true_bdm_1.astype(np.float32))
            #true_bdm = true_bdm.unsqueeze(1)

            patch = patch.cuda()
            true_masks = true_masks.cuda()
            true_bdm = true_bdm.cuda()

            net.forward(patch, true_masks, training=True)
            elbo = net.elbo(true_masks,true_bdm)
            # print(pred_mask.shape)
            reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior)
            loss = 1e-5 * reg_loss - elbo
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch {} finished !'.format(epoch + 1),
              'Loss: {}'.format(loss / (step + 1)))  # ,'EncLoss: {}'.format(enc_loss / i)
        val_dice = eval_net(net, testloader, gpu=True)
            #print('Validation Dice Coeff: {}  '.format(val_dice))
        if val_dice > mx_val:
            mx_val=val_dice
            torch.save(net.state_dict(),dir_checkpoint + 'CP{}_{}.pth'.format(k+1,epoch + 1))
            print('Validation Dice Coeff: {}_{}_{}  '.format(k+1,epoch + 1,val_dice))
            print('Checkpoint {}_{} saved !'.format(k+1,epoch + 1))
        torch.save(net.state_dict(), dir_checkpoint + 'latest_{}.pth'.format(k + 1))