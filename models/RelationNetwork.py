#-------------------------------------
# Source code extracted and modified from:
# Project: Learning to Compare: Relation Network for Few-Shot Learning
#-------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import argparse
import scipy as sp
import scipy.stats


class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.view(out.size(0),-1)
        return out # 64

### Modification to adjust 40x40 images. 256 in_features for self.fc1
class RelationNetwork_n(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size,ds_name):
        super(RelationNetwork_n, self).__init__()

        if True:
            self.layer1 = nn.Sequential(
                            nn.Conv2d(128,64,kernel_size=3,padding=1),
                            nn.BatchNorm2d(64, momentum=1, affine=True),
                            nn.ReLU(),
                            nn.MaxPool2d(2))
            self.layer2 = nn.Sequential(
                            nn.Conv2d(64,64,kernel_size=3,padding=1),
                            nn.BatchNorm2d(64, momentum=1, affine=True),
                            nn.ReLU(),
                            nn.MaxPool2d(2))
            self.fc1 = nn.Linear(input_size*2*2,hidden_size)
            self.fc2 = nn.Linear(hidden_size,1)

        # elif ds_name.startswith("omniglot"):
        #     self.layer1 = nn.Sequential(
        #                     nn.Conv2d(128,64,kernel_size=3,padding=1),
        #                     nn.BatchNorm2d(64, momentum=1, affine=True),
        #                     nn.ReLU(),
        #                     nn.MaxPool2d(2))
        #     self.layer2 = nn.Sequential(
        #                     nn.Conv2d(64,64,kernel_size=3,padding=1),
        #                     nn.BatchNorm2d(64, momentum=1, affine=True),
        #                     nn.ReLU(),
        #                     nn.MaxPool2d(2))
        #     self.fc1 = nn.Linear(input_size,hidden_size)
        #     self.fc2 = nn.Linear(hidden_size,1)
        # else:
        #     self.layer1 = nn.Sequential(
        #                     nn.Conv2d(128,64,kernel_size=3,padding=0),
        #                     nn.BatchNorm2d(64, momentum=1, affine=True),
        #                     nn.ReLU(),
        #                     nn.MaxPool2d(2))
        #     self.layer2 = nn.Sequential(
        #                     nn.Conv2d(64,64,kernel_size=3,padding=0),
        #                     nn.BatchNorm2d(64, momentum=1, affine=True),
        #                     nn.ReLU(),
        #                     nn.MaxPool2d(2))
        #     self.fc1 = nn.Linear(input_size*3*3,hidden_size)
        #     self.fc2 = nn.Linear(hidden_size,1)


    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out


class RelationNetwork(nn.Module):
    def __init__(self, feature_dimension, ds_name, spc, input_size):
        super(RelationNetwork, self).__init__()
        FEATURE_DIM = 64
        RELATION_DIM = 8

        self.ds_name = ds_name

        self.feature_encoder = CNNEncoder()
        self.relation_network = RelationNetwork_n(FEATURE_DIM, RELATION_DIM, ds_name=ds_name)

        self.feature_encoder.apply(weights_init)
        self.relation_network.apply(weights_init)

        self.feature_encoder.cuda()
        self.relation_network.cuda()

        ## Hight/Width of net [19,19] for 84x84; [5,5] for 28x28
        # self.H = ( (input_size - 28) / (84 - 28) ) * (19 - 5) + 5
        # self.W = self.H


    def forward(self, sample_images, sample_labels, query_images, query_labels, train, SAMPLE_NUM_PER_CLASS, CLASS_NUM):
        # Process all batch
        acc, loss, out, dists = self.execute_nets___(
            sample_images, sample_labels, query_images, query_labels,
            train, SAMPLE_NUM_PER_CLASS, CLASS_NUM
        )
        return torch.tensor(acc), loss, out, dists



    ## As this is for RelationNetworks, we just element-wise sum embeddings of each class
    def compute_prototypes(self, support_features, support_labels):
        """
        Compute class prototypes from support features and labels
        Args:
            support_features: for each instance in the support set, its feature vector
            support_labels: for each instance in the support set, its label

        Returns:
            for each label of the support set, the sum feature vector of instances with this label
        """

        n_way = len(torch.unique(support_labels))
        
        return torch.cat(
            [
                support_features[torch.nonzero(support_labels == label)].sum(0)
                for label in range(n_way)
            ]
        )



    def execute_nets___(self, sample_images_, sample_labels_, query_images_, query_labels_, train, SAMPLE_NUM_PER_CLASS, CLASS_NUM):
        FEATURE_DIM = 64 
        if train:
            self.feature_encoder.train()
            self.relation_network.train()
            self.train()
        else:
            self.feature_encoder.eval()
            self.relation_network.eval()
            self.eval()

        batch_size = sample_images_.shape[0]
        B = 1
        all_rewards, all_loss, all_predicted_labels, all_dists = [], [], [], []

        for b in range(batch_size):
            sample_images, query_images = sample_images_[b], query_images_[b] 
            sample_images = sample_images.view(-1, sample_images.shape[1], sample_images.shape[2], sample_images.shape[3])    # [CLASS_NUM*SPC, 3, 28, 28]
            query_images  = query_images.view(-1, query_images.shape[1], query_images.shape[2], query_images.shape[3])        # [B*CLASS_NUM, 3, 28, 28]
            sample_labels, query_labels  = sample_labels_[b], query_labels_[b]
            

            
            #### Interp
            # if not self.ds_name.startswith("omniglot"): # Not anymore, as the sizes have been forced to 40x40
            # if sample_images.shape[3] != 84 and sample_images.shape[3] != 28:
            #     sample_images = F.interpolate(sample_images, size=(28, 28), mode='bilinear', align_corners=False)
            #     query_images = F.interpolate(query_images, size=(28, 28), mode='bilinear', align_corners=False)            
            
            # 1st net, encoder
            sample_features = self.feature_encoder(sample_images.cuda())   # [CLASS_NUM*SPC, 64, 5, 5]

            W, H = sample_features.shape[2], sample_features.shape[3]

            # sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,W,H) # [CLASS_NUM, SPC, 64, 5, 5]
            # Join features of each sample class per batch
            sample_features = self.compute_prototypes(sample_features.view(CLASS_NUM*SAMPLE_NUM_PER_CLASS, FEATURE_DIM,W,H), sample_labels.view(-1))
            # sample_features = torch.sum(sample_features,1).squeeze(1).view(CLASS_NUM, FEATURE_DIM,5,5)   # Remove SPC [CLASS_NUM, 64, 5, 5]


            query_features  = self.feature_encoder(query_images.cuda())    # [B*CLASS_NUM, 64, 5, 5]
            
            Q = query_features.shape[0] # Number of querys per batch

            # Creates correspondences
            sample_exp = sample_features.unsqueeze(dim=0).expand(query_features.shape[0],-1,-1,-1,-1)  # [B*CLASS_NUM, CLASS_NUM, 64, 5, 5]
            query_exp = query_features.unsqueeze(dim=1).expand(-1,sample_features.shape[0],-1,-1,-1)      # [CLASS_NUM, Q, 64, 5, 5]
            # query_exp = torch.transpose(query_exp,0,1)
            # sample_exp = sample_features.unsqueeze(1).expand(-1, ALLQ, -1, -1, -1)  # [Q, CLASS_NUM, 64, 5, 5]
            # query_exp = query_features.unsqueeze(0).expand(S, -1, -1, -1, -1)       # [Q, CLASS_NUM, 64, 5, 5]
            



            # Concatenation matrix
            relation_pairs = torch.cat((sample_exp, query_exp), dim=2)            # [B*CLASS_NUM, CLASS_NUM, 128, 5, 5]
            relation_pairs = relation_pairs.view(-1, FEATURE_DIM * 2, *query_features.shape[2:]) # [CLASS_NUM, 128, 5, 5]
            try:
                relations = self.relation_network(relation_pairs).view(-1, sample_features.shape[0])       # [CLASS_NUM , CLASS_NUM]
            except:
                breakpoint()

            # Preds
            _, predicted_labels = torch.max(relations, dim=1)  # Índices de muestras más similares [ALLQ]
            # predicted_labels = torch.gather(sample_labels.view(-1).cuda(), 0, predicted_labels)


            query_labels_flat = query_labels.view(-1)
            # print("Relations", relations)
            rewards = [1. if predicted_labels[j]==query_labels_flat[j] else 0. for j in range(len(query_labels_flat))]

            # Loss MSE
            mse = nn.MSELoss().cuda()
            one_hot_labels = Variable(torch.zeros(B*CLASS_NUM, CLASS_NUM).cuda().scatter_(1, query_labels.view(-1,1), 1)).cuda()

            loss = mse(relations,one_hot_labels)

            if train:
                self.feature_encoder.zero_grad()
                self.relation_network.zero_grad()
                loss.backward()
                self.feature_encoder_optim.step()
                self.relation_network_optim.step()
                torch.nn.utils.clip_grad_norm_(self.feature_encoder.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.relation_network.parameters(), 0.5)

            all_rewards.append(rewards)
            all_loss.append(loss.cpu().item())
            all_predicted_labels += list(predicted_labels.cpu().numpy())
            all_dists.append(relations.cpu().detach().numpy())

        # print("\nPRED", all_predicted_labels, "\nGT__", query_labels_.view(-1))
        # return np.mean(all_rewards), np.mean(all_loss), torch.from_numpy(np.array(all_predicted_labels)), all_dists
        return np.mean(all_rewards), np.mean(all_loss), None, all_dists #None





    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0*np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
        return m,h


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

   
