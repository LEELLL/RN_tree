#-------------------------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
#-------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator as tg
import os
import math
import argparse
import scipy as sp
import scipy.stats
from resnet_model import resnet50,resnet18

from polyaxon_client import tracking
from loguru import logger

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 8)#64
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 1)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 15)#15
parser.add_argument("-e","--episode",type = int, default= 500000)
parser.add_argument("-t","--test_episode", type = int, default = 600)#600
parser.add_argument("-i","--test_interval", type = int, default = 1000)#5000

parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
parser.add_argument("--model_name_or_path", default='lilingling/fewshot_apt/', type=str)
parser.add_argument("--datas_name_or_path", default='lilingling/miniimagenet/', type=str)
args = parser.parse_args()

base_path = tracking.get_data_paths()['ceph'] #　base_path为 datauser@192.168.68.79:/atpcephdata/（固态地址）
args.output_dir = tracking.get_outputs_path()
logger_path = os.path.join(tracking.get_outputs_path(), 'log_runtime.log')
logger.add(logger_path)
logger.info('!!!!!!!!')

args.model_name_or_path = os.path.join(base_path, args.model_name_or_path)
args.datas_name_or_path = os.path.join(base_path, args.datas_name_or_path)

logger.info('start model')
logger.info('args.datas_name_or_path args.model_name_or_path'+args.datas_name_or_path + args.model_name_or_path)




# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = 1
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
TEST_INTERVAL = args.test_interval

LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

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

# class RelationNetwork(nn.Module):
#     """docstring for RelationNetwork"""
#     def __init__(self,input_size,hidden_size):
#         super(RelationNetwork, self).__init__()
#         self.layer1 = nn.Sequential(
#                         nn.Conv2d(128,64,kernel_size=3,padding=0),
#                         nn.BatchNorm2d(64, momentum=1, affine=True),
#                         nn.ReLU(),
#                         nn.MaxPool2d(2))
#         self.layer2 = nn.Sequential(
#                         nn.Conv2d(64,64,kernel_size=3,padding=0),
#                         nn.BatchNorm2d(64, momentum=1, affine=True),
#                         nn.ReLU(),
#                         nn.MaxPool2d(2))
#         self.fc1 = nn.Linear(input_size*3*3,hidden_size)
#         self.fc2 = nn.Linear(hidden_size,1)

#     def forward(self,x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = out.view(out.size(0),-1)
#         out = F.relu(self.fc1(out))
#         out = F.sigmoid(self.fc2(out))
#         return out

class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,hidden_size):
        # print(input_size,hidden_size) #64 8(原)　　resnet50:32 8    
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        # nn.Conv2d(64*2,64,kernel_size=3,padding=0),
                        # nn.Conv2d(64,32,kernel_size=3,padding=1), #resmet50 18
                        nn.Conv2d(FEATURE_DIM*2,FEATURE_DIM,kernel_size=3,padding=1), ## 16 8 resnet裁剪layer34+128+8 8 8
                        # nn.Conv2d(32,16,kernel_size=3,padding=1), #resmet50+cut64 #resnet50裁剪layer34+filter为256 (16,8,8)
                        # nn.BatchNorm2d(64, momentum=1, affine=True),
                        # nn.BatchNorm2d(32, momentum=1, affine=True),#resmet50 18
                        # nn.BatchNorm2d(16, momentum=1, affine=True),#resmet50cut64 #resnet50裁剪layer34+filter为256 (16,8,8)
                        nn.BatchNorm2d(FEATURE_DIM, momentum=1, affine=True),# 8 resnet裁剪layer34+128+8 8 8
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        # nn.Conv2d(64,64,kernel_size=3,padding=0),
                        # nn.Conv2d(32,32,kernel_size=3,padding=1),#resmet50 18
                        nn.Conv2d(FEATURE_DIM,FEATURE_DIM,kernel_size=3,padding=1),#resmet50cut64 #resnet50裁剪layer34+filter为256 (16,8,8)
                        # nn.Conv2d(16,16,kernel_size=3,padding=1),#resmet50cut64 #resnet50裁剪layer34+filter为256 (16,8,8)
                        # nn.BatchNorm2d(64, momentum=1, affine=True),
                        # nn.BatchNorm2d(32, momentum=1, affine=True),#resmet50 18
                        nn.BatchNorm2d(FEATURE_DIM, momentum=1, affine=True),#8 resnet裁剪layer34+128+8 8 8
                        # nn.BatchNorm2d(16, momentum=1, affine=True),#resmet50cut64 #resnet50裁剪layer34+filter为256 (16,8,8)
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        # self.fc1 = nn.Linear(input_size*3*3,hidden_size)
        # self.fc1 = nn.Linear(input_size,hidden_size)#resnet18　resnet50裁剪版　resnet50filter裁剪版
        self.fc1 = nn.Linear(input_size*4,hidden_size)#resnet50 #resnet50裁剪layer34+filter为256 (16,8,8)

        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        # print('RN:', x.size()) #RN: torch.Size([125, 128, 19, 19])   # resnet50: RN: torch.Size([250, 64, 8, 8])
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        # print('RelationNet out size:', out.size()) # torch.Size([125, 1])
        return out

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

def main():
    # Step 1: init data folders
    logger.info("init data folders")
    # init character folders for dataset construction
    metatrain_folders,metatest_folders = tg.mini_imagenet_folders()

    # Step 2: init neural networks
    logger.info("init neural networks")

    # feature_encoder = CNNEncoder()
    feature_encoder = resnet50()
    relation_network = RelationNetwork(FEATURE_DIM,RELATION_DIM)

    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    feature_encoder.cuda(GPU)
    relation_network.cuda(GPU)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)

    if os.path.exists(str(args.model_name_or_path + "miniimagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str(args.model_name_or_path + "miniimagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        logger.info("load feature encoder success")
    if os.path.exists(str(args.model_name_or_path + "miniimagenet_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str(args.model_name_or_path + "miniimagenet_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        logger.info("load relation network success")

    # Step 3: build graph
    logger.info("Training...")

    last_accuracy = 0.0

    for episode in range(EPISODE):

        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        # init dataset
        # sample_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training
        task = tg.MiniImagenetTask(metatrain_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
        batch_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True)

        # sample datas
        samples,sample_labels = sample_dataloader.__iter__().next()
        batches,batch_labels = batch_dataloader.__iter__().next()

        # calculate features
        sample_features = feature_encoder(Variable(samples).cuda(GPU)) # 5x64*5*5
        batch_features = feature_encoder(Variable(batches).cuda(GPU)) # 20x64*5*5

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        batch_features_ext = torch.transpose(batch_features_ext,0,1)
        # relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,FEATURE_DIM*2,19,19)
        relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,FEATURE_DIM*2,8,8) #resnet50


        relations = relation_network(relation_pairs).view(-1,CLASS_NUM*SAMPLE_NUM_PER_CLASS)

        mse = nn.MSELoss().cuda(GPU)
        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1)).cuda(GPU)
        loss = mse(relations,one_hot_labels)


        # training

        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm(feature_encoder.parameters(),0.5)
        torch.nn.utils.clip_grad_norm(relation_network.parameters(),0.5)

        feature_encoder_optim.step()
        relation_network_optim.step()


        if (episode+1)%100 == 0:
                logger.info("episode:"+str(episode+1) +" loss:"+ str(loss.data))

        if episode%TEST_INTERVAL == 0:

            # test
            logger.info("Testing...")
            accuracies = []
            for i in range(TEST_EPISODE):
                total_rewards = 0
                counter = 0
                task = tg.MiniImagenetTask(metatest_folders,CLASS_NUM,1,15)
                # task = tg.MiniImagenetTask(metatest_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
                sample_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=1,split="train",shuffle=False)

                num_per_class = 3
                test_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=num_per_class,split="test",shuffle=True)
                sample_images,sample_labels = sample_dataloader.__iter__().next()
                for test_images,test_labels in test_dataloader:
                    batch_size = test_labels.shape[0]
                    # calculate features
                    sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # 5x64
                    test_features = feature_encoder(Variable(test_images).cuda(GPU)) # 20x64

                    # calculate relations
                    # each batch sample link to every samples to calculate relations
                    # to form a 100x128 matrix for relation network
                    sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1,1,1)
                    test_features_ext = test_features.unsqueeze(0).repeat(1*CLASS_NUM,1,1,1,1)
                    test_features_ext = torch.transpose(test_features_ext,0,1)
                    # relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,19,19)
                    relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,8,8) #resnet50

                    relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

                    _,predict_labels = torch.max(relations.data,1)

                    rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)
                    counter += batch_size
                accuracy = total_rewards/1.0/counter
                accuracies.append(accuracy)

            test_accuracy,h = mean_confidence_interval(accuracies)

            logger.info("test accuracy:"+str(test_accuracy) + " h:" +str(h))

            if test_accuracy > last_accuracy:

                # save networks
                modelname = str("miniimagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")
                model_path = os.path.join(tracking.get_outputs_path(), modelname)
                torch.save(feature_encoder.state_dict(),model_path)

                modelname = str("miniimagenet_relation_network_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")
                model_path = os.path.join(tracking.get_outputs_path(), modelname)
                torch.save(relation_network.state_dict(),model_path)

                logger.info("save networks for episode: " + str(episode))

                last_accuracy = test_accuracy





if __name__ == '__main__':
    main()
