# Test Script to run forward on sparse resnet
import argparse
import os
import sys
import pickle
import resource
import traceback
import logging
from collections import defaultdict

import _init_paths  # pylint: disable=unused-import
import nn as mynn
import utils.net as net_utils
import utils.misc as misc_utils
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from datasets.roidb import combined_roidb_for_training
from roi_data.loader import RoiDataLoader, MinibatchSampler, BatchSampler, collate_minibatch
from modeling.model_builder import Generalized_RCNN
from utils.detectron_weight_helper import load_detectron_weight
from utils.logging import setup_logging
from utils.timer import Timer
from utils.training_stats import TrainingStats
import numpy as np
import yaml
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import cv2
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader



#to Vis
import numpy as np
import time
import random
import sparseconvnet as scn
from torch.nn import Module



def deal_card(deck):
    if len(deck) < 1:
        print("Going to have a problem. Dealing from empty deck")
        return None
    card = -1
    idx = np.random.randint(0,len(deck))
    card = deck[idx]
    deck.pop(idx)
    return card

def shuffle(full_deck):
    copy_deck = full_deck.copy()
    new_deck = []
    while (len(copy_deck) > 0):
        new_deck.append( deal_card(copy_deck) )
    return new_deck

def hand_total(hand):
    total = 0
    aces_count = 0
    for card in hand:
        if card > 9:
            total = total + 10
        elif card > 1:
            total = total + card
        else:
            aces_count = aces_count + 1
    if (aces_count != 0):
        if ((total + 11 + aces_count - 1) > 21):
            return (total + aces_count)
        else:
            # This is not strictly true, could have lower value
            return (total + 11 + aces_count -1)
    else:
        return total

def winner(your_total, dealer_total):
    if your_total > 21:
        return 0
    elif dealer_total > 21:
        return 1
    elif your_total > dealer_total:
        return 1
    elif your_total == dealer_total:
        return 2 #no contest
    else:
        return 0

def deal_table(deck, num_extras):
    assert len(deck) > (num_extras+2)*2
    your_hand = []
    dealer_hand = []
    extra_hands = [[] for i in range(num_extras)]
    your_hand.append(deal_card(deck))
    your_hand.append(deal_card(deck))
    dealer_hand.append(deal_card(deck))
    for hand in extra_hands:
        hand.append(deal_card(deck))
        hand.append(deal_card(deck))
    return your_hand, dealer_hand, extra_hands

class mymodel(Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,stride=1,padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=3,out_channels=5,kernel_size=5,stride=1,padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=5,out_channels=7,kernel_size=7,stride=1,padding=3)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(in_channels=7,out_channels=14,kernel_size=3,stride=1,padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(14*7*4,1)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x= self.relu1(x)
        # print(x.shape)
        x= self.conv2(x)
        # print(x.shape)
        x= self.relu2(x)
        # print(x.shape)
        x= self.conv3(x)
        # print(x.shape)
        x= self.relu3(x)
        # print(x.shape)
        x= self.conv4(x)
        # print(x.shape)
        x= self.linear1(x.view(-1))
        # print(x.shape)
        return x


device = 'cuda:2'
num_decks = 1
full_deck = []

for num  in range(1,14):
    for i in range(num_decks*4):
        full_deck.append(num)
winrate = 0
results = []
for i in range(20000):
    current_deck = full_deck.copy()
    your_hand, dealer_hand, extra_hands = deal_table(current_deck, 2)
    max_hand_size = 7
    your_total = hand_total(your_hand)
    dealer_total = hand_total(dealer_hand)
    data_image = np.zeros((1,1,max_hand_size,2+len(extra_hands)),dtype=np.float32)

    for idx in range(len(your_hand)):
        data_image[0][0][0][idx] = your_hand[idx]
    for idx in range(len(dealer_hand)):
        data_image[0][0][1][idx] = dealer_hand[idx]
    for hand in extra_hands:
        for idx in range(len(hand)):
            data_image[0][0][1][idx] = hand[idx]
    data_torch = torch.from_numpy(data_image)
    model = mymodel()
    LossFun = nn.L1Loss()

    hand_over = False
    while hand_over != True:
        truth = -1
        net_out = model(data_torch)
        # Stay (not hit) 0
        if net_out < 0.5 or len(your_hand) == max_hand_size:
            hand_over = True
            while (dealer_total < 17) and len(dealer_hand) != max_hand_size :
                dealer_hand.append(deal_card(current_deck))
                dealer_total = hand_total(dealer_hand)
            if (your_total > dealer_total) or (dealer_total > 21):
                truth = torch.tensor([0],dtype=torch.float32)
                # print("You win!")
                if len(results) == 50:
                    results.pop(0)
                results.append(1)
            elif your_total < dealer_total:
                truth = torch.tensor([1],dtype=torch.float32)
                # print("You lost!")
                if len(results) == 50:
                    results.pop(0)
                results.append(0)
            elif your_total == dealer_total:
                print("Tie")
                if len(results) == 50:
                    results.pop(0)
                results.append(0.5)
                continue
            output = LossFun(net_out, truth)
            output.backward()
        # Hit (not stay) 1
        else:
            your_hand.append(deal_card(current_deck))
            new_your_total = hand_total(your_hand)
            if ((new_your_total > your_total) and (new_your_total < 22)):
                truth = torch.tensor([1],dtype=torch.float32)
            else:
                truth = torch.tensor([0],dtype=torch.float32)
            output = LossFun(net_out, truth)
            output.backward()
            # Update Numpy array and then the tensor
            your_total = new_your_total
            if your_total > 21:
                # print("You lost!")
                if len(results) == 50:
                    results.pop(0)
                results.append(0)
                hand_over = True
                continue
            for idx in range(len(your_hand)):
                data_image[0][0][0][idx] = your_hand[idx]
            data_torch = torch.from_numpy(data_image)
    total = 0.
    for result in results:
        total = float(total+result)
    print("Winrate is: ", total/float(len(results)))


# # Play a Round
# action = ''
# while action != 'stay':
#     print()
#     print("Your hand is:", your_hand)
#     your_total = hand_total(your_hand)
#     print("Your total is:", your_total)
#     print("Dealer hand is:", dealer_hand)
#     dealer_total = hand_total(dealer_hand)
#
#     print("Dealer's total is:", dealer_total)
#     result = winner(your_total, dealer_total)
#     if your_total > 21:
#         print()
#         print("You Busted!")
#         print()
#         break
#     if result ==0:
#         print("You're losing")
#     elif result ==1:
#         print("You're winning!")
#     else:
#         print("Tied, Push!")
#     print("Would you like to hit or stay? (hit/stay)")
#     action = input()
#     if action != 'stay':
#         your_hand.append(deal_card(current_deck))
#     else:
#         print("-----------------------------")
#         print("Staying!")
#         while dealer_total < 17:
#             print("Dealer Hits")
#             dealer_hand.append(deal_card(current_deck))
#             print("Your total is:", your_total)
#             print("Dealer hand is:", dealer_hand)
#             dealer_total = hand_total(dealer_hand)
#             print("Dealer's total is:", dealer_total)
#
#         result = winner(your_total, dealer_total)
#         print("------------------------------")
#         if result ==0:
#             print("You Lost!")
#         elif result ==1:
#             print("You Won!")
#         else:
#             print("Tied!")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # EoF
