import numpy as np
import torch
import torch.nn as nn

class HUNGREY(torch.nn.Module):

    def __init__(self,args):
        super(HUNGREY,self).__init__()
        self.args = args
        self.userEmbedding = torch.nn.Embedding(args.num_users,args.rank)
        self.servEmbedding = torch.nn.Embedding(args.num_servs,args.rank)
        self.timeEmbedding = torch.nn.Embedding(args.num_times,args.rank)

    def forward(self,timeIdx, userIdx, servIdx):
        userEmbeds = self.userEmbedding(userIdx)
        servEmbeds = self.servEmbedding(servIdx)
        timeEmbeds = self.timeEmbedding(timeIdx)
        y = torch.sum(userEmbeds*servEmbeds*timeEmbeds, dim=-1)
        y = torch.sigmoid(y)
        return y


