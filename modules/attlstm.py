# coding : utf-8
# Author : yuxiang Zeng


import torch

from modules.pred_layer import Predictor

import torch


class AttnLSTMTimeRefiner(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.num_windows = args.num_windows
        self._time_linear = torch.nn.Linear(args.rank, args.rank)
        self.lstm = torch.nn.LSTM(args.rank, args.rank, batch_first=False)
        self.attn = torch.nn.Sequential(torch.nn.Linear(args.rank, 1), torch.nn.Tanh())

    def attention_layer(self, lstm_output):
        attention_scores = self.attn(lstm_output).squeeze(-1)  # [bs, window]
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=1)  # [bs, window]
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * lstm_output, dim=1)  # [bs, hidden_dim]
        return context_vector, attention_weights

    def forward(self, time_embeds):
        outputs, (hs, _) = self.lstm(time_embeds)
        time_embeds, _ = self.attention_layer(outputs)
        return time_embeds


class AttLstm(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, args):
        super().__init__()
        self.args = args
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.trans = torch.nn.Linear(input_size, hidden_size)
        self.time_encoder = AttnLSTMTimeRefiner(args)
        self.predictor = Predictor(self.hidden_size, self.hidden_size, self.args.num_preds)

    def forward(self, x):
        x = x.to(torch.float32)  # 确保输入是float32类型
        x = self.trans(x)
        output = self.time_encoder(x)
        y = self.predictor(output)
        return y

