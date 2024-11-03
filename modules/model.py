import torch
from torch.nn import *
# from utils.metrics import ErrorMetrics
from dgl.nn.pytorch import EdgeGATConv


def to_device(args, device):
    return [arg.to(device) for arg in args]


class Attention(Module):

    def __init__(self, args):
        super().__init__()
        self.linear = Linear(args.rank, args.rank, bias=False)
        self.fcn = Sequential(
            Linear(4 * args.rank, args.rank),
            ReLU(),
            Linear(args.rank, 1)
        )
        self.softmax = Softmax(dim=-1)

    def forward(self, query, embeds):
        query = query.unsqueeze(1).repeat(1, embeds.shape[1], 1)
        embeds_inputs = self.linear(embeds)
        attn_inputs = torch.cat([
            query, embeds_inputs,
            query - embeds_inputs,
            query * embeds_inputs], dim=-1)
        attn_score = self.fcn(attn_inputs).squeeze()
        attn_score = self.softmax(attn_score).unsqueeze(-1)
        attn_embeds = torch.sum(attn_score * embeds, dim=1)
        return attn_embeds


class AttentiveLSTM(Module):

    def __init__(self, args):
        super().__init__()
        self.lstm = LSTM(args.rank, args.rank, batch_first=False)
        self.attn = Attention(args)

    def forward(self, query, embeds):
        outputs, _ = self.lstm.forward(embeds)
        attn_embeds = self.attn.forward(query, outputs)
        return attn_embeds


class BPRLoss(Module):
    def __init__(self):
        super().__init__()
        self.softplus = Softplus()

    def forward(self, anchor, pos, neg):
        pos_score = torch.sum(anchor * pos, dim=-1)
        neg_score = torch.sum(anchor * neg, dim=-1)
        loss = self.softplus(neg_score - pos_score)
        loss = torch.mean(loss)
        return loss


class LSInterestEncoder(Module):

    def __init__(self, args):
        super().__init__()

        # Basic Params
        self.args = args
        self.device = args.device
        self.L_windows = args.L_windows
        self.S_windows = args.S_windows
        self.L_rainbow = torch.arange(-self.L_windows + 1, 1).reshape(1, -1).to(self.device)
        self.S_rainbow = torch.arange(-self.S_windows + 1, 1).reshape(1, -1).to(self.device)

        # Time Embeddingss
        self.time_embeds = Embedding(args.num_times + 1, args.rank, padding_idx=0)

        # Embedding Transformation
        self.time_fusion = Linear(3 * args.rank, args.rank, bias=True)

        # Long Short Term Interest Encoder
        self.L_interest_encoder = Attention(args)
        self.S_interest_encoder = AttentiveLSTM(args)

        # Self-Supervised Loss
        self.loss = BPRLoss()

        # LSTM Pooling
        self.lstm = LSTM(args.rank, args.rank, batch_first=True)

    def contrastive_loss(self, L_proxy, S_proxy, L_interest, S_interest):
        term1 = self.loss.forward(L_interest, L_proxy, S_proxy)
        term2 = self.loss.forward(S_interest, S_proxy, L_proxy)
        term3 = self.loss.forward(L_proxy, L_interest, S_interest)
        term4 = self.loss.forward(S_proxy, S_interest, L_interest)
        loss = term1 + term2 + term3 + term4
        return loss


    def to_seq_id(self, tids, wsize):
        tids = tids.reshape(-1, 1).repeat(1, wsize)
        if wsize == self.L_windows:
            tids += self.L_rainbow
        else:
            tids += self.S_rainbow
        return tids.relu()


    def forward(self, user_embeds, item_embeds, time):
        # 把单个时间步的Time，转化为长短期Time序列
        L_time = self.to_seq_id(time, self.L_windows)
        S_time = self.to_seq_id(time, self.S_windows)

        # 读取长短期Time Embeddings
        L_time_embeds = self.time_embeds(L_time)
        S_time_embeds = self.time_embeds(S_time)

        # Embedding Fusion
        repeated_L_user_embeds = user_embeds.unsqueeze(1).repeat(1, self.L_windows, 1)
        repeated_S_user_embeds = user_embeds.unsqueeze(1).repeat(1, self.S_windows, 1)
        repeated_L_item_embeds = item_embeds.unsqueeze(1).repeat(1, self.L_windows, 1)
        repeated_S_item_embeds = item_embeds.unsqueeze(1).repeat(1, self.S_windows, 1)

        L_time_embeds = self.time_fusion(torch.cat([repeated_L_user_embeds, repeated_L_item_embeds, L_time_embeds], dim=-1))
        S_time_embeds = self.time_fusion(torch.cat([repeated_S_user_embeds, repeated_S_item_embeds, S_time_embeds], dim=-1))

        # Proxy Embeddings
        L_proxy = torch.mean(L_time_embeds, dim=1)
        S_proxy = torch.mean(S_time_embeds, dim=1)

        # 计算 Long Short Term Interest
        L_interest = self.L_interest_encoder.forward(user_embeds * item_embeds, L_time_embeds)
        S_interest = self.S_interest_encoder.forward(user_embeds * item_embeds, S_time_embeds)

        # 计算 Self-Supervised Loss
        loss = self.contrastive_loss(L_proxy, S_proxy, L_interest, S_interest)

        # LSTM on Long Time Embeddings
        _, (lstm_pooling, _) = self.lstm.forward(L_time_embeds)
        return L_interest, S_interest, lstm_pooling.squeeze(), loss


class LSTC(Module):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device = args.device
        ## 宝宝改
        self.user_embeds = Embedding(args.num_users, args.rank)
        self.item_embeds = Embedding(args.num_servs, args.rank)
        self.interest_encoder = LSInterestEncoder(args)

        self.alpha_net = Sequential(
            Linear(3 * args.rank, args.rank),
            LayerNorm(args.rank),
            ReLU(),
            Linear(args.rank, 1)
        )

        self.pred_layer = Sequential(
            Linear(3 * args.rank, 2 * args.rank),
            LayerNorm(2 * args.rank),
            ReLU(),
            Linear(2 * args.rank, 1),
            Sigmoid()
        )


    def forward(self, timeidx, useridx, servidx, contrastive=False):
        user_embeds = self.user_embeds(useridx)
        serv_embeds = self.item_embeds(servidx)
        L_interest, S_interest, lstm_pooling, loss = self.interest_encoder.forward(user_embeds, serv_embeds, timeidx)
        # 计算 Time Embeddings
        alpha = self.alpha_net.forward(torch.cat([L_interest, S_interest, lstm_pooling], dim=-1))
        time_embeds = alpha * L_interest + (1 - alpha) * S_interest
        # 计算最终得分
        score_inputs = torch.cat([user_embeds, serv_embeds, time_embeds], dim=-1)
        score = self.pred_layer(score_inputs).squeeze()

        # 0.001 * con_loss
        if contrastive:
            return score, loss
        else:
            return score
