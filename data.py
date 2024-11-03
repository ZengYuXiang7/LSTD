# coding : utf-8
# Author : yuxiang Zeng
import numpy as np
import torch
import pickle

from torch.utils.data import DataLoader

from utils.config import get_config
from utils.logger import Logger
from utils.plotter import MetricsPlotter
from utils.utils import set_settings


import dgl
from scipy.sparse import csr_matrix


class experiment:
    def __init__(self, args):
        self.args = args

    @staticmethod
    def load_data(args):
        # 异常机制。访问本地文件，如果没有不会报错终止程序，而是避开错误执行报错情况的程序
        # 而我们利用这个机制，读取本地文件，没有本地文件就报错，然后执行文件存储，制作数据集
        try:
            with open('./datasets/rtdata.pkl', 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            string = './datasets/' + 'rtdata' + '.txt'
            rtdata = np.loadtxt(open(string, 'rb')).astype(np.float64)
            index_max = np.max(rtdata, axis=0)[:-1].astype(np.int32) + 1
            data = np.zeros(index_max)
            for i, j, k, value in rtdata:
                data[int(i), int(j), int(k)] = value
            data = np.transpose(data, (2, 0, 1))
            with open('./datasets/rtdata.pkl', 'wb') as f:
                pickle.dump(data, f)

        return data

    @staticmethod
    def preprocess_data(data, args):
        data[data == -1] = 0
        data[data > np.percentile(data, q=99)] = 0
        timeIdx, userIdx, servIdx = data.nonzero()
        values = data[timeIdx, userIdx, servIdx]
        index = np.stack([timeIdx, userIdx, servIdx], axis = 1)
        return index, values


# 数据集定义
class DataModule:
    def __init__(self, exper_type, args):
        self.args = args
        self.path = args.path
        self.raw_data = exper_type.load_data(args)
        self.x, self.y = exper_type.preprocess_data(self.raw_data, args)
        self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y, self.max_value = self.get_train_valid_test_dataset(self.x, self.y, args)
        self.train_set, self.valid_set, self.test_set = self.get_dataset(self.train_x, self.train_y, self.valid_x,self.valid_y, self.test_x, self.test_y, args)
        self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(self.train_set, self.valid_set,self.test_set, args)
        args.log.only_print(f'Train_length : {len(self.train_loader.dataset)} Valid_length : {len(self.valid_loader.dataset)} Test_length : {len(self.test_loader.dataset)} Max_value : {self.max_value:.2f}')

    def get_dataset(self, train_x, train_y, valid_x, valid_y, test_x, test_y, args):
        return (
            TensorDataset(train_x, train_y, args),
            TensorDataset(valid_x, valid_y, args),
            TensorDataset(test_x, test_y, args)
        )

    def get_train_valid_test_dataset(self, x, y, args):
        x, y = np.array(x), np.array(y)
        indices = np.random.permutation(len(x))
        x, y = x[indices], y[indices]

        # print(y.shape)

        if not args.classification:
            max_value = y.max()
            y /= max_value
        else:
            max_value = 1

        train_size = int(len(x) * args.density)
        # train_size = int(args.train_size)
        valid_size = int(len(x) * 0.05)

        train_x = x[:train_size]
        train_y = y[:train_size]

        valid_x = x[train_size:train_size + valid_size]
        valid_y = y[train_size:train_size + valid_size]

        test_x = x[train_size + valid_size:]
        test_y = y[train_size + valid_size:]

        return train_x, train_y, valid_x, valid_y, test_x, test_y, max_value

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, args):
        self.args = args
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        userIdx, servIdx, timeIdx = self.x[idx]
        value = self.y[idx]
        return userIdx, servIdx, timeIdx, value


def custom_collate_fn(batch, args):
    from torch.utils.data.dataloader import default_collate
    timeIdx, userIdx, servIdx, values = zip(*batch)
    timeIdx = torch.as_tensor(timeIdx)
    userIdx = torch.as_tensor(userIdx)
    servIdx = torch.as_tensor(servIdx)
    values = torch.as_tensor(values)
    return timeIdx, userIdx, servIdx, values


def get_dataloaders(train_set, valid_set, test_set, args):
    import multiprocessing
    max_workers = multiprocessing.cpu_count() // 4
    train_loader = DataLoader(
        train_set,
        batch_size=args.bs,
        drop_last=False,
        shuffle=True,
        pin_memory=True,
        collate_fn=lambda batch: custom_collate_fn(batch, args),
        num_workers=max_workers,
        prefetch_factor=4
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.bs,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda batch: custom_collate_fn(batch, args),
        num_workers=max_workers,
        prefetch_factor=4
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.bs,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda batch: custom_collate_fn(batch, args),
        num_workers=max_workers,
        prefetch_factor=4
    )

    return train_loader, valid_loader, test_loader




if __name__ == '__main__':

    args = get_config()
    set_settings(args)
    args.experiment = True

    # logger plotter
    exper_detail = f"Dataset : {args.dataset.upper()}, Model : {args.model}, Train_size : {args.train_size}"
    log_filename = f'{args.train_size}_r{args.rank}'
    log = Logger(log_filename, exper_detail, args)
    plotter = MetricsPlotter(log_filename, args)
    args.log = log
    log(str(args.__dict__))


    exper = experiment(args)
    datamodule = DataModule(exper, args)
    for train_batch in datamodule.train_loader:
        num_windowss, value = train_batch
        print(num_windowss.shape, value.shape)
        # break
    print('Done!')
