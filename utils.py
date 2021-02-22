from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import time
from datetime import timedelta
import random
import os
import pickle as pkl


PAD, CLS = "[PAD]", "[CLS]"

label_dict = {'体育': 0, '娱乐': 1, '家居': 2, '房产': 3, '教育': 4, '时尚': 5, '时政': 6, '游戏': 7, '科技': 8, '财经': 9}


def load_dataset(filepath, config):
    """
    :param filepath 路径
    :param config: 配置类
    :return: 四个list
    """
    contents_no_shuffle = []  # 用于存取[ids, label, ids_len, mask]
    contents_shuffle = []
    with open(filepath, "r", encoding="UTF-8") as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            label, content = line.split("\t")  # content为字符串（文本内容）
            token = config.tokenizer.tokenize(content)
            token = [CLS] + token
            seq_len = len(token)

            token_ids = config.tokenizer.convert_tokens_to_ids(token)
            max_len = config.max_len

            if max_len:
                if len(token) < max_len:
                    mask = [1] * len(token_ids) + [0] * (max_len - len(token))
                    token_ids = token_ids + ([0] * (max_len - len(token)))
                else:
                    mask = [1] * max_len
                    token_ids = token_ids[:max_len]
                    seq_len = max_len
            contents_no_shuffle.append((token_ids, label_dict[label], seq_len, mask))

    random_num_array = random.sample(range(0, len(contents_no_shuffle)), len(contents_no_shuffle))
    for i in random_num_array:
        contents_shuffle.append(contents_no_shuffle[i])

    return contents_shuffle


def build_dataset(config):
    """
    :param config:
    :return: train, dev, test
        其中每个元素为四维list[ids, label, ids_len, mask]
    """

    if os.path.exists(config.datasetpkl):
        dataset = pkl.load(open(config.datasetpkl, 'rb'))
        train = dataset['train']
        dev = dataset['dev']
        test = dataset['test']
    else:
        train = load_dataset(config.train_path, config)
        dev = load_dataset(config.dev_path, config)
        test = load_dataset(config.test_path, config)
        dataset = {}
        dataset['train'] = train
        dataset['dev'] = dev
        dataset['test'] = test
        pkl.dump(dataset, open(config.datasetpkl, 'wb'))
    return train, dev, test


class DatasetIterator(object):
    def __init__(self, dataset, batch_size, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.index = 0
        self.n_batches = len(dataset) // batch_size

    def _to_tensor(self, datas):
        ids = torch.LongTensor([item[0] for item in datas]).to(self.device)  # 样本数据ids
        label = torch.LongTensor([item[1] for item in datas]).to(self.device)  # 标签数据label

        seq_len = torch.LongTensor([item[2] for item in datas]).to(self.device)  # 每一个序列的真实长度
        mask = torch.LongTensor([item[3] for item in datas]).to(self.device)

        return (ids, seq_len, mask), label

    def __next__(self):
        if self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.dataset[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches


def build_iterator(dataset, config):
    iterator = DatasetIterator(dataset, config.batch_size, config.device)
    return iterator


def get_time_dif(start_time):
    """
    获取已经使用的时间
    :param start_time:
    :return:
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
