import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):
    """
    配置参数
    """
    def __init__(self, dataset):
        self.model_name = "ERNIE-Base"
        # 训练集
        self.train_path = dataset + '/cnews.train.txt'
        # 校验集
        self.dev_path = dataset + '/cnews.dev.txt'
        # 测试集
        self.test_path = dataset + '/cnews.test.txt'
        # dataset
        self.datasetpkl = dataset + '/dataset.pkl'
        # 类别名单
        self.class_list = [x.strip() for x in open(dataset + '/class.txt').readlines()]

        # 模型保存路径
        self.save_path = "./" + self.model_name + ".ckpt"
        # 训练设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 超过1000哥batch效果没有提升则停止训练
        self.require_improvement = 1000
        # 标签种类数目
        self.num_classes = len(self.class_list)
        # epoch数目
        self.num_epochs = 6
        # batch_size
        self.batch_size = 8
        # 每条数据最大长度
        self.max_len = 512
        # 学习率
        self.learning_rate = 1e-5
        # bert预训练模型位置
        self.ernie_path = "ERNIE_pretrain"
        # bert切词器
        self.tokenizer = BertTokenizer.from_pretrained(self.ernie_path)
        # 隐藏层输出特征维度
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.ernie_path)
        for param in self.bert.parameters():
            param.requires_grad = True  # True为允许fine-tuning， False为禁止
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        """
        x: 三维数据[ids, seq_len, mask]
            ids: 每条数据中每个字对应的编号
            seq_len: 句子长度
            mask: 用于掩盖padding的0
        """
        content = x[0]  # shape=[128, 32]
        mask = x[2]  # shape=[128, 32]
        _, pooled = self.bert(content, attention_mask=mask, output_all_encoded_layers=False)  # pooled shape=[128, 768]
        out = self.fc(pooled)  # shape=[128, 10]

        return out
