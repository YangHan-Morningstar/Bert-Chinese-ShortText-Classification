# Bert-Chinese-ShortText-Classification

## 一、简介

基于Bert的中文短文本分类，通过Pytorch深度学习框架实现，采取fine-tuning的策略，在Bert模型（ERNIE模型）后接入CNN、RNN、RCNN、DPCNN等模型（其他仓库中所实现的上述模型大部分多少与论文中的结构有所出入，如RCNN，本仓库所实现的模型除少部分超参数外均与原论文相同），附带数据集为取自THUCNews新闻数据集的65000条新闻数据。

## 二、说明

### 2.1 模型效果

* 当选择Bert作为预训练词向量模型时，直接接全连接层和接更加复杂的深度学习模型在最终测试集f1-score相差不超过2%。后续模型不变，预词向量模型更换为ERNIE后提升4%～5%。

### 2.2 数据介绍

* THUCNews数据集的一个真子集，分为体育、娱乐、家居、房产、教育、时尚、时政、游戏、科技、财经十类。
* 训练集：50000条，验证集：5000条，测试集：10000条。

### 2.3 论文

* Bert：Pre-training of Deep Bidirectional Transformers for Language Understanding
* ERNIE：Enhanced Representation through Knowledge Integration
* TextCNN：Convolutional Neural Networks for Sentence Classification
* TextRCNN：Recurrent Convolutional Neural Networks for Text Classification
* TextDPCNN：Deep Pyramid Convolutional Neural Networks for Text Categorization

### 2.4 使用

* Models文件下存储模型的py文件，每个文件中都有一个Config类，用于初始化数据路径、预训练模型路径等。
* 使用之前需要将数据集、预训练文件放在相应目录下，详情请看cnews、bert_pretrain、ERNIE_pretrain下的readme。
* 直接使用命令`CUDA_VISIBLE_DEVICES=0 python main.py --model="模型所在的py文件名，如bert_base"`即可开始训练。
