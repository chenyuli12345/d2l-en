DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

nn_Module = nn.Module

#################   WARNING   ################
# The below part is generated automatically through:
#    d2lbook build lib
# Don't edit it directly

import collections
import hashlib
import inspect
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
import pandas as pd
import requests
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline

d2l = sys.modules[__name__]

import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.spatial import distance_matrix
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.

    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points.

    Defined in :numref:`sec_calculus`"""

    def has_one_axis(X):  # True if X (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    set_figsize(figsize)
    if axes is None:
        axes = d2l.plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

def add_to_class(Class):
    """Register functions as methods in created class.

    Defined in :numref:`sec_oo-design`"""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper

class HyperParameters:
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        """Defined in :numref:`sec_oo-design`"""
        raise NotImplemented

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.
    
        Defined in :numref:`sec_utils`"""
        #inspect模块用于检查和获取有关活动对象的信息，例如模块、类、方法、函数、帧对象和代码对象。可用于调试、测试和理解大门
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)

class ProgressBoard(d2l.HyperParameters):
    """The board that plots data points in animation.

    Defined in :numref:`sec_oo-design`"""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented

    def draw(self, x, y, label, every_n=1):
        """Defined in :numref:`sec_utils`"""
        Point = collections.namedtuple('Point', ['x', 'y'])
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]),
                          mean([p.y for p in points])))
        points.clear()
        if not self.display:
            return
        d2l.use_svg_display()
        if self.fig is None:
            self.fig = d2l.plt.figure(figsize=self.figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(d2l.plt.plot([p.x for p in v], [p.y for p in v],
                                          linestyle=ls, color=color)[0])
            labels.append(k)
        axes = self.axes if self.axes else d2l.plt.gca()
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        if not self.xlabel: self.xlabel = self.x
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        display.display(self.fig)
        display.clear_output(wait=True)

class Module(d2l.nn_Module, d2l.HyperParameters):
    """模型的基类 Defined in :numref:`sec_oo-design`"""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1): #模型的构造函数，接受两个参数，分别代表训练情况下每多少个epoch绘制动图以及测试状态下每多少个epoch绘制动图
        super().__init__()
        self.save_hyperparameters() #调用父类的方法，保存传入的超参数（构造函数的几个传入参数被保存为self.plot_train_per_epoch,self.plot_valid_per_epoch）
        self.board = ProgressBoard() #创建一个ProgressBoard实例board，用于绘制图形

    #模型的损失函数
    def loss(self, y_hat, y): #定义损失函数，一个占位方法，继承类需要实现这个方法。接受参数为预测值和真实值
        raise NotImplementedError

    #模型的前向传播方法
    def forward(self, X): #定义前向传播方法
        assert hasattr(self, 'net'), 'Neural network is defined'
        #断言self对象有net属性，否则抛出异常
        return self.net(X)
    #绘制图像的方法
    def plot(self, key, value, train): #接受三个参数
        """在动图中绘制一个点"""
        assert hasattr(self, 'trainer'), 'Trainer is not inited' #断言self对象有trainer属性，否则抛出异常
        self.board.xlabel = 'epoch' #设置动图横轴标签为epoch
        if train: #如果是训练
            x = self.trainer.train_batch_idx / self.trainer.num_train_batches #x为当前训练的批次号/训练的总批次数（即数据集大小/小批量大小）（这两个参数均来自于d2l.Trainer中）
            n = self.trainer.num_train_batches / self.plot_train_per_epoch #n为训练的总批次数/训练情况下每多少个epoch绘制动图（来自输入参数）
        else: #如果是验证集
            x = self.trainer.epoch + 1 #x为epoch数+1
            n = self.trainer.num_val_batches / self.plot_valid_per_epoch #n为n为验证的总批次数（来自于d2l.Trainer中）/验证情况下每多少个epoch绘制动图（来自输入参数）
        #更新动图，传入四个参数
        self.board.draw(x, d2l.numpy(d2l.to(value, d2l.cpu())), #x为横坐标，value为纵坐标（将value转换到cpu上，然后转换为numpy数组）
                        ('train_' if train else 'val_') + key, #标签，根据train参数来决定标签前缀，然后前缀与key（也是一个标签）连接生成一个完整的字符串标签
                        every_n=int(n)) #每n个epoch绘制一个点

    #用于打印并获取训练损失的方法
    def training_step(self, batch): #接受一个参数batch，表示数据批次，其中批次中真实值（标签）为最后一个元素，除了最后一个元素为输入特征
        l = self.loss(self(*batch[:-1]), batch[-1]) #调用此类中的另一个方法loss，获取训练集损失（这里的self()相当于利用所有数据进行前向传播的预测值，最后一个元素相当于真实值）
        self.plot('loss', l, train=True) #调用上面的plot方法，传入三个参数，key为'loss'，value为损失，train为True
        return l #返回损失

    #用于打印验证损失的方法
    def validation_step(self, batch): #接受一个参数batch，表示数据批次，其中批次中真实值（标签）为最后一个元素，除了最后一个元素为输入特征
        l = self.loss(self(*batch[:-1]), batch[-1]) #调用此类中的另一个方法loss，获取验证集的损失（这里的self()相当于利用所有数据进行前向传播的预测值，最后一个元素相当于真实值）
        self.plot('loss', l, train=False) #调用上面的plot方法，传入三个参数，key为'loss'，value为损失，train为True
    
    #初始configure_optimizers方法，返回用于更新可学习参数的优化方法或其列表（后续会被覆盖，需要在继承类实现这个方法）
    def configure_optimizers(self):
        raise NotImplementedError
    
    #重新定义的configure_optimizers方法，返回用于更新可学习参数的优化方法或其列表（最终使用的configure_optimizers方法）
    def configure_optimizers(self):
        """Defined in :numref:`sec_classification`"""
        return torch.optim.SGD(self.parameters(), lr=self.lr) #返回一个优化器，使用SGD算法，参数为模型的参数和学习率（应该来源于继承d2l.Module的模型）
    
    #该方法在前向传播后用自定义的初始化函数来初始化模型的参数
    def apply_init(self, inputs, init=None): #接受两个参数，分别为输入和init（一般是一个初始化方法）
        """Defined in :numref:`sec_lazy_init`"""
        self.forward(*inputs) #将inputs解包作为参数传递给网络进行前向传播
        if init is not None: #若传入的init不为None
            self.net.apply(init) #对模型的每个子模块应用init方法（来源于传入参数）

class DataModule(d2l.HyperParameters):
    """数据的基类，Defined in :numref:`subsec_oo-design-models`"""
    def __init__(self, root='../data', num_workers=4): #构造函数，接受六个参数，root默认值为'../data'，num_workers默认值为4
        self.save_hyperparameters() #调用父类的方法，保存传入的超参数（构造函数的几个传入参数被保存为self.root,self.num_workers）
    
    #一个占位方法，继承类需要实现这个方法，用于获取数据集的DataLoader对象
    def get_dataloader(self, train): #接受参数train
        raise NotImplementedError
    
    #获取训练数据，具体方法要根据get_dataloader方法来实现
    def train_dataloader(self):
        return self.get_dataloader(train=True)
    
    #获取验证数据，具体方法要根据get_dataloader方法来实现
    def val_dataloader(self):
        return self.get_dataloader(train=False)
    
    #调用内部函数，将数据集tensors在用indices切片后封装成一个DataLoader对象，可以像普通的Python迭代器一样进行迭代
    def get_tensorloader(self, tensors, train, indices=slice(0, None)): #接受参数tensors（包含多个张量的可迭代对象）,train,indices（默认为slice(0,None)，表示一个切片对象，可以用来对序列（如列表、元组、字符串等）进行切片操作，None表示表示一直切到序列的末尾）
        """Defined in :numref:`sec_synthetic-regression-data`"""
        tensors = tuple(a[indices] for a in tensors) #对tensors中的每个张量进行切片操作，截取indices对应的部分
        dataset = torch.utils.data.TensorDataset(*tensors) #创建一个数据集对象，将切片后的张量包装成一个数据集，每个样本都是一个元组
        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train) #将数据集对象dataset封装成可迭代对象的类，支持按批次加载数据、打乱数据等功能，并指定每个批次的样本数量以及是否打乱数据
        #最后返回一个DataLoader对象，可以像普通的Python迭代器一样进行迭代

class Trainer(d2l.HyperParameters):
    """使用数据训练模型的基类，Defined in :numref:`subsec_oo-design-models`"""
    #初始构造函数（不使用GPU），后续会重写
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0): #构造函数，接受三个参数，max_epochs表示最大训练轮数，num_gpus表示gpu数量（默认0不使用）和梯度裁剪值（默认0）
        self.save_hyperparameters() #调用父类的方法，保存传入的超参数（构造函数的几个传入参数被保存为self.max_epochs,self.num_gpus,self.gradient_clip_val）
        assert num_gpus == 0, 'No GPU support yet' #断言使用gpu数量为0（来源于输入参数，即默认不使用）

    #用于准备数据集（来源于d2l.DataModule）
    def prepare_data(self, data): #接受参数data
        self.train_dataloader = data.train_dataloader() #获取训练数据（来源于d2l.DataModule）
        self.val_dataloader = data.val_dataloader() #获取验证数据（来源于d2l.DataModule）
        self.num_train_batches = len(self.train_dataloader) #获取训练集的批次数（即训练集数据量/小批量大小，多少个批次）
        self.num_val_batches = (len(self.val_dataloader) if self.val_dataloader is not None else 0) #获取验证集的批次数（即验证集数据量/小批量大小，多少个批次），不过这里要先判断是否存在验证集
    
    #初始prepare_model方法（不使用gpu，后续会重写），作用是将模型（传入参数model，来源于d2l.Module）与Trainer实例关联起来
    def prepare_model(self, model): #传入参数为模型(来源于d2l.Module)
        model.trainer = self #将当前的Trainer实例（self参数总是指向调用这个方法的实例，因此self是一个Trainer实例）赋值给模型(来源于d2l.Module)的trainer属性（使模型可以访问Trainer实例的属性和方法）
        model.board.xlim = [0, self.max_epochs]  #获取模型(来源于d2l.Module)的动态绘图实例board，将其横轴范围设置为0到最大的epoch（来源于传入的参数）
        self.model = model #将传入的model(来源于d2l.Module)赋值给Trainer实例的model属性，这使得Trainer实例可以访问和操作模型

    #遍历整个数据集max_epochs次来训练模型
    def fit(self, model, data): #接受参数为模型(来源于d2l.Module)和数据集（来源于d2l.DataModule）
        self.prepare_data(data) #调用自己类的prepare_data方法获取数据集
        self.prepare_model(model) #调用自己类的prepare_model方法获取模型（并与Trainer实例关联）
        self.optim = model.configure_optimizers() #调用模型的configure_optimizers方法（来源于d2l.Module）获取优化器
        self.epoch = 0 #初始化epoch
        self.train_batch_idx = 0 #初始化当前训练的批次（第几个，即索引）
        self.val_batch_idx = 0 #初始化当前验证的批次（第几个，即索引）
        #执行下面的fit_epoch方法max_epochs次（来源于类的输入参数）
        for self.epoch in range(self.max_epochs): 
            self.fit_epoch()

    #初始fit_epoch方法（后续会重写），一个占位方法，继承类需要实现这个方法，用于训练模型（一个epoch）
    def fit_epoch(self):
        raise NotImplementedError

    #初始prepare_batch方法（不使用gpu，后续会重写），直接返回输入的数据小批次（相当于不使用GPU），这提供了一个钩子，在之后重新定义
    def prepare_batch(self, batch): #接受参数batch
        """Defined in :numref:`sec_linear_scratch`"""
        return batch #直接返回输入的batch

    #重新定义的fit_epoch方法，用于训练模型一次
    def fit_epoch(self):
        """Defined in :numref:`sec_linear_scratch`"""
        self.model.train() #将模型(来源于d2l.Module)设置为训练模式
        for batch in self.train_dataloader: #对于训练数据（来源于d2l.DataModule）的每一个批次
            loss = self.model.training_step(self.prepare_batch(batch)) #调用模型的training_step方法（来源于d2l.Module）打印并获取训练集的损失
            self.optim.zero_grad() #将优化器（来源于fit）的梯度清空
            with torch.no_grad(): #不计算梯度？？？
                loss.backward() #反向传播loss
                if self.gradient_clip_val > 0:  # To be discussed later，如果梯度裁剪值大于0
                    self.clip_gradients(self.gradient_clip_val, self.model) #调用下面的clip_gradients方法，传入梯度裁剪值（来源于输入）模型(来源于d2l.Module)
                self.optim.step() #更新模型参数
            self.train_batch_idx += 1 #在一个小批次的训练结束后更新当前批次的索引
        if self.val_dataloader is None: #如果没有验证数据（来源于d2l.DataModule）
            return #直接返回
        #如果有验证数据（来源于d2l.DataModule）则继续进行下一步
        self.model.eval() #将模型(来源于d2l.Module)设置为评估模式
        for batch in self.val_dataloader: #对于验证数据（来源于d2l.DataModule）的每一个批次
            with torch.no_grad(): #不计算梯度
                self.model.validation_step(self.prepare_batch(batch)) #调用模型的validation_step方法（来源于d2l.Module）打印验证集的损失
            self.val_batch_idx += 1 #在一个小批次

    #重新定义的构造函数，多定义了可用GPU列表
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0): #接受三个参数，max_epochs表示最大训练轮数，num_gpus表示gpu数量（默认0不使用）和梯度裁剪值（默认0）
        """Defined in :numref:`sec_use_gpu`"""
        self.save_hyperparameters() #调用父类的方法，保存传入的超参数（构造函数的几个传入参数被保存为self.max_epochs,self.num_gpus,self.gradient_clip_val）
        self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))] #初始化一个包含可用GPU设备的列表，取系统可用gpu数量（d2l.num_gpus()）和自定义使用gpu数量（num_gpus，来源于输入参数）的较小值，然后创建GPU列表（d2l.gpu(i)获取第i个GPU设备）
    
    #重新定义的prepare_batch方法，将数据批次传递到GPU上
    def prepare_batch(self, batch):
        """Defined in :numref:`sec_use_gpu`"""
        if self.gpus: #如果有可用GPU（来自于重新定义的构造函数）
            batch = [d2l.to(a, self.gpus[0]) for a in batch] #将数据批次batch中的每个元素转移到可用的第一个gpu上
        return batch #返回移动到第一块GPU上的数据批次
    
    #重新定义的prepare_model方法，作用是将模型（传入参数model，来源于d2l.Module）与Trainer实例关联起来，多了一步把模型转移到GPU上
    def prepare_model(self, model):
        """Defined in :numref:`sec_use_gpu`"""
        model.trainer = self #将当前的Trainer实例（self参数总是指向调用这个方法的实例，因此self是一个Trainer实例）赋值给模型(来源于d2l.Module)的trainer属性（使模型可以访问Trainer实例的属性和方法）
        model.board.xlim = [0, self.max_epochs] #获取模型(来源于d2l.Module)的动态绘图实例board，将其横轴范围设置为0到最大的epoch（来源于传入的参数）
        if self.gpus: #如果GPU可用（来自于重新定义的构造函数）
            model.to(self.gpus[0]) #将模型转移到可用的第一个GPU上
        self.model = model #将传入的model(来源于d2l.Module)赋值给Trainer实例的model属性，这使得Trainer实例可以访问和操作模型

    #用于对模型的梯度进行裁剪，以防止梯度爆炸
    def clip_gradients(self, grad_clip_val, model): #接受两个参数，梯度裁剪值（来源于输入参数）和模型(来源于d2l.Module)
        """Defined in :numref:`sec_rnn-scratch`"""
        params = [p for p in model.parameters() if p.requires_grad] #获取模型参数中所有需要计算梯度的参数
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params)) #计算筛选后参数的梯度的平方和，然后开方，即计算筛选后参数的L2范数
        if norm > grad_clip_val: #如果参数L2范数大于梯度裁剪值
            for param in params: #对于筛选后的每个参数
                param.grad[:] *= grad_clip_val / norm #将每个参数的梯度按比例缩小（不超过梯度裁剪值）

class SyntheticRegressionData(d2l.DataModule):
    """生成线性回归的合成数据
    Defined in :numref:`sec_synthetic-regression-data`"""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, batch_size=32): #构造函数，接受参数权重向量w、偏置b、噪声noise（默认0.01）、训练样本数量num_train（默认1000）、验证样本数量num_val（默认1000）和批量大小batch_size（默认32）
        super().__init__() #调用父类的构造函数
        self.save_hyperparameters() #调用父类的方法，保存传入的超参数（构造函数的几个传入参数被保存为self.w,self.b,self.noise等等）
        n = num_train + num_val #n为训练样本数量和验证样本数量之和
        self.X = d2l.randn(n, len(w)) #创建特征X，是一个形状为n*len(w)的张量，每个元素都是从均值为0，标准差为1的正态分布中随机采样得到
        noise = d2l.randn(n, 1) * noise #生成形状为n*1的噪声张量，每个元素都是从均值为0，标准差为1的正态分布中随机采样得到，然后乘上噪声系数noise=0.01
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + noise #生成标签y，是特征X和权重w的矩阵乘法加上偏置b和噪声张量
    #定义一个获取数据加载器的方法（重写d2l.DataModule中的get_dataloader方法），接受一个参数train，表示是否是训练集（这个方法调用的是父类d2l.DataModule的基于深度学习内置API的Data Loader）
    def get_dataloader(self, train):
        """Defined in :numref:`sec_synthetic-regression-data`"""
        i = slice(0, self.num_train) if train else slice(self.num_train, None) #如果是训练集，i为从0到num_train（训练集数量）的切片，否则为从num_train（训练集数量）到最后（其实就是训练集数量+测试集数量）的切片
        return self.get_tensorloader((self.X, self.y), train, i) #调用父类d2l.DataModule的get_tensorloader方法，传入特征X和标签y，训练集标志train和切片i
        #最后返回一个包含所有训练集和测试集的DataLoader对象（小批量为self.batch_size），可以像普通的Python迭代器一样进行迭代

class LinearRegressionScratch(d2l.Module):  #@save，继承自d2l.Module类
    """从零开始实现线性回归模型的类"""
    def __init__(self, num_inputs, lr, sigma=0.01): #构造函数，接受输入特征数num_inputs、学习率lr和标准差sigma（默认0.01）
        super().__init__()
        self.save_hyperparameters() #调用父类的方法，保存传入的超参数（构造函数的几个传入参数被保存为self.num_inputs,self.lr,self.sigma）
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True) #初始化权重w，是一个形状为num_inputs*1的张量，每个元素都是从均值为0，标准差为sigma的正态分布中随机采样得到，需要计算梯度
        self.b = torch.zeros(1, requires_grad=True) #初始化偏置b，是一个形状为1的张量，每个元素都是0，需要计算梯度

    def forward(self, X): #定义前向传播方法，接受输入特征X
        return torch.matmul(X, self.w) + self.b #返回Xw+b（无隐藏层，只有输入层和输出层）

    def loss(self, y_hat, y): #定义损失函数，接受预测值y_hat和真实值y
        l = (y_hat - y) ** 2 / 2 #计算平方损失
        return l.mean() #返回平均损失的均值（一个minibatch的几个值的平均）

    def configure_optimizers(self): #定义配置优化器的方法，这里是重写了d2l.Module中的configure_optimizers方法
        return SGD([self.w, self.b], self.lr) #返回一个SGD优化器实例，传入要优化的参数（权重w和偏置b）和学习率lr    

class SGD(d2l.HyperParameters):  #@save，继承自d2l.HyperParameters类
    """小批量随机梯度下降算法"""
    def __init__(self, params, lr): #构造函数，接受参数params和学习率lr
        self.save_hyperparameters() #调用父类的方法，保存传入的超参数（构造函数的几个传入参数被保存为self.params和self.lr）
    #利用SGD更新参数的方法
    def step(self):
        for param in self.params: ##对每一个要优化的参数（来源于输入）执行操作
            param -= self.lr * param.grad #学习率*参数梯度
    #实现优化算法的梯度清零
    def zero_grad(self): 
        for param in self.params: #对每一个要优化的参数（来源于输入）执行操作
            if param.grad is not None: #如果参数的梯度不为空
                param.grad.zero_() #设置梯度为0（避免下一次计算梯度时和上一次相关）

class LinearRegression(d2l.Module):  #@save，继承自d2l.Module类
    """利用高级API实现的线性回归模型"""
    def __init__(self, lr): #构造函数，接受参数学习率lr
        super().__init__()
        self.save_hyperparameters() #调用父类的方法，保存传入的超参数（构造函数的几个传入参数被保存为self.lr）
        self.net = nn.LazyLinear(1) #创建一个全连接层，输出特指数为1，输入特征数会在第一次前向传播时自动确定
        self.net.weight.data.normal_(0, 0.01) #将网络的权重初始化为均值为0，标准差为0.01的正态分布 
        self.net.bias.data.fill_(0) #将网络的偏置初始化为0

    def forward(self, X): #定义前向传播方法，接受输入特征X
        return self.net(X)

    def loss(self, y_hat, y): #定义损失函数，接受预测值y_hat和真实值y
        fn = nn.MSELoss() #创建均方误差损失函数实例
        return fn(y_hat, y) #返回预测值y_hat和真实值y的均方误差

    def configure_optimizers(self): #定义配置优化器的方法
        return torch.optim.SGD(self.parameters(), self.lr) #返回一个SGD优化器实例，传入要优化的参数（模型的参数）和学习率lr

    def get_w_b(self): #定义获取网络权重和偏置的方法
        return (self.net.weight.data, self.net.bias.data) #返回网络的权重和偏置 

class FashionMNIST(d2l.DataModule): #继承自d2l.DataModule类
    """Fashion-MNIST数据集 Defined in :numref:`sec_fashion_mnist`"""
    def __init__(self, batch_size=64, resize=(28, 28)): #构造函数，接受批量大小和调整图像大小作为参数，得到self.train和self.val，分别为训练集和验证集，长度为60000和10000，其每个数据是一个元组，元组第一个元素为图像数据（形状为[通道，高度，宽度]），第二个元素为标签
        super().__init__() 
        self.save_hyperparameters() #调用父类的方法，保存传入的超参数（构造函数的几个传入参数被保存为self.batch_size和self.resize）
        #创建一个包含ToTensor转换和Resize转换两个转换操作的列表，然后将列表中的多个转换操作组合成一个Compose实例trans，按顺序依次应用。
        #ToTensor转换将图像从PIL图像或numpy数组转换为PyTorch张量，并将像素值缩放到[0, 1]范围内。Resize转换则用于将图像的高度和宽度调整为给定的resize尺寸
        trans = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])

        #加载训练集，目录为root（来源于d2l.DataModule），如果没有则下载，同时对每个样本进行上面的trans转换
        self.train = torchvision.datasets.FashionMNIST(root=self.root, train=True, transform=trans, download=True)
        #加载测试集（验证集），目录为root（来源于d2l.DataModule），如果没有则下载，同时对每个样本进行上面的trans
        self.val = torchvision.datasets.FashionMNIST(root=self.root, train=False, transform=trans, download=True)

    def text_labels(self, indices): #将数字标签转换为文本标签的方法，接受一个索引列表indices作为参数
        """返回FashionMNIST文本标签 Defined in :numref:`sec_fashion_mnist`"""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'] #定义标签列表
        return [labels[int(i)] for i in indices] #将标签转换为对应的文本标签，首先遍历labels中的每一个标签，如何将标签i转换为整数，从自定义的标签文本列表中获取对应的文本标签
        #相当于根据数据集的标签i，返回对应的文本标签（排序好的）的list 

    def get_dataloader(self, train): #重写d2l.DataModule中的get_dataloader方法，接受一个train参数（是否是训练），返回一个DataLoader实例（可迭代数据对象）
        data = self.train if train else self.val #如果train为True，返回训练集，否则返回验证集（来源于构造函数）
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train, num_workers=self.num_workers) #将数据集对象data封装成可迭代对象的类，指定每个批次的样本数量（来源于FashionMNIST方法）以及是否打乱数据（来源于get_dataloader方法的输入），并且设置用于数据加载的子进程数量（来源于d2l.DataModule）

    def visualize(self, batch, nrows=1, ncols=8, labels=[]): #定义visualize方法，接受一个批次的数据batch，以及可选的行数nrows、列数ncols和标签列表labels
        X, y = batch #将一个批次的数据分别赋值给X和y
        if not labels: #如果标签列表为空
            labels = self.text_labels(y) #调用text_labels方法，将标签y转换为文本标签
        d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels) #调用d2l.show_images函数，将X中的图像数据展示出来，去掉第二个维度（即通道），指定行数和列数，以及标签列表


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images.

    Defined in :numref:`sec_fashion_mnist`"""
    raise NotImplementedError

class Classifier(d2l.Module):
    """分类模型的基类  Defined in :numref:`sec_classification`"""
    def validation_step(self, batch): #重写d2l.Module中的validation_step方法，接受一个批次的数据batch作为参数
        Y_hat = self(*batch[:-1]) #调用模型的__call__方法，传入batch中除了最后一个元素的所有元素（解包后），得到预测值Y_hat,其第二个维度存储了每个类别的预测得分
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False) #调用loss方法（具体需要自定义，也是d2l.Module中的loss方法的重写），传入Y_hat和batch的最后一个元素（来源于DataLoader，一般是标签），得到损失函数值。然后调用d2l.Module中的plot方法，传入损失函数值，train参数为False
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False) #调用accuracy方法（具体需要自定义），传入Y_hat和batch的最后一个元素，得到准确率。然后调用d2l.Module中的plot方法，传入准确率，train参数为False

    def accuracy(self, Y_hat, Y, averaged=True): #计算预测正确的数量，接受预测值Y_hat、标签Y和是否平均化的参数averaged
        """Compute the number of correct predictions."""
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1])) #将预测值Y_hat的形状调整，保持最后一个维度大小不变（若原始形状是(batch_size, num_classes)，那么该操作不会改变Y_hat的形状。但是若Y_hat的形状更复杂如(batch_size, num_steps, num_classes)，那么该操作会将其变为(batch_size*num_steps,num_classes)
        preds = Y_hat.argmax(axis=1).type(Y.dtype) #返回预测值Y_hat中每一行中最大元素的索引，即预测的类别，然后将数据类型转换为标签Y的数据类型（为了后面的==运算服务）
        compare = (preds == Y.reshape(-1)).type(torch.float32) #将预测值preds和标签进行比较（先将Y转换为1维张量，匹配preds的形状），得到一个布尔张量，1.0表示预测正确，0.0表示预测错误
        return compare.mean() if averaged else compare #如果输入参数averaged为true，返回正确预测的比例，否则返回一个布尔张量（每个图像是否预测正确）

    def loss(self, Y_hat, Y, averaged=True): #重写d2l.Module中的loss方法（交叉熵损失），接受两个参数，第一个参数为数据样本，包含所有数据样本的预测概率（每个样本是一个概率分布列表），第二个参数为标签向量（即样本所属类别），以及一个可选的参数averaged（是否求平均）
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1])) #将Y_hat重塑为二维张量，假设原始形状为 (batch_size, num_samples, num_classes)，则重塑后为 (batch_size*num_samples, num_classes)
        Y = Y.reshape((-1,)) #将Y重塑为一个一维张量，假设原始形状为 (batch_size, num_samples)，则重塑后为 (batch_size*num_samples,)
        return F.cross_entropy(Y_hat, Y, reduction='mean' if averaged else 'none') #传入Y_hat和Y得到交叉熵损失，根据传入参数average判断是返回小批量的交叉熵平均损失还是每个样本的损失，默认返回平均损失

    def layer_summary(self, X_shape): #用于打印神经网络每一层的输出形状，接受参数为输入数据的形状
        """Defined in :numref:`sec_lenet`"""
        X = d2l.randn(*X_shape) #生成一个形状为X_shape的随机张量
        for layer in self.net: #遍历神经网络的每一层
            X = layer(X) #将输入数据X传入每一层，得到每层的输出数据X
            print(layer.__class__.__name__, 'output shape:\t', X.shape) #打印当前层的类名和当前层输出张量的形状

class SoftmaxRegression(d2l.Classifier):  #@save，继承自d2l.Classifier类
    """softmax回归模型"""
    def __init__(self, num_outputs, lr): #构造函数，接受输出数num_outputs和学习率lr作为参数
        super().__init__() #调用父类的构造函数
        self.save_hyperparameters() #保存传入的超参数（构造函数的几个传入参数被保存为self.num_outputs和self.lr）
        self.net = nn.Sequential(nn.Flatten(), 
                                 nn.LazyLinear(num_outputs)) #创建一个Sequential实例，包含一个Flatten层和一个全连接LazyLinear层，Flatten层用于将输入数据展平，LazyLinear的输出数为num_outputs

    def forward(self, X): #重写d2l.Module中的forward方法，接受输入数据X作为参数
        return self.net(X) #调用Sequential实例的__call__方法，传入X，得到模型输出

def cpu():
    """Get the CPU device.

    Defined in :numref:`sec_use_gpu`"""
    return torch.device('cpu')

def gpu(i=0):
    """Get a GPU device.

    Defined in :numref:`sec_use_gpu`"""
    return torch.device(f'cuda:{i}')

def num_gpus():
    """Get the number of available GPUs.

    Defined in :numref:`sec_use_gpu`"""
    return torch.cuda.device_count()

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists.

    Defined in :numref:`sec_use_gpu`"""
    return [gpu(i) for i in range(num_gpus())]

def corr2d(X, K):
    """计算二维互相关运算。
    Defined in :numref:`sec_conv_layer`"""
    h, w = K.shape #获取卷积核张量的高度h和宽度w
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)) #初始化输出张量，全零矩阵，形状由卷积核张量和输入张量形状决定
    for i in range(Y.shape[0]): 
        for j in range(Y.shape[1]):
            #使用两个嵌套for循环遍历输出张量Y的每个元素
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum() #对于Y的第i行第j列个元素，计算张量X的子区域X[i:i+h,j:j+w]和卷积核张量K的按元素乘积的和
    return Y

def init_cnn(module):
    """初始化cnn中的权重，Defined in :numref:`sec_lenet`"""
    if type(module) == nn.Linear or type(module) == nn.Conv2d: #如果是全连接层或卷积层
        nn.init.xavier_uniform_(module.weight) #使用Xavier随机初始化权重

class LeNet(d2l.Classifier):
    """LeNet-5模型，Defined in :numref:`sec_lenet`"""
    def __init__(self, lr=0.1, num_classes=10): #构造函数，接受学习率lr和输出维度/类别数作为参数
        super().__init__() #调用父类的构造函数
        self.save_hyperparameters() ##保存传入的超参数（构造函数的几个传入参数被保存为self.lr和self.num_classes）
        #构造网络
        self.net = nn.Sequential(#顺序连接下面的多个层
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(), #第一层是卷积层，输入通道数延后，输出通道数为6，卷积核大小为5*5，填充为2*2，激活函数为Sigmoid
            nn.AvgPool2d(kernel_size=2, stride=2), #第二层是平均池化层，池化窗口大小为2*2，步幅为2*2
            nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(), #第三层是卷积层，输入通道数延后，输出通道数为16，卷积核大小为5*5，激活函数为Sigmoid
            nn.AvgPool2d(kernel_size=2, stride=2), #第四层是平均池化层，池化窗口大小为2*2，步幅为2*2
            nn.Flatten(), #将多维张量展平为一维张量
            nn.LazyLinear(120), nn.Sigmoid(), #第五层是全连接层，输入维度延后，输出维度为120，激活函数为Sigmoid
            nn.LazyLinear(84), nn.Sigmoid(), #第六层是全连接层，输入维度延后，输出维度为84，激活函数为Sigmoid
            nn.LazyLinear(num_classes)) #第七层是全连接层，输入维度延后，输出维度为num_classes，即类别数

class Residual(nn.Module):
    """残差网络ResNet模型的残差块

    Defined in :numref:`sec_resnet`"""
    def __init__(self, num_channels, use_1x1conv=False, strides=1): #接受几个参数，分别为卷积层的输出通道数num_channels，是否使用额外的1x1卷积层use_1x1conv，卷积层的步幅strides
        super().__init__() #调用父类的构造函数
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=strides) #第一个卷积层，卷积核大小为3，填充为1，步幅为strides，输出通道数为num_channels
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1) #第二个卷积层，卷积核大小为3，填充为1，输出通道数为num_channels
        if use_1x1conv: #如果使用额外的1x1卷积层
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides) #第三个卷积层，卷积核大小为1，步幅为strides，输出通道数为num_channels
        else: #如果不使用额外的1x1卷积层则没有第三个卷积层
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d() #第一个批量规范化层
        self.bn2 = nn.LazyBatchNorm2d() #第二个批量规范化层

    def forward(self, X): #前向传播
        Y = F.relu(self.bn1(self.conv1(X))) #首先是第一个卷积层，然后是批量规范化层，最后是ReLU激活函数
        Y = self.bn2(self.conv2(Y)) #第二个卷积层，然后是批量规范化层
        if self.conv3: #如果使用额外的1x1卷积层
            X = self.conv3(X) #第三个卷积层
        Y += X #将输入X与Y相加，即经过两个卷积层的输出+输入/经过1x1卷积层的输入
        return F.relu(Y) #最后使用ReLU激活函数

class ResNeXtBlock(nn.Module):
    """ResNeXt块

    Defined in :numref:`subsec_residual-blks`"""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False,
                 strides=1): #接受几个参数，分别为卷积层的输出通道数num_channels，分组数groups，每组的卷积核数量bot_mul，是否使用额外的1x1卷积层use_1x1conv，第二个卷积层的步幅strides
        super().__init__() #调用父类的构造函数
        bot_channels = int(round(num_channels * bot_mul)) #计算每组的卷积核数量为输出通道数*，这里round用于四舍五入
        self.conv1 = nn.LazyConv2d(bot_channels, kernel_size=1, stride=1) #第一个卷积层，卷积核大小为1，步幅为1，输出通道数为bot_channels
        self.conv2 = nn.LazyConv2d(bot_channels, kernel_size=3,
                                   stride=strides, padding=1,
                                   groups=bot_channels//groups) #第二个卷积层，卷积核大小为3，填充为1，步幅为strides，分组数为bot_channels/groups
        self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=1) #第三个卷积层，卷积核大小为1，步幅为1，输出通道数为num_channels
        self.bn1 = nn.LazyBatchNorm2d() #第一个批量规范化层
        self.bn2 = nn.LazyBatchNorm2d() #第二个批量规范化层
        self.bn3 = nn.LazyBatchNorm2d() #第三个批量规范化层
        if use_1x1conv: #如果使用额外的1x1卷积层
            self.conv4 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides) #第四个卷积层，卷积核大小为1，步幅为strides，输出通道数为num_channels
            self.bn4 = nn.LazyBatchNorm2d() #第四个批量规范化层
        else: #如果不使用额外的1x1卷积层
            self.conv4 = None #第四个卷积层为空

    def forward(self, X): #前向传播
        Y = F.relu(self.bn1(self.conv1(X))) #首先是第一个卷积层，然后是批量规范化层，最后是ReLU激活函数
        Y = F.relu(self.bn2(self.conv2(Y))) #第二个卷积层，然后是批量规范化层，最后是ReLU激活函数
        Y = self.bn3(self.conv3(Y)) #第三个卷积层，然后是批量规范化层
        if self.conv4: #如果使用额外的1x1卷积层
            X = self.bn4(self.conv4(X)) #第四个卷积层，然后是批量规范化层
        return F.relu(Y + X) #最后将输入X与经过几个卷积层得到的输出Y相加，然后使用ReLU激活函数，为新的输出

class TimeMachine(d2l.DataModule):
    """The Time Machine dataset.

    Defined in :numref:`sec_text-sequence`"""
    def _download(self):
        fname = d2l.download(d2l.DATA_URL + 'timemachine.txt', self.root,
                             '090b5e7e70c295757f55df93cb0a180b9691891a')
        with open(fname) as f:
            return f.read()

    def _preprocess(self, text):
        """Defined in :numref:`sec_text-sequence`"""
        return re.sub('[^A-Za-z]+', ' ', text).lower()

    def _tokenize(self, text):
        """Defined in :numref:`sec_text-sequence`"""
        return list(text)

    def build(self, raw_text, vocab=None):
        """Defined in :numref:`sec_text-sequence`"""
        tokens = self._tokenize(self._preprocess(raw_text))
        if vocab is None: vocab = Vocab(tokens)
        corpus = [vocab[token] for token in tokens]
        return corpus, vocab

    def __init__(self, batch_size, num_steps, num_train=10000, num_val=5000):
        """Defined in :numref:`sec_language-model`"""
        super(d2l.TimeMachine, self).__init__()
        self.save_hyperparameters()
        corpus, self.vocab = self.build(self._download())
        array = d2l.tensor([corpus[i:i+num_steps+1]
                            for i in range(len(corpus)-num_steps)])
        self.X, self.Y = array[:,:-1], array[:,1:]

    def get_dataloader(self, train):
        """Defined in :numref:`subsec_partitioning-seqs`"""
        idx = slice(0, self.num_train) if train else slice(
            self.num_train, self.num_train + self.num_val)
        return self.get_tensorloader([self.X, self.Y], train, idx)

class Vocab:
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        """Defined in :numref:`sec_text-sequence`"""
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']

class RNNScratch(d2l.Module):
    """The RNN model implemented from scratch.

    Defined in :numref:`sec_rnn-scratch`"""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W_xh = nn.Parameter(
            d2l.randn(num_inputs, num_hiddens) * sigma)
        self.W_hh = nn.Parameter(
            d2l.randn(num_hiddens, num_hiddens) * sigma)
        self.b_h = nn.Parameter(d2l.zeros(num_hiddens))

    def forward(self, inputs, state=None):
        """Defined in :numref:`sec_rnn-scratch`"""
        if state is None:
            # Initial state with shape: (batch_size, num_hiddens)
            state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                              device=inputs.device)
        else:
            state, = state
        outputs = []
        for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs)
            state = d2l.tanh(d2l.matmul(X, self.W_xh) +
                             d2l.matmul(state, self.W_hh) + self.b_h)
            outputs.append(state)
        return outputs, state

def check_len(a, n):
    """Check the length of a list.

    Defined in :numref:`sec_rnn-scratch`"""
    assert len(a) == n, f'list\'s length {len(a)} != expected length {n}'

def check_shape(a, shape):
    """Check the shape of a tensor.

    Defined in :numref:`sec_rnn-scratch`"""
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'

class RNNLMScratch(d2l.Classifier):
    """The RNN-based language model implemented from scratch.

    Defined in :numref:`sec_rnn-scratch`"""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()

    def init_params(self):
        self.W_hq = nn.Parameter(
            d2l.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)
        self.b_q = nn.Parameter(d2l.zeros(self.vocab_size))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)

    def one_hot(self, X):
        """Defined in :numref:`sec_rnn-scratch`"""
        # Output shape: (num_steps, batch_size, vocab_size)
        return F.one_hot(X.T, self.vocab_size).type(torch.float32)

    def output_layer(self, rnn_outputs):
        """Defined in :numref:`sec_rnn-scratch`"""
        outputs = [d2l.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
        return d2l.stack(outputs, 1)
    

    def forward(self, X, state=None):
        """Defined in :numref:`sec_rnn-scratch`"""
        embs = self.one_hot(X)
        rnn_outputs, _ = self.rnn(embs, state)
        return self.output_layer(rnn_outputs)

    def predict(self, prefix, num_preds, vocab, device=None):
        """Defined in :numref:`sec_rnn-scratch`"""
        state, outputs = None, [vocab[prefix[0]]]
        for i in range(len(prefix) + num_preds - 1):
            X = d2l.tensor([[outputs[-1]]], device=device)
            embs = self.one_hot(X)
            rnn_outputs, state = self.rnn(embs, state)
            if i < len(prefix) - 1:  # Warm-up period
                outputs.append(vocab[prefix[i + 1]])
            else:  # Predict num_preds steps
                Y = self.output_layer(rnn_outputs)
                outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), 1)))
        return ''.join([vocab.idx_to_token[i] for i in outputs])

class RNN(d2l.Module):
    """The RNN model implemented with high-level APIs.

    Defined in :numref:`sec_rnn-concise`"""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = nn.RNN(num_inputs, num_hiddens)

    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)

class RNNLM(d2l.RNNLMScratch):
    """The RNN-based language model implemented with high-level APIs.

    Defined in :numref:`sec_rnn-concise`"""
    def init_params(self):
        self.linear = nn.LazyLinear(self.vocab_size)

    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)

class GRU(d2l.RNN):
    """The multilayer GRU model.

    Defined in :numref:`sec_deep_rnn`"""
    def __init__(self, num_inputs, num_hiddens, num_layers, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.GRU(num_inputs, num_hiddens, num_layers,
                          dropout=dropout)

class MTFraEng(d2l.DataModule):
    """The English-French dataset.

    Defined in :numref:`sec_machine_translation`"""
    def _download(self):
        d2l.extract(d2l.download(
            d2l.DATA_URL+'fra-eng.zip', self.root,
            '94646ad1522d915e7b0f9296181140edcf86a4f5'))
        with open(self.root + '/fra-eng/fra.txt', encoding='utf-8') as f:
            return f.read()

    def _preprocess(self, text):
        """Defined in :numref:`sec_machine_translation`"""
        # Replace non-breaking space with space
        text = text.replace('\u202f', ' ').replace('\xa0', ' ')
        # Insert space between words and punctuation marks
        no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
        out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
               for i, char in enumerate(text.lower())]
        return ''.join(out)

    def _tokenize(self, text, max_examples=None):
        """Defined in :numref:`sec_machine_translation`"""
        src, tgt = [], []
        for i, line in enumerate(text.split('\n')):
            if max_examples and i > max_examples: break
            parts = line.split('\t')
            if len(parts) == 2:
                # Skip empty tokens
                src.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
                tgt.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])
        return src, tgt

    def __init__(self, batch_size, num_steps=9, num_train=512, num_val=128):
        """Defined in :numref:`sec_machine_translation`"""
        super(MTFraEng, self).__init__()
        self.save_hyperparameters()
        self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(
            self._download())

    def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
        """Defined in :numref:`subsec_loading-seq-fixed-len`"""
        def _build_array(sentences, vocab, is_tgt=False):
            pad_or_trim = lambda seq, t: (
                seq[:t] if len(seq) > t else seq + ['<pad>'] * (t - len(seq)))
            sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
            if is_tgt:
                sentences = [['<bos>'] + s for s in sentences]
            if vocab is None:
                vocab = d2l.Vocab(sentences, min_freq=2)
            array = d2l.tensor([vocab[s] for s in sentences])
            valid_len = d2l.reduce_sum(
                d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
            return array, vocab, valid_len
        src, tgt = self._tokenize(self._preprocess(raw_text),
                                  self.num_train + self.num_val)
        src_array, src_vocab, src_valid_len = _build_array(src, src_vocab)
        tgt_array, tgt_vocab, _ = _build_array(tgt, tgt_vocab, True)
        return ((src_array, tgt_array[:,:-1], src_valid_len, tgt_array[:,1:]),
                src_vocab, tgt_vocab)

    def get_dataloader(self, train):
        """Defined in :numref:`subsec_loading-seq-fixed-len`"""
        idx = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader(self.arrays, train, idx)

    def build(self, src_sentences, tgt_sentences):
        """Defined in :numref:`subsec_loading-seq-fixed-len`"""
        raw_text = '\n'.join([src + '\t' + tgt for src, tgt in zip(
            src_sentences, tgt_sentences)])
        arrays, _, _ = self._build_arrays(
            raw_text, self.src_vocab, self.tgt_vocab)
        return arrays

def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """Plot the histogram for list length pairs.

    Defined in :numref:`sec_machine_translation`"""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)

class Encoder(nn.Module):
    """The base encoder interface for the encoder--decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def forward(self, X, *args):
        raise NotImplementedError

class Decoder(nn.Module):
    """The base decoder interface for the encoder--decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError

class EncoderDecoder(d2l.Classifier):
    """The base class for the encoder--decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state)[0]

    def predict_step(self, batch, device, num_steps,
                     save_attention_weights=False):
        """Defined in :numref:`sec_seq2seq_training`"""
        batch = [d2l.to(a, device) for a in batch]
        src, tgt, src_valid_len, _ = batch
        enc_all_outputs = self.encoder(src, src_valid_len)
        dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
        outputs, attention_weights = [d2l.expand_dims(tgt[:, 0], 1), ], []
        for _ in range(num_steps):
            Y, dec_state = self.decoder(outputs[-1], dec_state)
            outputs.append(d2l.argmax(Y, 2))
            # Save attention weights (to be covered later)
            if save_attention_weights:
                attention_weights.append(self.decoder.attention_weights)
        return d2l.concat(outputs[1:], 1), attention_weights

def init_seq2seq(module):
    """Initialize weights for sequence-to-sequence learning.

    Defined in :numref:`sec_seq2seq`"""
    if type(module) == nn.Linear:
         nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])

class Seq2SeqEncoder(d2l.Encoder):
    """The RNN encoder for sequence-to-sequence learning.

    Defined in :numref:`sec_seq2seq`"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(embed_size, num_hiddens, num_layers, dropout)
        self.apply(init_seq2seq)

    def forward(self, X, *args):
        # X shape: (batch_size, num_steps)
        embs = self.embedding(d2l.astype(d2l.transpose(X), d2l.int64))
        # embs shape: (num_steps, batch_size, embed_size)
        outputs, state = self.rnn(embs)
        # outputs shape: (num_steps, batch_size, num_hiddens)
        # state shape: (num_layers, batch_size, num_hiddens)
        return outputs, state

class Seq2Seq(d2l.EncoderDecoder):
    """The RNN encoder--decoder for sequence to sequence learning.

    Defined in :numref:`sec_seq2seq_decoder`"""
    def __init__(self, encoder, decoder, tgt_pad, lr):
        super().__init__(encoder, decoder)
        self.save_hyperparameters()

    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)

    def configure_optimizers(self):
        # Adam optimizer is used here
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def bleu(pred_seq, label_seq, k):
    """Compute the BLEU.

    Defined in :numref:`sec_seq2seq_training`"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, min(k, len_pred) + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """Show heatmaps of matrices.

    Defined in :numref:`sec_queries-keys-values`"""
    d2l.use_svg_display()
    num_rows, num_cols, _, _ = matrices.shape
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(d2l.numpy(matrix), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);

def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis.

    Defined in :numref:`sec_attention-scoring-functions`"""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class DotProductAttention(nn.Module):
    """Scaled dot product attention.

    Defined in :numref:`subsec_batch_dot`"""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

class AdditiveAttention(nn.Module):
    """Additive attention.

    Defined in :numref:`subsec_batch_dot`"""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)

class AttentionDecoder(d2l.Decoder):
    """The base attention-based decoder interface.

    Defined in :numref:`sec_seq2seq_attention`"""
    def __init__(self):
        super().__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError

class MultiHeadAttention(d2l.Module):
    """多头注意力机制."""
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs): #构造函数，接受num_hiddens隐藏单元数（方法里的p_o），num_heads注意力头数，dropout丢弃概率，bias是否使用偏置
        super().__init__() #调用父类d2l.Module的构造函数
        self.num_heads = num_heads #初始化注意力头数
        self.attention = d2l.DotProductAttention(dropout) #初始化注意力评分函数为缩放点积注意力
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias) #初始化查询的权重矩阵，输出维度为num_hiddens，是否使用偏置由bias决定
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias) #初始化键的权重矩阵，输出维度为num_hiddens，是否使用偏置由bias决定
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias) #初始化值的权重矩阵，输出维度为num_hiddens，是否使用偏置由bias决定
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias) #初始化多头注意力输出的权重矩阵，输出维度为num_hiddens，是否使用偏置由bias决定

    def forward(self, queries, keys, values, valid_lens): 
        #定义前向传播函数，接受四个参数，分别为查询，键，值，有效长度，其中valid_lens的形状为: (batch_size,)或(batch_size, 查询的数量)；查询、键、值的形状为：(batch_size, 查询/键/值的数量, 查询/键/值的长度)
        
        #经过W（全连接层）后，查询、键和值的形状分别为：(batch_size,查询/键/值的数量,num_hiddens)
        #转换后,查询、键和值的输出的形状分别为:(batch_size*注意力头数,查询或键值对的数量,num_hiddens/注意力头数)
        queries = self.transpose_qkv(self.W_q(queries)) 
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))
        

        if valid_lens is not None:
            #在轴0上，将第一个元素（标量或向量）复制num_heads次，然后复制下一个元素，以此类推。
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        #经过点积注意力评分函数的输出形状为(batch_size*注意力头数, 查询的数量, num_hiddens/注意力头数)
        output = self.attention(queries, keys, values, valid_lens)
        #将输出进行之前qkv转换的反转，形状变为(batch_size, 查询的数量, num_hiddens)，本质是对多个注意力输出进行拼接
        output_concat = self.transpose_output(output)

        return self.W_o(output_concat) #将拼接后的注意力输出通过全连接层，得到多头注意力的输出，形状(batch_size, 查询的数量, num_hiddens)

    def transpose_qkv(self, X):
        """"用于多头注意力并行计算的转置操作"""
        # 输入X的形状: (batch_size, 查询数量或键值对数量,num_hiddens) 
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1) #重塑X的形状为(batch_size, 查询数量或键值对数量, num_heads, num_hiddens / num_heads)
        X = X.permute(0, 2, 1, 3) #交换X的第二个维度和第三个维度，形状变为(batch_size, num_heads, 查询数量或键值对数量, num_hiddens / num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3]) #返回重塑后的X，形状为(batch_size * num_heads, 查询数量或键值对数量, num_hiddens / num_heads)

    def transpose_output(self, X): #输入X的形状为(batch_size * num_heads, 查询数量或键值对数量, num_hiddens / num_heads)
        """反转transpose_qkv的操作"""
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2]) #将X形状重塑为(batch_size, num_heads, 查询数量或键值对数量, num_hiddens / num_heads)
        X = X.permute(0, 2, 1, 3) #交换X的第二个维度和第三个维度，形状变为(batch_size, 查询数量或键值对数量, num_heads, num_hiddens / num_heads)
        return X.reshape(X.shape[0], X.shape[1], -1) #返回重塑后的X，形状变为(batch_size, 查询数量或键值对数量, num_hiddens)

class PositionalEncoding(nn.Module):
    """Positional encoding.

    Defined in :numref:`sec_self-attention-and-positional-encoding`"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class PositionWiseFFN(nn.Module):
    """The positionwise feed-forward network.

    Defined in :numref:`sec_transformer`"""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    """The residual connection followed by layer normalization.

    Defined in :numref:`subsec_positionwise-ffn`"""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class TransformerEncoderBlock(nn.Module):
    """The Transformer encoder block.

    Defined in :numref:`subsec_positionwise-ffn`"""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder(d2l.Encoder):
    """The Transformer encoder.

    Defined in :numref:`subsec_transformer-encoder`"""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X

def annotate(text, xy, xytext):
    """Defined in :numref:`sec_optimization-intro`"""
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

def train_2d(trainer, steps=20, f_grad=None):
    """Optimize a 2D objective function with a customized trainer.

    Defined in :numref:`subsec_gd-learningrate`"""
    # `s1` and `s2` are internal state variables that will be used in Momentum, adagrad, RMSProp
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results

def show_trace_2d(f, results):
    """Show the trace of 2D variables during optimization.

    Defined in :numref:`subsec_gd-learningrate`"""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1), indexing='ij')
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')

class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`sec_minibatch_sgd`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

def get_data_ch11(batch_size=10, n=1500):
    """Defined in :numref:`sec_minibatches`"""
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1

def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    """Defined in :numref:`sec_minibatches`"""
    # Initialization
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]

def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    """Defined in :numref:`sec_minibatches`"""
    # Initialization
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(module):
        if type(module) == nn.Linear:
            torch.nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                # `MSELoss` computes squared error without the 1/2 factor
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')

class Benchmark:
    """For measuring running time."""
    def __init__(self, description='Done'):
        """Defined in :numref:`sec_hybridize`"""
        self.description = description

    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')

def split_batch(X, y, devices):
    """Split `X` and `y` into multiple devices.

    Defined in :numref:`sec_multi_gpu`"""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))

def resnet18(num_classes, in_channels=1):
    """A slightly modified ResNet-18 model.

    Defined in :numref:`sec_multi_gpu_concise`"""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(out_channels, use_1x1conv=True,
                                        strides=2))
            else:
                blk.append(d2l.Residual(out_channels))
        return nn.Sequential(*blk)

    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the max-pooling layer
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net

def train_batch_ch13(net, X, y, loss, trainer, devices):
    """Train for a minibatch with multiple GPUs (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """Train a model with multiple GPUs (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')

d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')

def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height).

    Defined in :numref:`sec_bbox`"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = d2l.stack((cx, cy, w, h), axis=-1)
    return boxes

def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right).

    Defined in :numref:`sec_bbox`"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = d2l.stack((x1, y1, x2, y2), axis=-1)
    return boxes

def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format.

    Defined in :numref:`sec_bbox`"""
    # Convert the bounding box (upper-left x, upper-left y, lower-right x,
    # lower-right y) format to the matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)

def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel.

    Defined in :numref:`sec_anchor`"""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, device=device)
    ratio_tensor = d2l.tensor(ratios, device=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y axis
    steps_w = 1.0 / in_width  # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # Handle rectangular inputs
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)

def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes.

    Defined in :numref:`sec_anchor`"""

    def make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = make_list(labels)
    colors = make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(d2l.numpy(bbox), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes.

    Defined in :numref:`sec_anchor`"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes.

    Defined in :numref:`sec_anchor`"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offsets.

    Defined in :numref:`subsec_labeling-anchor-boxes`"""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * d2l.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = d2l.concat([offset_xy, offset_wh], axis=1)
    return offset

def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes.

    Defined in :numref:`subsec_labeling-anchor-boxes`"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)

def offset_inverse(anchors, offset_preds):
    """Predict bounding boxes based on anchor boxes with predicted offsets.

    Defined in :numref:`subsec_labeling-anchor-boxes`"""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = d2l.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = d2l.concat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox

def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes.

    Defined in :numref:`subsec_predicting-bounding-boxes-nms`"""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return d2l.tensor(keep, device=boxes.device)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression.

    Defined in :numref:`subsec_predicting-bounding-boxes-nms`"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return d2l.stack(out)

d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')

def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels.

    Defined in :numref:`sec_object-detection-dataset`"""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256

class BananasDataset(torch.utils.data.Dataset):
    """A customized dataset to load the banana detection dataset.

    Defined in :numref:`sec_object-detection-dataset`"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)

def load_data_bananas(batch_size):
    """Load the banana detection dataset.

    Defined in :numref:`sec_object-detection-dataset`"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter

d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images.

    Defined in :numref:`sec_semantic_segmentation`"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

def voc_colormap2label():
    """Build the mapping from RGB to class indices for VOC labels.

    Defined in :numref:`sec_semantic_segmentation`"""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

def voc_label_indices(colormap, colormap2label):
    """Map any RGB values in VOC labels to their class indices.

    Defined in :numref:`sec_semantic_segmentation`"""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]

def voc_rand_crop(feature, label, height, width):
    """Randomly crop both feature and label images.

    Defined in :numref:`sec_semantic_segmentation`"""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

class VOCSegDataset(torch.utils.data.Dataset):
    """A customized dataset to load the VOC dataset.

    Defined in :numref:`sec_semantic_segmentation`"""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)

def load_data_voc(batch_size, crop_size):
    """Load the VOC semantic segmentation dataset.

    Defined in :numref:`sec_semantic_segmentation`"""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter

d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
                                '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')

def read_csv_labels(fname):
    """Read `fname` to return a filename to label dictionary.

    Defined in :numref:`sec_kaggle_cifar10`"""
    with open(fname, 'r') as f:
        # Skip the file header line (column name)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

def copyfile(filename, target_dir):
    """Copy a file into a target directory.

    Defined in :numref:`sec_kaggle_cifar10`"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

def reorg_train_valid(data_dir, labels, valid_ratio):
    """Split the validation set out of the original training set.

    Defined in :numref:`sec_kaggle_cifar10`"""
    # The number of examples of the class that has the fewest examples in the
    # training dataset
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # The number of examples per class for the validation set
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label

def reorg_test(data_dir):
    """Organize the testing set for data loading during prediction.

    Defined in :numref:`sec_kaggle_cifar10`"""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))

d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                            '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')

d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')

def read_ptb():
    """Load the PTB dataset into a list of text lines.

    Defined in :numref:`sec_word2vec_data`"""
    data_dir = d2l.download_extract('ptb')
    # Read the training set
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

def subsample(sentences, vocab):
    """Subsample high-frequency words.

    Defined in :numref:`sec_word2vec_data`"""
    # Exclude unknown tokens ('<unk>')
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = collections.Counter([
        token for line in sentences for token in line])
    num_tokens = sum(counter.values())

    # Return True if `token` is kept during subsampling
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

def get_centers_and_contexts(corpus, max_window_size):
    """Return center words and context words in skip-gram.

    Defined in :numref:`sec_word2vec_data`"""
    centers, contexts = [], []
    for line in corpus:
        # To form a "center word--context word" pair, each sentence needs to
        # have at least 2 words
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # Context window centered at `i`
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # Exclude the center word from the context words
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts

class RandomGenerator:
    """Randomly draw among {1, ..., n} according to n sampling weights."""
    def __init__(self, sampling_weights):
        """Defined in :numref:`sec_word2vec_data`"""
        # Exclude
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # Cache `k` random sampling results
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]

def get_negatives(all_contexts, vocab, counter, K):
    """Return noise words in negative sampling.

    Defined in :numref:`sec_word2vec_data`"""
    # Sampling weights for words with indices 1, 2, ... (index 0 is the
    # excluded unknown token) in the vocabulary
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # Noise words cannot be context words
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

def batchify(data):
    """Return a minibatch of examples for skip-gram with negative sampling.

    Defined in :numref:`sec_word2vec_data`"""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (d2l.reshape(d2l.tensor(centers), (-1, 1)), d2l.tensor(
        contexts_negatives), d2l.tensor(masks), d2l.tensor(labels))

def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory.

    Defined in :numref:`subsec_word2vec-minibatch-loading`"""
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
                                      collate_fn=batchify,
                                      num_workers=num_workers)
    return data_iter, vocab

d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')

d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')

class TokenEmbedding:
    """Token Embedding."""
    def __init__(self, embedding_name):
        """Defined in :numref:`sec_synonyms`"""
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)

def get_tokens_and_segments(tokens_a, tokens_b=None):
    """Get tokens of the BERT input sequence and their segment IDs.

    Defined in :numref:`sec_bert`"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 and 1 are marking segment A and B, respectively
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

class BERTEncoder(nn.Module):
    """BERT encoder.

    Defined in :numref:`subsec_bert_input_rep`"""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout, max_len=1000, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", d2l.TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, True))
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X

class MaskLM(nn.Module):
    """The masked language model task of BERT.

    Defined in :numref:`subsec_bert_input_rep`"""
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.LazyLinear(num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.LazyLinear(vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `torch.tensor([0, 0, 0, 1, 1, 1])`
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat

class NextSentencePred(nn.Module):
    """The next sentence prediction task of BERT.

    Defined in :numref:`subsec_mlm`"""
    def __init__(self, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.LazyLinear(2)

    def forward(self, X):
        # `X` shape: (batch size, `num_hiddens`)
        return self.output(X)

class BERTModel(nn.Module):
    """The BERT model.

    Defined in :numref:`subsec_nsp`"""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, max_len=1000):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                   num_heads, num_blks, dropout,
                                   max_len=max_len)
        self.hidden = nn.Sequential(nn.LazyLinear(num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens)
        self.nsp = NextSentencePred()

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat

d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')

def _read_wiki(data_dir):
    """Defined in :numref:`sec_bert-dataset`"""
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # Uppercase letters are converted to lowercase ones
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs

def _get_next_sentence(sentence, next_sentence, paragraphs):
    """Defined in :numref:`sec_bert-dataset`"""
    if random.random() < 0.5:
        is_next = True
    else:
        # `paragraphs` is a list of lists of lists
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next

def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    """Defined in :numref:`sec_bert-dataset`"""
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # Consider 1 '<cls>' token and 2 '<sep>' tokens
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph

def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    """Defined in :numref:`sec_bert-dataset`"""
    # For the input of a masked language model, make a new copy of tokens and
    # replace some of them by '<mask>' or random tokens
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # Shuffle for getting 15% random tokens for prediction in the masked
    # language modeling task
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80% of the time: replace the word with the '<mask>' token
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10% of the time: keep the word unchanged
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10% of the time: replace the word with a random word
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels

def _get_mlm_data_from_tokens(tokens, vocab):
    """Defined in :numref:`subsec_prepare_mlm_data`"""
    candidate_pred_positions = []
    # `tokens` is a list of strings
    for i, token in enumerate(tokens):
        # Special tokens are not predicted in the masked language modeling
        # task
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 15% of random tokens are predicted in the masked language modeling task
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]

def _pad_bert_inputs(examples, max_len, vocab):
    """Defined in :numref:`subsec_prepare_mlm_data`"""
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        # `valid_lens` excludes count of '<pad>' tokens
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)

class _WikiTextDataset(torch.utils.data.Dataset):
    """Defined in :numref:`subsec_prepare_mlm_data`"""
    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Get data for the masked language model task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)

def load_data_wiki(batch_size, max_len):
    """Load the WikiText-2 dataset.

    Defined in :numref:`subsec_prepare_mlm_data`"""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab

def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    """Defined in :numref:`sec_bert-pretraining`"""
    # Forward pass
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # Compute masked language model loss
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # Compute next sentence prediction loss
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l

d2l.DATA_HUB['aclImdb'] = (d2l.DATA_URL + 'aclImdb_v1.tar.gz',
                          '01ada507287d82875905620988597833ad4e0903')

def read_imdb(data_dir, is_train):
    """Read the IMDb review dataset text sequences and labels.

    Defined in :numref:`sec_sentiment`"""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset.

    Defined in :numref:`sec_sentiment`"""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab

def predict_sentiment(net, vocab, sequence):
    """Predict the sentiment of a text sequence.

    Defined in :numref:`sec_sentiment_rnn`"""
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'

d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

def read_snli(data_dir, is_train):
    """Read the SNLI dataset into premises, hypotheses, and labels.

    Defined in :numref:`sec_natural-language-inference-and-dataset`"""
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels

class SNLIDataset(torch.utils.data.Dataset):
    """A customized dataset to load the SNLI dataset.

    Defined in :numref:`sec_natural-language-inference-and-dataset`"""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)

def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary.

    Defined in :numref:`sec_natural-language-inference-and-dataset`"""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab

def predict_snli(net, vocab, premise, hypothesis):
    """Predict the logical relationship between the premise and hypothesis.

    Defined in :numref:`sec_natural-language-inference-attention`"""
    net.eval()
    premise = torch.tensor(vocab[premise], device=d2l.try_gpu())
    hypothesis = torch.tensor(vocab[hypothesis], device=d2l.try_gpu())
    label = torch.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), dim=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'

def rbfkernel(x1, x2, ls=4.):
    dist = distance_matrix(np.expand_dims(x1, 1), np.expand_dims(x2, 1))
    return np.exp(-(1. / ls / 2) * (dist ** 2))

class HPOTrainer(d2l.Trainer):  #@save，承自d2l.Trainer类
    def validation_error(self): #计算验证集上的误差，定义为1-验证集所有小批次的准确率之和/验证集小批次总数
        self.model.eval() #将模型设置为评估模式
        accuracy = 0 #初始化准确率为0
        val_batch_idx = 0 #初始化验证批次索引为0
        for batch in self.val_dataloader: #遍历验证集（来源于d2l.Trainer）的每一个批次
            with torch.no_grad(): #不追踪梯度
                x, y = self.prepare_batch(batch) #获取一个批次的数据和标签
                y_hat = self.model(x) #前向传播获得模型预测
                accuracy += self.model.accuracy(y_hat, y) #计算准确率(来源于模型d2l.Classifier)
            val_batch_idx += 1 #验证批次索引加1
        return 1 -  accuracy / val_batch_idx #返回验证集上的误差，即1-平均准确率（所有小批次的准确率之和/小批次总数）



class HPOSearcher(d2l.HyperParameters):  #@save，承自d2l.HyperParameters类
    def sample_configuration() -> dict: #定义采样配置方法（提供新的候选超参数配置），该方法没有参数，会返回一个字典dict
        raise NotImplementedError #该方法目前没有实现，子类需要重写

    def update(self, config: dict, error: float, additional_info=None): #update方法，用于更新搜索器的内部状态（然后可以利用这些记录来改进采样分布），传入超参数配置config（字典）、误差error（浮点数）和额外信息additional_info（可选参数，默认为None）
        pass #该方法目前没有实现，pass表示什么都不做



class RandomSearcher(HPOSearcher):  #@save，承自HPOSearcher类，用于实现随机搜索（高级API方法）
    def __init__(self, config_space: dict, initial_config=None): #构造函数，传入超参数空间config_space（字典）和初始超参数配置initial_config（可选参数，默认为None）
        self.save_hyperparameters() #保存传入的超参数（构造函数的几个传入参数被保存为self.config_space、self.initial_config属性）

    def sample_configuration(self) -> dict: #重写HPOSearcher类的sample_configuration方法，返回一个字典dict
        if self.initial_config is not None: #如果初始超参数配置不为空
            result = self.initial_config #将初始超参数配置保存为result
            self.initial_config = None #将初始配置self.initial_config置为None
        else: #如果初始配置为空
            result = {
                name: domain.rvs() #rvs表示从对应超参数的搜索空间中随机采样一个值作为字典的值
                for name, domain in self.config_space.items() #字典推导式，遍历config_space字典的每一个键值对（items()返回一个视图对象，包含字典中所有键值对），键为name（超参数名称），值为domain（对应超参数的搜索空间）
            } #从超参数空间中随机采样一个超参数配置
        return result



class HPOScheduler(d2l.HyperParameters):  #@save，承自d2l.HyperParameters类
    def suggest(self) -> dict: #定义suggest方法，该方法没有参数，会返回一个字典dict
        raise NotImplementedError #该方法目前没有实现，子类需要重写

    def update(self, config: dict, error: float, info=None): #update方法，用于更新搜索器的内部状态，传入超参数配置config（字典）、误差error（浮点数）和信息info（可选参数，默认为None）
        raise NotImplementedError #该方法目前没有实现，子类需要重写

class BasicScheduler(HPOScheduler):  #@save，承自HPOScheduler类，用于实现基本的调度器（高级API方法）
    def __init__(self, searcher: HPOSearcher): #构造函数，传入搜索器searcher（HPOSearcher类的实例）
        self.save_hyperparameters() #保存传入的超参数（构造函数的几个传入参数被保存为self.searcher属性）

    def suggest(self) -> dict: #重写HPOScheduler类的suggest方法，返回一个字典dict
        return self.searcher.sample_configuration() #调用搜索器的sample_configuration方法（d2l.HPOSearcher），返回一个超参数配置(需要重写)

    def update(self, config: dict, error: float, info=None): #重写HPOScheduler类的update方法，传入超参数配置config（字典）、误差error（浮点数）和信息info（可选参数，默认为None）
        self.searcher.update(config, error, additional_info=info) #调用搜索器的update方法（d2l.HPOSearcher），更新搜索器的内部状态


class HPOTuner(d2l.HyperParameters):  #@save，承自d2l.HyperParameters类
    def __init__(self, scheduler: HPOScheduler, objective: callable): #构造函数，传入调度器scheduler（HPOScheduler类的实例）和目标函数objective（可调用对象，用于计算验证误差）
        self.save_hyperparameters() #保存传入的超参数（构造函数的几个传入参数被保存为self.scheduler、self.objective属性）
        # Bookeeping results for plotting
        self.incumbent = None #初始化最优超参数配置incumbent为None
        self.incumbent_error = None #初始化最优误差incumbent_error为None
        self.incumbent_trajectory = [] #初始化最优误差轨迹incumbent_trajectory为空列表（存储了每次迭代的验证误差（对应每一组超参数））
        self.cumulative_runtime = [] #初始化累计运行时间cumulative_runtime为空列表
        self.current_runtime = 0 #初始化当前运行时间current_runtime为0
        self.records = [] #记录每次试验的结果，是一个字典，包含超参数、验证误差和运行时间

    def run(self, number_of_trials): #定义run方法，用于运行超参数优化，传入试验次数number_of_trials
        for i in range(number_of_trials): #迭代number_of_trials次
            start_time = time.time() #记录当前时间
            config = self.scheduler.suggest() #调用调度器的suggest方法（d2l.HPOScheduler），返回一个超参数配置
            print(f"Trial {i}: config = {config}") #打印当前试验次数和超参数配置
            error = self.objective(**config) #调用目标函数，传入超参数配置，返回验证误差
            error = float(error.cpu().detach().numpy()) #将验证误差转换为numpy数组，再转换为浮点数
            self.scheduler.update(config, error) #调用调度器的update方法，传入超参数配置和验证误差，更新调度器的内部状态
            runtime = time.time() - start_time #计算当前试验的运行时间
            self.bookkeeping(config, error, runtime) #调用bookkeeping方法，记录当前HPO算法的性能
            print(f"    error = {error}, runtime = {runtime}") #打印当前迭代次数的验证误差和运行时间

    def bookkeeping(self, config: dict, error: float, runtime: float): #定义bookkeeping方法，用于记录当前HPO算法的性能，传入超参数配置config（字典）、验证误差error（浮点数）和运行时间runtime（浮点数）
        self.records.append({"config": config, "error": error, "runtime": runtime}) #将当前试验的超参数配置、验证误差和运行时间记录到records列表中（来源于构造函数）
        #检查最后一个超参数配置是否比当前最优超参数配置（incumbent）表现更好
        if self.incumbent is None or self.incumbent_error > error:
            self.incumbent = config #将当前超参数配置设置为最优超参数配置
            self.incumbent_error = error #将当前验证误差设置为最优验证误差
        #将当前观察到的最佳验证误差添加到优化轨迹中
        self.incumbent_trajectory.append(self.incumbent_error)
        #更新运行时间
        self.current_runtime += runtime #当前运行时间为上一次运行时间加上此次update的时间
        self.cumulative_runtime.append(self.current_runtime) #将当前运行时间添加到累计运行时间列表中


def hpo_objective_lenet(learning_rate, batch_size, max_epochs=10):
    """Defined in :numref:`sec_api_hpo`"""
    model = d2l.LeNet(lr=learning_rate, num_classes=10)
    trainer = d2l.HPOTrainer(max_epochs=max_epochs, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size)
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    trainer.fit(model=model, data=data)
    validation_error = trainer.validation_error()
    return validation_error

class SuccessiveHalvingScheduler(d2l.HPOScheduler):
    """Defined in :numref:`sec_mf_hpo`"""
    def __init__(self, searcher, eta, r_min, r_max, prefact=1):
        self.save_hyperparameters()
        # Compute K, which is later used to determine the number of configurations
        self.K = int(np.log(r_max / r_min) / np.log(eta))
        # Define the rungs
        self.rung_levels = [r_min * eta ** k for k in range(self.K + 1)]
        if r_max not in self.rung_levels:
            # The final rung should be r_max
            self.rung_levels.append(r_max)
            self.K += 1
        # Bookkeeping
        self.observed_error_at_rungs = defaultdict(list)
        self.all_observed_error_at_rungs = defaultdict(list)
        # Our processing queue
        self.queue = []

    def suggest(self):
        """Defined in :numref:`sec_mf_hpo_sh`"""
        if len(self.queue) == 0:
            # Start a new round of successive halving
            # Number of configurations for the first rung:
            n0 = int(self.prefact * self.eta ** self.K)
            for _ in range(n0):
                config = self.searcher.sample_configuration()
                config["max_epochs"] = self.r_min  # Set r = r_min
                self.queue.append(config)
        # Return an element from the queue
        return self.queue.pop()

    def update(self, config: dict, error: float, info=None):
        """Defined in :numref:`sec_mf_hpo_sh`"""
        ri = int(config["max_epochs"])  # Rung r_i
        # Update our searcher, e.g if we use Bayesian optimization later
        self.searcher.update(config, error, additional_info=info)
        self.all_observed_error_at_rungs[ri].append((config, error))
        if ri < self.r_max:
            # Bookkeeping
            self.observed_error_at_rungs[ri].append((config, error))
            # Determine how many configurations should be evaluated on this rung
            ki = self.K - self.rung_levels.index(ri)
            ni = int(self.prefact * self.eta ** ki)
            # If we observed all configuration on this rung r_i, we estimate the
            # top 1 / eta configuration, add them to queue and promote them for
            # the next rung r_{i+1}
            if len(self.observed_error_at_rungs[ri]) >= ni:
                kiplus1 = ki - 1
                niplus1 = int(self.prefact * self.eta ** kiplus1)
                best_performing_configurations = self.get_top_n_configurations(
                    rung_level=ri, n=niplus1
                )
                riplus1 = self.rung_levels[self.K - kiplus1]  # r_{i+1}
                # Queue may not be empty: insert new entries at the beginning
                self.queue = [
                    dict(config, max_epochs=riplus1)
                    for config in best_performing_configurations
                ] + self.queue
                self.observed_error_at_rungs[ri] = []  # Reset

    def get_top_n_configurations(self, rung_level, n):
        """Defined in :numref:`sec_mf_hpo_sh`"""
        rung = self.observed_error_at_rungs[rung_level]
        if not rung:
            return []
        sorted_rung = sorted(rung, key=lambda x: x[1])
        return [x[0] for x in sorted_rung[:n]]

def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Update discriminator.

    Defined in :numref:`sec_basic_gan`"""
    batch_size = X.shape[0]
    ones = torch.ones((batch_size,), device=X.device)
    zeros = torch.zeros((batch_size,), device=X.device)
    trainer_D.zero_grad()
    real_Y = net_D(X)
    fake_X = net_G(Z)
    # Do not need to compute gradient for `net_G`, detach it from
    # computing gradients.
    fake_Y = net_D(fake_X.detach())
    loss_D = (loss(real_Y, ones.reshape(real_Y.shape)) +
              loss(fake_Y, zeros.reshape(fake_Y.shape))) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D

def update_G(Z, net_D, net_G, loss, trainer_G):
    """Update generator.

    Defined in :numref:`sec_basic_gan`"""
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size,), device=Z.device)
    trainer_G.zero_grad()
    # We could reuse `fake_X` from `update_D` to save computation
    fake_X = net_G(Z)
    # Recomputing `fake_Y` is needed since `net_D` is changed
    fake_Y = net_D(fake_X)
    loss_G = loss(fake_Y, ones.reshape(fake_Y.shape))
    loss_G.backward()
    trainer_G.step()
    return loss_G

d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip',
                           'c065c0e2593b8b161a2d7873e42418bf6a21106c')

def frozen_lake(seed):
    """Defined in :numref:`sec_utils`"""
    # See https://www.gymlibrary.dev/environments/toy_text/frozen_lake/ to learn more about this env
    # How to process env.P.items is adpated from https://sites.google.com/view/deep-rl-bootcamp/labs
    import gym

    env = gym.make('FrozenLake-v1', is_slippery=False)
    env.seed(seed)
    env.action_space.np_random.seed(seed)
    env.action_space.seed(seed)
    env_info = {}
    env_info['desc'] = env.desc  # 2D array specifying what each grid item means
    env_info['num_states'] = env.nS  # Number of observations/states or obs/state dim
    env_info['num_actions'] = env.nA  # Number of actions or action dim
    # Define indices for (transition probability, nextstate, reward, done) tuple
    env_info['trans_prob_idx'] = 0  # Index of transition probability entry
    env_info['nextstate_idx'] = 1  # Index of next state entry
    env_info['reward_idx'] = 2  # Index of reward entry
    env_info['done_idx'] = 3  # Index of done entry
    env_info['mdp'] = {}
    env_info['env'] = env

    for (s, others) in env.P.items():
        # others(s) = {a0: [ (p(s'|s,a0), s', reward, done),...], a1:[...], ...}

        for (a, pxrds) in others.items():
            # pxrds is [(p1,next1,r1,d1),(p2,next2,r2,d2),..].
            # e.g. [(0.3, 0, 0, False), (0.3, 0, 0, False), (0.3, 4, 1, False)]
            env_info['mdp'][(s,a)] = pxrds

    return env_info

def make_env(name ='', seed=0):
    """Defined in :numref:`sec_utils`"""
    # Input parameters:
    # name: specifies a gym environment.
    # For Value iteration, only FrozenLake-v1 is supported.
    if name == 'FrozenLake-v1':
        return frozen_lake(seed)

    else:
        raise ValueError("%s env is not supported in this Notebook")

def show_value_function_progress(env_desc, V, pi):
    """Defined in :numref:`sec_utils`"""
    # This function visualizes how value and policy changes over time.
    # V: [num_iters, num_states]
    # pi: [num_iters, num_states]
    # How to visualize value function is adapted (but changed) from: https://sites.google.com/view/deep-rl-bootcamp/labs

    num_iters = V.shape[0]
    fig, ax  = plt.subplots(figsize=(15, 15))

    for k in range(V.shape[0]):
        plt.subplot(4, 4, k + 1)
        plt.imshow(V[k].reshape(4,4), cmap="bone")
        ax = plt.gca()
        ax.set_xticks(np.arange(0, 5)-.5, minor=True)
        ax.set_yticks(np.arange(0, 5)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticks([])
        ax.set_yticks([])

        # LEFT action: 0, DOWN action: 1
        # RIGHT action: 2, UP action: 3
        action2dxdy = {0:(-.25, 0),1: (0, .25),
                       2:(0.25, 0),3: (-.25, 0)}

        for y in range(4):
            for x in range(4):
                action = pi[k].reshape(4,4)[y, x]
                dx, dy = action2dxdy[action]

                if env_desc[y,x].decode() == 'H':
                    ax.text(x, y, str(env_desc[y,x].decode()),
                       ha="center", va="center", color="y",
                         size=20, fontweight='bold')

                elif env_desc[y,x].decode() == 'G':
                    ax.text(x, y, str(env_desc[y,x].decode()),
                       ha="center", va="center", color="w",
                         size=20, fontweight='bold')

                else:
                    ax.text(x, y, str(env_desc[y,x].decode()),
                       ha="center", va="center", color="g",
                         size=15, fontweight='bold')

                # No arrow for cells with G and H labels
                if env_desc[y,x].decode() != 'G' and env_desc[y,x].decode() != 'H':
                    ax.arrow(x, y, dx, dy, color='r', head_width=0.2, head_length=0.15)

        ax.set_title("Step = "  + str(k + 1), fontsize=20)

    fig.tight_layout()
    plt.show()

def show_Q_function_progress(env_desc, V_all, pi_all):
    """Defined in :numref:`sec_utils`"""
    # This function visualizes how value and policy changes over time.
    # V: [num_iters, num_states]
    # pi: [num_iters, num_states]

    # We want to only shows few values
    num_iters_all = V_all.shape[0]
    num_iters = num_iters_all // 10

    vis_indx = np.arange(0, num_iters_all, num_iters).tolist()
    vis_indx.append(num_iters_all - 1)
    V = np.zeros((len(vis_indx), V_all.shape[1]))
    pi = np.zeros((len(vis_indx), V_all.shape[1]))

    for c, i in enumerate(vis_indx):
        V[c]  = V_all[i]
        pi[c] = pi_all[i]

    num_iters = V.shape[0]
    fig, ax = plt.subplots(figsize=(15, 15))

    for k in range(V.shape[0]):
        plt.subplot(4, 4, k + 1)
        plt.imshow(V[k].reshape(4,4), cmap="bone")
        ax = plt.gca()
        ax.set_xticks(np.arange(0, 5)-.5, minor=True)
        ax.set_yticks(np.arange(0, 5)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_xticks([])
        ax.set_yticks([])

        # LEFT action: 0, DOWN action: 1
        # RIGHT action: 2, UP action: 3
        action2dxdy = {0:(-.25, 0),1:(0, .25),
                       2:(0.25, 0),3:(-.25, 0)}

        for y in range(4):
            for x in range(4):
                action = pi[k].reshape(4,4)[y, x]
                dx, dy = action2dxdy[action]

                if env_desc[y,x].decode() == 'H':
                    ax.text(x, y, str(env_desc[y,x].decode()),
                       ha="center", va="center", color="y",
                         size=20, fontweight='bold')

                elif env_desc[y,x].decode() == 'G':
                    ax.text(x, y, str(env_desc[y,x].decode()),
                       ha="center", va="center", color="w",
                         size=20, fontweight='bold')

                else:
                    ax.text(x, y, str(env_desc[y,x].decode()),
                       ha="center", va="center", color="g",
                         size=15, fontweight='bold')

                # No arrow for cells with G and H labels
                if env_desc[y,x].decode() != 'G' and env_desc[y,x].decode() != 'H':
                    ax.arrow(x, y, dx, dy, color='r', head_width=0.2, head_length=0.15)

        ax.set_title("Step = "  + str(vis_indx[k] + 1), fontsize=20)

    fig.tight_layout()
    plt.show()

def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_utils`"""
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)

def synthetic_data(w, b, num_examples):
    """Generate y = Xw + b + noise.

    Defined in :numref:`sec_utils`"""
    X = d2l.normal(0, 1, (num_examples, len(w)))
    y = d2l.matmul(X, w) + b
    y += d2l.normal(0, 0.01, y.shape)
    return X, d2l.reshape(y, (-1, 1))

def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent.

    Defined in :numref:`sec_utils`"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def get_dataloader_workers():
    """Use 4 processes to read the data.

    Defined in :numref:`sec_utils`"""
    return 4

def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory.

    Defined in :numref:`sec_utils`"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                        num_workers=get_dataloader_workers()),
            torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                        num_workers=get_dataloader_workers()))

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_utils`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6).

    Defined in :numref:`sec_utils`"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5): #接受五个参数，分别为imgs图像列表、num_rows图像显示的行数、num_cols图像显示的列数、titles每张图像的标题、scale图像的尺寸
    """绘制一列图像 Defined in :numref:`sec_utils`"""
    figsize = (num_cols * scale, num_rows * scale) #根据列数和行数以及缩放比例计算图像尺寸
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize) #使用d2l.plt.subplots创建一个包含多个子图的图像窗口，其中行数为num_rows，列数为num_cols，图像窗口的尺寸为figsize，axes为子图数组
    axes = axes.flatten() #将子图数组展平
    for i, (ax, img) in enumerate(zip(axes, imgs)): #将子图数组与图像列表配对，返回每一个子图ax、对应图像img和序号i
        try: #尝试将图像转换为numpy数组
            img = d2l.numpy(img)
        except: #如果失败，保持原样
            pass
        ax.imshow(img) #在子图上绘制图像
        ax.axes.get_xaxis().set_visible(False) #隐藏子图的x轴
        ax.axes.get_yaxis().set_visible(False) #隐藏子图的y轴
        if titles: #如果有标题
            ax.set_title(titles[i]) #设置子图标题
    return axes #返回子图数组

def linreg(X, w, b):
    """The linear regression model.

    Defined in :numref:`sec_utils`"""
    return d2l.matmul(X, w) + b

def squared_loss(y_hat, y):
    """Squared loss.

    Defined in :numref:`sec_utils`"""
    return (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2

def get_fashion_mnist_labels(labels):
    """Return text labels for the Fashion-MNIST dataset.

    Defined in :numref:`sec_utils`"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

class Animator:
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """Defined in :numref:`sec_utils`"""
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_utils`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    """Compute the number of correct predictions.

    Defined in :numref:`sec_utils`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))

def download(url, folder='../data', sha1_hash=None):
    """Download a file to folder and return the local filepath.

    Defined in :numref:`sec_utils`"""
    if not url.startswith('http'):
        # For back compatability
        url, sha1_hash = DATA_HUB[url]
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, url.split('/')[-1])
    # Check if hit cache
    if os.path.exists(fname) and sha1_hash:
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    # Download
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def extract(filename, folder=None):
    """Extract a zip/tar file into folder.

    Defined in :numref:`sec_utils`"""
    base_dir = os.path.dirname(filename)
    _, ext = os.path.splitext(filename)
    assert ext in ('.zip', '.tar', '.gz'), 'Only support zip/tar files.'
    if ext == '.zip':
        fp = zipfile.ZipFile(filename, 'r')
    else:
        fp = tarfile.open(filename, 'r')
    if folder is None:
        folder = base_dir
    fp.extractall(folder)

def download_extract(name, folder=None):
    """Download and extract a zip/tar file.

    Defined in :numref:`sec_utils`"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def tokenize(lines, token='word'):
    """Split text lines into word or character tokens.

    Defined in :numref:`sec_utils`"""
    assert token in ('word', 'char'), 'Unknown token type: ' + token
    return [line.split() if token == 'word' else list(line) for line in lines]

def evaluate_loss(net, data_iter, loss):
    """Evaluate the loss of a model on the given dataset.

    Defined in :numref:`sec_utils`"""
    metric = d2l.Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        out = net(X)
        y = d2l.reshape(y, out.shape)
        l = loss(out, y)
        metric.add(d2l.reduce_sum(l), d2l.size(l))
    return metric[0] / metric[1]

def grad_clipping(net, theta):
    """Clip the gradient.

    Defined in :numref:`sec_utils`"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

def read_data_nmt():
    """Load the English-French dataset.

    Defined in :numref:`sec_utils`"""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()

def preprocess_nmt(text):
    """Preprocess the English-French dataset.

    Defined in :numref:`sec_utils`"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # Replace non-breaking space with space, and convert uppercase letters to
    # lowercase ones
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Insert space between words and punctuation marks
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

def tokenize_nmt(text, num_examples=None):
    """Tokenize the English-French dataset.

    Defined in :numref:`sec_utils`"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences.

    Defined in :numref:`sec_utils`"""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad


def build_array_nmt(lines, vocab, num_steps):
    """Transform text sequences of machine translation into minibatches.

    Defined in :numref:`sec_utils`"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = d2l.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = d2l.reduce_sum(
        d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
    return array, valid_len


def load_data_nmt(batch_size, num_steps, num_examples=600):
    """Return the iterator and the vocabularies of the translation dataset.

    Defined in :numref:`sec_utils`"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.

    Defined in :numref:`sec_utils`"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """The softmax cross-entropy loss with masks.

    Defined in :numref:`sec_utils`"""
    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """Train a model for sequence to sequence.

    Defined in :numref:`sec_utils`"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = d2l.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # Make the loss scalar for `backward`
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """Predict for sequence to sequence.

    Defined in :numref:`sec_utils`"""
    # Set `net` to eval mode for inference
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as input
        # of the decoder at the next time step
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


# Alias defined in config.ini
nn_Module = nn.Module

ones_like = torch.ones_like
ones = torch.ones
zeros_like = torch.zeros_like
zeros = torch.zeros
tensor = torch.tensor
arange = torch.arange
meshgrid = torch.meshgrid
sin = torch.sin
sinh = torch.sinh
cos = torch.cos
cosh = torch.cosh
tanh = torch.tanh
linspace = torch.linspace
exp = torch.exp
log = torch.log
normal = torch.normal
rand = torch.rand
randn = torch.randn
matmul = torch.matmul
int32 = torch.int32
int64 = torch.int64
float32 = torch.float32
concat = torch.cat
stack = torch.stack
abs = torch.abs
eye = torch.eye
sigmoid = torch.sigmoid
batch_matmul = torch.bmm
numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)
reduce_mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs)
expand_dims = lambda x, *args, **kwargs: x.unsqueeze(*args, **kwargs)
swapaxes = lambda x, *args, **kwargs: x.swapaxes(*args, **kwargs)
repeat = lambda x, *args, **kwargs: x.repeat(*args, **kwargs)

