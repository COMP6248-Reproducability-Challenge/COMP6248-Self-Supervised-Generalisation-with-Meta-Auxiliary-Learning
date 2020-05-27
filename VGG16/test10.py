from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

def load_CIFAR_batch(filename):
  with open(filename, 'rb') as f:
    data_dict = pickle.load(f, encoding='latin1')
  return data_dict

# 实现数据集的完整读取
def load_CIFAR_data(data_dir):
  keys=['data','labels']
  train_dict = load_CIFAR_batch(os.path.join(data_dir, 'data_batch_1'))
#  print(train_dict.keys())
  for i in range(1,5):
      f = os.path.join(data_dir, 'data_batch_%d' % (i+1))
      print('loading', f)
      # 调用load_CIFAR_batch()获得批量的图像及其对应的标签
      image_batch_dict = load_CIFAR_batch(f)
      for k in keys:
        train_dict[k] = np.concatenate((train_dict[k], image_batch_dict[k]))
      del image_batch_dict
  print('finished loadding CIFAR-10 data')

  # 返回训练集的图像和标签，测试集的图像和标签
  return train_dict


class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    # url = "https://drive.google.com/drive/my-drive"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # R\read the data file
        if train:
            self.data_path = root+'/cifar-10-batches-py/'
            self.data_info = load_CIFAR_data(self.data_path)
        else:
            self.data_path = root+'/cifar-10-batches-py/test_batch'
            with open(self.data_path, 'rb') as fo:
              self.data_info = pickle.load(fo, encoding='latin1')
            fo.close()
        # with open(self.data_path, 'rb') as fo:
        #     self.data_info = pickle.load(fo, encoding='latin1')
        # fo.close()
        print(self.data_info.keys())
        # calculate data length
        self.data_len = len(self.data_info['data'])
        print(self.data_len)

        # first column contains the image paths
        self.image_arr = self.data_info['data'].reshape([self.data_len, 3, 32, 32])
        self.image_arr = self.image_arr.transpose((0, 2, 3, 1))  # convert to HWC

        # 10 Class, build dict from 20 class:
        class_10 = {0: 1, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10}
        # 3 Class, build dict from 10 class:
        class_3 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2}

        # second column is the labels
        self.labels   = self.data_info['labels']
#        self.label_coarse = self.data_info['coarse_labels']
        self.label_c10    = np.vectorize(class_10.get)(self.labels)
        self.label_c3     = np.vectorize(class_3.get)(self.label_c10)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __getitem__(self, index):
        # get image name from the pandas df
        single_image_name = self.image_arr[index]

        # open image
        img_as_img = Image.fromarray(single_image_name)

        # transform image to tensor
        if self.transform is not None:
            img_as_img = self.transform(img_as_img)
        else:
            img_as_img = self.to_tensor(img_as_img)

        # get label(class) of the image (from coarse to fine)
        label = np.zeros(4)
        label[-1] = self.labels[index]
        # label[-2] = self.label_coarse[index]
        label[-2] = self.label_c10[index]
        label[-3] = self.label_c3[index]

        return img_as_img, label

    def __len__(self):
        return self.data_len

from collections import OrderedDict
# from create_dataset import *

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F


class LabelGenerator(nn.Module):
    def __init__(self, psi):
        super(LabelGenerator, self).__init__()
        """
            label-generation network:
            takes the input and generates auxiliary labels with masked softmax for an auxiliary task.
        """
        filter = [64, 128, 256, 512, 512]
        self.class_nb = psi

        # define convolution block in VGG-16
        self.block1 = self.conv_layer(3, filter[0], 1)
        self.block2 = self.conv_layer(filter[0], filter[1], 2)
        self.block3 = self.conv_layer(filter[1], filter[2], 3)
        self.block4 = self.conv_layer(filter[2], filter[3], 4)
        self.block5 = self.conv_layer(filter[3], filter[4], 5)

        # define fc-layers in VGG-16 (output auxiliary classes \sum_i\psi[i])
        self.classifier = nn.Sequential(
            nn.Linear(filter[-1], filter[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(filter[-1], int(np.sum(self.class_nb))),
        )

        # apply weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def conv_layer(self, in_channel, out_channel, index):
        if index < 3:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        return conv_block

    # define masked softmax
    def mask_softmax(self, x, mask, dim=1):
        logits = torch.exp(x) * mask / torch.sum(torch.exp(x) * mask, dim=dim, keepdim=True)
        return logits

    def forward(self, x, y):
        g_block1 = self.block1(x)
        g_block2 = self.block2(g_block1)
        g_block3 = self.block3(g_block2)
        g_block4 = self.block4(g_block3)
        g_block5 = self.block5(g_block4)

        # build a binary mask by psi, we add epsilon=1e-8 to avoid nans
        index = torch.zeros([len(self.class_nb), np.sum(self.class_nb)]) + 1e-8
        for i in range(len(self.class_nb)):
            index[i, int(np.sum(self.class_nb[:i])):np.sum(self.class_nb[:i+1])] = 1
        mask = index[y].to(device)

        predict = self.classifier(g_block5.view(g_block5.size(0), -1))
        label_pred = self.mask_softmax(predict, mask, dim=1)

        return label_pred


class VGG16(nn.Module):
    def __init__(self, psi):
        super(VGG16, self).__init__()
        """
            multi-task network:
            takes the input and predicts primary and auxiliary labels (same network structure as in human)
        """
        filter = [64, 128, 256, 512, 512]

        # define convolution block in VGG-16
        self.block1 = self.conv_layer(3, filter[0], 1)
        self.block2 = self.conv_layer(filter[0], filter[1], 2)
        self.block3 = self.conv_layer(filter[1], filter[2], 3)
        self.block4 = self.conv_layer(filter[2], filter[3], 4)
        self.block5 = self.conv_layer(filter[3], filter[4], 5)

        # primary task prediction
        self.classifier1 = nn.Sequential(
            nn.Linear(filter[-1], filter[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(filter[-1], len(psi)),
            nn.Softmax(dim=1)
        )

        # auxiliary task prediction
        self.classifier2 = nn.Sequential(
            nn.Linear(filter[-1], filter[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(filter[-1], int(np.sum(psi))),
            nn.Softmax(dim=1)
        )

        # apply weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def conv_layer(self, in_channel, out_channel, index):
        if index < 3:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
        return conv_block

    # define forward conv-layer (will be used in second-derivative step)
    def conv_layer_ff(self, input, weights, index):
        if index < 3:
            net = F.conv2d(input, weights['block{:d}.0.weight'.format(index)], weights['block{:d}.0.bias'.format(index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device), torch.ones(net.data.size()[1]).to(device),
                               weights['block{:d}.1.weight'.format(index)], weights['block{:d}.1.bias'.format(index)],
                               training=True)
            net = F.relu(net, inplace=True)
            net = F.conv2d(net, weights['block{:d}.3.weight'.format(index)], weights['block{:d}.3.bias'.format(index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device), torch.ones(net.data.size()[1]).to(device),
                               weights['block{:d}.4.weight'.format(index)], weights['block{:d}.4.bias'.format(index)],
                               training=True)
            net = F.relu(net, inplace=True)
            net = F.max_pool2d(net, kernel_size=2, stride=2, )
        else:
            net = F.conv2d(input, weights['block{:d}.0.weight'.format(index)], weights['block{:d}.0.bias'.format(index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device), torch.ones(net.data.size()[1]).to(device),
                               weights['block{:d}.1.weight'.format(index)], weights['block{:d}.1.bias'.format(index)],
                               training=True)
            net = F.relu(net, inplace=True)
            net = F.conv2d(net, weights['block{:d}.3.weight'.format(index)], weights['block{:d}.3.bias'.format(index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device), torch.ones(net.data.size()[1]).to(device),
                               weights['block{:d}.4.weight'.format(index)], weights['block{:d}.4.bias'.format(index)],
                               training=True)
            net = F.relu(net, inplace=True)
            net = F.conv2d(net, weights['block{:d}.6.weight'.format(index)], weights['block{:d}.6.bias'.format(index)], padding=1)
            net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device), torch.ones(net.data.size()[1]).to(device),
                               weights['block{:d}.7.weight'.format(index)], weights['block{:d}.7.bias'.format(index)],
                               training=True)
            net = F.relu(net, inplace=True)
            net = F.max_pool2d(net, kernel_size=2, stride=2)
        return net

    # define forward fc-layer (will be used in second-derivative step)
    def dense_layer_ff(self, input, weights, index):
        net = F.linear(input, weights['classifier{:d}.0.weight'.format(index)], weights['classifier{:d}.0.bias'.format(index)])
        net = F.relu(net, inplace=True)
        net = F.linear(net, weights['classifier{:d}.2.weight'.format(index)], weights['classifier{:d}.2.bias'.format(index)])
        net = F.softmax(net, dim=1)
        return net

    def forward(self, x, weights=None):
        """
            if no weights given, use the direct training strategy and update network paramters
            else retain the computational graph which will be used in second-derivative step
        """
        if weights is None:
            g_block1 = self.block1(x)
            g_block2 = self.block2(g_block1)
            g_block3 = self.block3(g_block2)
            g_block4 = self.block4(g_block3)
            g_block5 = self.block5(g_block4)

            t1_pred = self.classifier1(g_block5.view(g_block5.size(0), -1))
            t2_pred = self.classifier2(g_block5.view(g_block5.size(0), -1))

        else:
            g_block1 = self.conv_layer_ff(x, weights, 1)
            g_block2 = self.conv_layer_ff(g_block1, weights, 2)
            g_block3 = self.conv_layer_ff(g_block2, weights, 3)
            g_block4 = self.conv_layer_ff(g_block3, weights, 4)
            g_block5 = self.conv_layer_ff(g_block4, weights, 5)

            t1_pred = self.dense_layer_ff(g_block5.view(g_block5.size(0), -1), weights, 1)
            t2_pred = self.dense_layer_ff(g_block5.view(g_block5.size(0), -1), weights, 2)

        return t1_pred, t2_pred

    def model_fit(self, x_pred, x_output, pri=True, num_output=10):
        if not pri:
            # generated auxiliary label is a soft-assignment vector (no need to change into one-hot vector)
            x_output_onehot = x_output
        else:
            # convert a single label into a one-hot vector
            x_output_onehot = torch.zeros((len(x_output), num_output)).to(device)
            x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)
            # a= torch.zeros((len(x_output), 10))
            # print('a',a.shape)
            # print('x_output_onehot',x_output_onehot.shape)
            # print('x_output',x_output.shape)
            # print(len(x_output))
        # print('1 - x_pred',(1 - x_pred).shape)
        # print((x_pred + 1e-20).shape)
        # print('x_pred',x_pred.shape)
        # apply focal loss
        loss = x_output_onehot * (1 - x_pred)**2 * torch.log(x_pred + 1e-20)
        return torch.sum(-loss, dim=1)

    def model_entropy(self, x_pred1):
        # compute entropy loss
        x_pred1 = torch.mean(x_pred1, dim=0)
        loss1 = x_pred1 * torch.log(x_pred1 + 1e-20)
        return torch.sum(loss1)


# load CIFAR10 dataset
trans_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),

])
trans_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),

])

# load CIFAR-10 dataset with batch-size 100
# set keyword download=True at the first time to download the dataset
cifar10_train_set = CIFAR10(root='./img_data', train=True, transform=trans_train, download=True)
cifar10_test_set = CIFAR10(root='./img_data', train=False, transform=trans_test, download=True)

batch_size = 100
kwargs = {'num_workers': 1, 'pin_memory': True}
cifar10_train_loader = torch.utils.data.DataLoader(
    dataset=cifar10_train_set,
    batch_size=batch_size,
    shuffle=True)

cifar10_test_loader = torch.utils.data.DataLoader(
    dataset=cifar10_test_set,
    batch_size=batch_size,
    shuffle=True)

# define label-generation model,
# and optimiser with learning rate 1e-3, drop half for every 50 epochs, weight_decay=5e-4,
psi = [3]*10  # for each primary class split into 3 auxiliary classes, with total 30 auxiliary classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LabelGenerator = LabelGenerator(psi=psi).to(device)
gen_optimizer = optim.SGD(LabelGenerator.parameters(), lr=1e-3, weight_decay=5e-4)
gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size=50, gamma=0.5)

# define parameters
total_epoch = 200
train_batch = len(cifar10_train_loader)
test_batch = len(cifar10_test_loader)

# define multi-task network, and optimiser with learning rate 0.01, drop half for every 50 epochs
VGG16_model = VGG16(psi=psi).to(device)
optimizer = optim.SGD(VGG16_model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
avg_cost = np.zeros([total_epoch, 9], dtype=np.float32)
vgg_lr = 0.01  # define learning rate for second-derivative step (theta_1^+)
k = 0
trainloss=[]
testloss=[]
trainaccuracy=[]
testaccuracy=[]
for index in range(total_epoch):
    cost = np.zeros(4, dtype=np.float32)

    # drop the learning rate with the same strategy in the multi-task network
    # note: not necessary to be consistent with the multi-task network's parameter,
    # it can also be learned directly from the network
    if (index + 1) % 50 == 0:
       vgg_lr = vgg_lr * 0.5

    scheduler.step()
    gen_scheduler.step()

    # evaluate training data (training-step, update on theta_1)
    VGG16_model.train()
    cifar10_train_dataset = iter(cifar10_train_loader)
    for i in range(train_batch):
        train_data, train_label = cifar10_train_dataset.next()
        train_label = train_label.type(torch.LongTensor)
        train_data, train_label = train_data.to(device), train_label.to(device)
        train_pred1, train_pred2 = VGG16_model(train_data)
        train_pred3 = LabelGenerator(train_data, train_label[:, 2])  # generate auxiliary labels
        # print('train_pred1',train_pred1.shape)
        # print(train_pred2.shape)
        # print(train_pred3.shape)
        # print('train_label',train_label)
        # reset optimizers with zero gradient
        optimizer.zero_grad()
        gen_optimizer.zero_grad()

        # choose level 2/3 hierarchy, 10-class (gt) / 30-class classification (generated by labelgeneartor)
        train_loss1 = VGG16_model.model_fit(train_pred1, train_label[:, 2], pri=True, num_output=10)
        train_loss2 = VGG16_model.model_fit(train_pred2, train_pred3, pri=False, num_output=30)
        train_loss3 = VGG16_model.model_entropy(train_pred3)

        # compute cosine similarity between gradients from primary and auxiliary loss
        grads1 = torch.autograd.grad(torch.mean(train_loss1), VGG16_model.parameters(), retain_graph=True, allow_unused=True)
        grads2 = torch.autograd.grad(torch.mean(train_loss2), VGG16_model.parameters(), retain_graph=True, allow_unused=True)
        cos_mean = 0
        for k in range(len(grads1) - 8):  # only compute on shared representation (ignore task-specific fc-layers)
            cos_mean += torch.mean(F.cosine_similarity(grads1[k], grads2[k], dim=0)) / (len(grads1) - 8)
        # cosine similarity evaluation ends here

        train_loss = torch.mean(train_loss1) + torch.mean(train_loss2)
        train_loss.backward()

        optimizer.step()

        train_predict_label1 = train_pred1.data.max(1)[1]
        train_acc1 = train_predict_label1.eq(train_label[:, 2]).sum().item() / batch_size

        cost[0] = torch.mean(train_loss1).item()
        cost[1] = train_acc1
        cost[2] = cos_mean
        k = k + 1
        avg_cost[index][0:3] += cost[0:3] / train_batch

    # evaluating training data (meta-training step, update on theta_2)
    cifar10_train_dataset = iter(cifar10_train_loader)
    for i in range(train_batch):
        train_data, train_label = cifar10_train_dataset.next()
        train_label = train_label.type(torch.LongTensor)
        train_data, train_label = train_data.to(device), train_label.to(device)
        train_pred1, train_pred2 = VGG16_model(train_data)
        train_pred3 = LabelGenerator(train_data, train_label[:, 2])

        # reset optimizer with zero gradient
        optimizer.zero_grad()
        gen_optimizer.zero_grad()

        # choose level 2/3 hierarchy, 10-class/30-class classification
        train_loss1 = VGG16_model.model_fit(train_pred1, train_label[:, 2], pri=True, num_output=10)
        train_loss2 = VGG16_model.model_fit(train_pred2, train_pred3, pri=False, num_output=30)
        train_loss3 = VGG16_model.model_entropy(train_pred3)

        # multi-task loss
        train_loss = torch.mean(train_loss1) + torch.mean(train_loss2)

        # current accuracy on primary task
        train_predict_label1 = train_pred1.data.max(1)[1]
        train_acc1 = train_predict_label1.eq(train_label[:, 2]).sum().item() / batch_size
        cost[0] = torch.mean(train_loss1).item()
        cost[1] = train_acc1

        # current theta_1
        fast_weights = OrderedDict((name, param) for (name, param) in VGG16_model.named_parameters())

        # create_graph flag for computing second-derivative
        grads = torch.autograd.grad(train_loss, VGG16_model.parameters(), create_graph=True)
        data = [p.data for p in list(VGG16_model.parameters())]

        # compute theta_1^+ by applying sgd on multi-task loss
        fast_weights = OrderedDict((name, param - vgg_lr * grad) for ((name, param), grad, data) in zip(fast_weights.items(), grads, data))

        # compute primary loss with the updated thetat_1^+
        train_pred1, train_pred2 = VGG16_model.forward(train_data, fast_weights)
        train_loss1 = VGG16_model.model_fit(train_pred1, train_label[:, 2], pri=True, num_output=10)

        # update theta_2 with primary loss + entropy loss
        (torch.mean(train_loss1) + 0.2*torch.mean(train_loss3)).backward()
        gen_optimizer.step()

        train_predict_label1 = train_pred1.data.max(1)[1]
        train_acc1 = train_predict_label1.eq(train_label[:, 2]).sum().item() / batch_size

        # accuracy on primary task after one update
        cost[2] = torch.mean(train_loss1).item()
        cost[3] = train_acc1
        avg_cost[index][3:7] += cost[0:4] / train_batch

    # evaluate on test data
    VGG16_model.eval()
    
    with torch.no_grad():
        cifar10_test_dataset = iter(cifar10_test_loader)
        for i in range(test_batch):
            test_data, test_label = cifar10_test_dataset.next()
            test_label = test_label.type(torch.LongTensor)
            test_data, test_label = test_data.to(device), test_label.to(device)
            test_pred1, test_pred2 = VGG16_model(test_data)

            test_loss1 = VGG16_model.model_fit(test_pred1, test_label[:, 2], pri=True, num_output=10)

            test_predict_label1 = test_pred1.data.max(1)[1]
            test_acc1 = test_predict_label1.eq(test_label[:, 2]).sum().item() / batch_size

            cost[0] = torch.mean(test_loss1).item()
            cost[1] = test_acc1

            avg_cost[index][7:] += cost[0:2] / test_batch
    
    torch.save(VGG16_model.state_dict(), "./model10")       
    trainloss.append(avg_cost[index][0])
    trainaccuracy.append(avg_cost[index][1])
    testloss.append(avg_cost[index][7])
    testaccuracy.append(avg_cost[index][8])
    
   

    print('EPOCH: {:04d} Iter {:04d} | TRAIN [LOSS|ACC.]: PRI {:.4f} {:.4f} COSSIM {:.4f} || '
          'META [LOSS|ACC.]: PRE {:.4f} {:.4f} AFTER {:.4f} {:.4f} || TEST: {:.4f} {:.4f}'
          .format(index, k, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2], avg_cost[index][3],
                  avg_cost[index][4], avg_cost[index][5], avg_cost[index][6], avg_cost[index][7], avg_cost[index][8]))
print(trainloss)
print(trainaccuracy)
print(testloss)
print(testaccuracy)
