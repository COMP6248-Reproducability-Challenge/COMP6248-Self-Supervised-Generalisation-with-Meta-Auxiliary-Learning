from collections import OrderedDict

import argparse
import torchvision
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

"""
This program is used to continue training 
network from the precious stored 3-primary model if the training is interrupted.

The ResNet-32 model is defined and written by Enze Pan originally. The labelgenerator function is from the author of the paper
and some modifications are made to fit the data and model.

The training framework codes are from the paper author, modifications are made to fit the ResNet model.
"""


def ClassGenerator(label):
    class_3 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2}
    label_c3 = np.vectorize(class_3.get)(label)
    label_c3 = torch.tensor(label_c3, dtype=torch.int64)
    target = torch.cat((label_c3.view(label_c3.shape[0], -1), label.view(label.shape[0], -1)), 1)
    return target

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
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 5, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 5, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 4, stride=2)


        self.classifier = nn.Sequential(
            nn.Linear(filter[-3], filter[-4]),
            nn.ReLU(inplace=True),
            nn.Linear(filter[-4],filter[-5]),   #128->64
            nn.ReLU(inplace=True),
            nn.Linear(filter[-5], int(np.sum(self.class_nb))),
        )

        
        # apply weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    # define masked softmax
    def mask_softmax(self, x, mask, dim=1):
        logits = torch.exp(x) * mask / torch.sum(torch.exp(x) * mask, dim=dim, keepdim=True)
        return logits

    def forward(self, x, y):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)


        # build a binary mask by psi, we add epsilon=1e-8 to avoid nans
        index = torch.zeros([len(self.class_nb), np.sum(self.class_nb)]) + 1e-8
        for i in range(len(self.class_nb)):
            index[i, int(np.sum(self.class_nb[:i])):np.sum(self.class_nb[:i+1])] = 1
        mask = index[y].to(device)

        predict = self.classifier(out.view(out.size(0), -1))
        label_pred = self.mask_softmax(predict, mask, dim=1)

        return label_pred

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,ResidualBlock, psi):
        super(ResNet, self).__init__()
        """
            multi-task network:
            takes the input and predicts primary and auxiliary labels (same network structure as in human)
        """
        filter = [64, 128, 256, 512, 512]
        net123=torch.zeros(5)
        self.lay1_out_net=F.relu(net123, inplace=True)
        self.lay2_out_net=F.relu(net123, inplace=True)
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 5, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 5, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 4, stride=2)

        # primary task prediction
        # modification: change the classifier's layer number
        self.classifier1 = nn.Sequential(
            nn.Linear(filter[-3], filter[-4]),  #256->128
            nn.ReLU(inplace=True),
            nn.Linear(filter[-4],filter[-5]),   #128->64
            nn.ReLU(inplace=True),
            nn.Linear(filter[-5], len(psi)),    #64-> primiary task num
            nn.Softmax(dim=1)
        )

        # auxiliary task prediction
        self.classifier2 = nn.Sequential(
            nn.Linear(filter[-3], filter[-4]),  #256->128
            nn.ReLU(inplace=True),
            nn.Linear(filter[-4],filter[-5]),   #128->64
            nn.ReLU(inplace=True),
            nn.Linear(filter[-5], int(np.sum(psi))),
            nn.Softmax(dim=1)
        )

        # apply weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)


    # define forward conv-layer (will be used in second-derivative step)

    def conv1_layer_ff(self,input,weights,index):
            net = F.conv2d(input, weights['conv1.0.weight'.format(index)], stride=1, padding=1)
            net=F.batch_norm(net,torch.zeros(net.data.size()[1]).to(device),torch.ones(net.data.size()[1]).to(device),
                             weights['conv1.1.weight'.format(index)], weights['conv1.1.bias'.format(index)],
                             training=True)
            net=F.relu(net, inplace=True)
            return net


    def res_layer_ff(self, input, weights, index):
        num_blocks=[0,5,5,4]
        stride=[0,1,2,2]
        strides = [stride[index]] + [1] * (num_blocks[1] - 1)  # strides=[1,1]
        if index==1:
            for stride_num in strides:  #1 1 1 1 1
                counter=0
                net = F.conv2d(input, weights['layer{layers}.{convnum}.left.{finepara}.weight'.format(layers=index,
                    convnum=counter,finepara=0)], stride=stride_num, padding=1)

                net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device),torch.ones(net.data.size()[1]).to(device),
                                   weights['layer{layers}.{convnum}.left.{finepara}.weight'.format(layers=index,convnum=counter,finepara=1)],
                                   weights['layer{layers}.{convnum}.left.{finepara}.bias'.format(layers=index,convnum=counter,finepara=1)],training=True)

                net = F.relu(net, inplace=True)

                net = F.conv2d(net, weights['layer{layers}.{convnum}.left.{finepara}.weight'.format(layers=index,
                    convnum=counter,finepara=3)], stride=1, padding=1)

                net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device),
                                   torch.ones(net.data.size()[1]).to(device),
                                   weights['layer{layers}.{convnum}.left.{finepara}.weight'.format(layers=index,convnum=counter,finepara=1)],
                                   weights['layer{layers}.{convnum}.left.{finepara}.bias'.format(layers=index,convnum=counter,finepara=4)],training=True)
                
                counter+=1

        else:
            for stride_num in strides:  #2 1 1 1 1
                counter=0
                if stride_num ==2:
                    net = F.conv2d(input, weights['layer{layers}.{convnum}.left.{finepara}.weight'.format(layers=index,
                                                                                    convnum=counter,finepara=0)],stride=stride_num, padding=1)
                    net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device),torch.ones(net.data.size()[1]).to(device),
                                       weights['layer{layers}.{convnum}.left.{finepara}.weight'.format(layers=index,
                                                                                    convnum=counter,finepara=1)],
                                       weights['layer{layers}.{convnum}.left.{finepara}.bias'.format(layers=index,
                                                                                    convnum=counter,finepara=1)],training=True)
                    net = F.relu(net, inplace=True)
                    net = F.conv2d(net, weights['layer{layers}.{convnum}.left.{finepara}.weight'.format(layers=index,
                                                                                    convnum=counter,finepara=3)],stride=1, padding=1)

                    net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device),
                                       torch.ones(net.data.size()[1]).to(device),
                                       weights['layer{layers}.{convnum}.left.{finepara}.weight'.format(layers=index,
                                                                                                       convnum=counter,
                                                                                                       finepara=1)],
                                       weights['layer{layers}.{convnum}.left.{finepara}.bias'.format(layers=index,
                                                                                                     convnum=counter,
                                                                                                     finepara=4)],training=True)
                    #####shoutcut####

                    if index ==2:                                              
                        net = F.conv2d(self.lay1_out_net,weights['layer{layers}.{convnum}.shortcut.{finepara}.weight'.format(layers=index,
                                                                                            convnum=counter,finepara=0)],stride=stride_num)

                        net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device),torch.ones(net.data.size()[1]).to(device),
                                           weights['layer{layers}.{convnum}.shortcut.{finepara}.weight'.format(layers=index,
                                                                                            convnum=counter,finepara=1)],
                                           weights['layer{layers}.{convnum}.shortcut.{finepara}.bias'.format(layers=index,
                                                                                            convnum=counter,finepara=1)],training=True)
                    else:
                        net = F.conv2d(self.lay2_out_net,weights['layer{layers}.{convnum}.shortcut.{finepara}.weight'.format(layers=index,
                                                                                            convnum=counter,finepara=0)],stride=stride_num)

                        net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device),torch.ones(net.data.size()[1]).to(device),
                                           weights['layer{layers}.{convnum}.shortcut.{finepara}.weight'.format(layers=index,
                                                                                            convnum=counter,finepara=1)],
                                           weights['layer{layers}.{convnum}.shortcut.{finepara}.bias'.format(layers=index,
                                                                                            convnum=counter,finepara=1)],training=True)
                    

                else:
                    net = F.conv2d(input, weights['layer{layers}.{convnum}.left.{finepara}.weight'.format(layers=index,
                                                                                        convnum=counter,finepara=0)],stride=stride_num, padding=1)
                    net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device),
                                       torch.ones(net.data.size()[1]).to(device),
                                       weights['layer{layers}.{convnum}.left.{finepara}.weight'.format(layers=index,
                                                                                        convnum=stride_num,finepara=1)],
                                       weights['layer{layers}.{convnum}.left.{finepara}.bias'.format(layers=index,
                                                                                        convnum=stride_num,finepara=1)],training=True)
                    net = F.relu(net, inplace=True)
                    net = F.conv2d(net, weights['layer{layers}.{convnum}.left.{finepara}.weight'.format(layers=index,
                                                                                        convnum=stride_num,finepara=3)],stride=1, padding=1)

                    net = F.batch_norm(net, torch.zeros(net.data.size()[1]).to(device),torch.ones(net.data.size()[1]).to(device),
                                       weights['layer{layers}.{convnum}.left.{finepara}.weight'.format(layers=index,
                                                                                        convnum=stride_num,finepara=1)],
                                       weights['layer{layers}.{convnum}.left.{finepara}.bias'.format(layers=index,
                                                                 convnum=stride_num,finepara=4)],training=True)
                counter += 1
      

        if index==1:
            self.lay1_out_net=net
        elif index ==2:
            self.lay2_out_net=net
        return net

    # define forward fc-layer (will be used in second-derivative step)
    def dense_layer_ff(self, input, weights, index):
        net = F.linear(input, weights['classifier{:d}.0.weight'.format(index)], weights['classifier{:d}.0.bias'.format(index)])
        net = F.relu(net, inplace=True)
        net = F.linear(net, weights['classifier{:d}.2.weight'.format(index)], weights['classifier{:d}.2.bias'.format(index)])
        net = F.relu(net, inplace=True)
        net = F.linear(net, weights['classifier{:d}.4.weight'.format(index)], weights['classifier{:d}.4.bias'.format(index)])
        net = F.softmax(net, dim=1)
        return net

    def forward(self, x, weights=None):
        """
            if no weights given, use the direct training strategy and update network paramters
            else retain the computational graph which will be used in second-derivative step
        """
        if weights is None:
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.avg_pool2d(out, out.size()[3])
            out = out.view(out.size(0), -1)
            t1_pred = self.classifier1(out.view(out.size(0), -1))
            t2_pred = self.classifier2(out.view(out.size(0), -1))

        else:
            out = self.conv1_layer_ff(x, weights, 1)
            out = self.res_layer_ff(out, weights, 1)
            out = self.res_layer_ff(out, weights, 2)
            out = self.res_layer_ff(out, weights, 3)
            out = F.avg_pool2d(out, out.size()[3])
            out = out.view(out.size(0), -1)

            t1_pred = self.dense_layer_ff(out.view(out.size(0), -1), weights, 1)
            t2_pred = self.dense_layer_ff(out.view(out.size(0), -1), weights, 2)

        return t1_pred, t2_pred

    def model_fit(self, x_pred, x_output, pri=True, num_output=3):
        if not pri:
            # generated auxiliary label is a soft-assignment vector (no need to change into one-hot vector)
            x_output_onehot = x_output
        else:
            # convert a single label into a one-hot vector
            x_output_onehot = torch.zeros((len(x_output), num_output)).to(device)
            x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)

        # apply focal loss
        loss = x_output_onehot * (1 - x_pred)**2 * torch.log(x_pred + 1e-20)
        return torch.sum(-loss, dim=1)

    def model_entropy(self, x_pred1):
        # compute entropy loss
        x_pred1 = torch.mean(x_pred1, dim=0)
        loss1 = x_pred1 * torch.log(x_pred1 + 1e-20)
        return torch.sum(loss1)


def ResNet32(psi):

    return ResNet(ResidualBlock,psi)


# load CINIC10 dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch ResNet32 CINIC10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')
args = parser.parse_args()

pre_epoch = 14


cinic_directory = './dataset/cinic10'
cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]

cinic_train = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(cinic_directory + '/train',
        transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
    batch_size=128, shuffle=True)

cinic_test = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(cinic_directory + '/test',
        transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
    batch_size=128, shuffle=True)

cinic_valid = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(cinic_directory + '/valid',
        transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
    batch_size=128, shuffle=True)

batch_size = 100
kwargs = {'num_workers': 1, 'pin_memory': True}
print("Data Loaded...")
# define label-generation model,
# and optimiser with learning rate 1e-3, drop half for every 10 epochs, weight_decay=5e-4,
psi = [3]*10  # for each primary class split into 5 auxiliary classes, with total 100 auxiliary classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LabelGenerator = LabelGenerator(psi=psi).to(device)
gen_optimizer = optim.SGD(LabelGenerator.parameters(), lr=1e-3, weight_decay=5e-4)
gen_scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size=50, gamma=0.5)

# define parameters
total_epoch = 30
train_batch = len(cinic_train)
test_batch = len(cinic_test)

# define multi-task network, and optimiser with learning rate 0.01, drop half for every 50 epochs
Res_model = ResNet32(psi=psi).to(device)
#Load stored models
pre=torch.load(r'net_014.pth')
Res_model.load_state_dict(pre)
optimizer = optim.SGD(Res_model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
avg_cost = np.zeros([total_epoch, 9], dtype=np.float32)
vgg_lr = 0.01  # define learning rate for second-derivative step (theta_1^+)
k = 0
print("Begin training...")
with open("log.txt", "w") as f:
    for index in range(pre_epoch,total_epoch):
        cost = np.zeros(4, dtype=np.float32)

        # drop the learning rate with the same strategy in the multi-task network
        # note: not necessary to be consistent with the multi-task network's parameter,
        # it can also be learned directly from the network
        if (index + 1) % 10 == 0:
           vgg_lr = vgg_lr * 0.5

        scheduler.step()
        gen_scheduler.step()

        # evaluate training data (training-step, update on theta_1)
        Res_model.train()
        cinic_train_dataset = iter(cinic_train)
        for i in range(train_batch):
            train_data, train_label = cinic_train_dataset.next()
            train_label = ClassGenerator(train_label)
            train_label = train_label.type(torch.LongTensor)
            train_data, train_label = train_data.to(device), train_label.to(device)
            train_pred1, train_pred2 = Res_model(train_data)
            train_pred3 = LabelGenerator(train_data, train_label[:, 1])  # generate auxiliary labels

            # reset optimizers with zero gradient
            optimizer.zero_grad()
            gen_optimizer.zero_grad()

            # choose level 2/3 hierarchy, 20-class (gt) / 100-class classification (generated by labelgeneartor)
            train_loss1 = Res_model.model_fit(train_pred1, train_label[:, 1], pri=True, num_output=10)
            train_loss2 = Res_model.model_fit(train_pred2, train_pred3, pri=False, num_output=30)
            train_loss3 = Res_model.model_entropy(train_pred3)

            # compute cosine similarity between gradients from primary and auxiliary loss
            grads1 = torch.autograd.grad(torch.mean(train_loss1), Res_model.parameters(), retain_graph=True, allow_unused=True)
            grads2 = torch.autograd.grad(torch.mean(train_loss2), Res_model.parameters(), retain_graph=True, allow_unused=True)
            cos_mean = 0
            for k in range(len(grads1) - 12):  # only compute on shared representation (ignore task-specific fc-layers)
                cos_mean += torch.mean(F.cosine_similarity(grads1[k], grads2[k], dim=0)) / (len(grads1) - 12)
            # cosine similarity evaluation ends here

            train_loss = torch.mean(train_loss1) + torch.mean(train_loss2)
            train_loss.backward()

            optimizer.step()

            train_predict_label1 = train_pred1.data.max(1)[1]
            train_acc1 = train_predict_label1.eq(train_label[:, 1]).sum().item() / batch_size

            cost[0] = torch.mean(train_loss1).item()
            cost[1] = train_acc1
            cost[2] = cos_mean
            k = k + 1
            avg_cost[index][0:3] += cost[0:3] / train_batch

        # evaluating training data (meta-training step, update on theta_2)
        cinic_train_dataset = iter(cinic_train)
        for i in range(train_batch):
            train_data, train_label = cinic_train_dataset.next()
            train_label = ClassGenerator(train_label)
            train_label = train_label.type(torch.LongTensor)
            train_data, train_label = train_data.to(device), train_label.to(device)
            train_pred1, train_pred2 = Res_model(train_data)
            train_pred3 = LabelGenerator(train_data, train_label[:, 1])

            # reset optimizer with zero gradient
            optimizer.zero_grad()
            gen_optimizer.zero_grad()

            # choose level 2/3 hierarchy, 20-class/100-class classification
            train_loss1 = Res_model.model_fit(train_pred1, train_label[:, 1], pri=True, num_output=10)
            train_loss2 = Res_model.model_fit(train_pred2, train_pred3, pri=False, num_output=30)
            train_loss3 = Res_model.model_entropy(train_pred3)

            # multi-task loss
            train_loss = torch.mean(train_loss1) + torch.mean(train_loss2)

            # current accuracy on primary task
            train_predict_label1 = train_pred1.data.max(1)[1]
            train_acc1 = train_predict_label1.eq(train_label[:, 1]).sum().item() / batch_size
            cost[0] = torch.mean(train_loss1).item()
            cost[1] = train_acc1

            # current theta_1
            fast_weights = OrderedDict((name, param) for (name, param) in Res_model.named_parameters())

            # create_graph flag for computing second-derivative
            grads = torch.autograd.grad(train_loss, Res_model.parameters(), create_graph=True)
            data = [p.data for p in list(Res_model.parameters())]

            # compute theta_1^+ by applying sgd on multi-task loss
            fast_weights = OrderedDict((name, param - vgg_lr * grad) for ((name, param), grad, data) in zip(fast_weights.items(), grads, data))

            # compute primary loss with the updated thetat_1^+
            train_pred1, train_pred2 = Res_model.forward(train_data, fast_weights)
            train_loss1 = Res_model.model_fit(train_pred1, train_label[:, 1], pri=True, num_output=10)

            # update theta_2 with primary loss + entropy loss
            (torch.mean(train_loss1) + 0.2*torch.mean(train_loss3)).backward()
            gen_optimizer.step()

            train_predict_label1 = train_pred1.data.max(1)[1]
            train_acc1 = train_predict_label1.eq(train_label[:, 1]).sum().item() / batch_size

            # accuracy on primary task after one update
            cost[2] = torch.mean(train_loss1).item()
            cost[3] = train_acc1
            avg_cost[index][3:7] += cost[0:4] / train_batch

        # evaluate on test data
        Res_model.eval()
        with torch.no_grad():
            cinic_test_dataset = iter(cinic_test)
            for i in range(test_batch):
                test_data, test_label = cinic_test_dataset.next()
                test_label = ClassGenerator(test_label)
                test_label = test_label.type(torch.LongTensor)
                test_data, test_label = test_data.to(device), test_label.to(device)
                test_pred1, test_pred2 = Res_model(test_data)

                test_loss1 = Res_model.model_fit(test_pred1, test_label[:, 1], pri=True, num_output=10)

                test_predict_label1 = test_pred1.data.max(1)[1]
                test_acc1 = test_predict_label1.eq(test_label[:, 1]).sum().item() / batch_size

                cost[0] = torch.mean(test_loss1).item()
                cost[1] = test_acc1

                avg_cost[index][7:] += cost[0:2] / test_batch

        torch.save(Res_model.state_dict(), '%s/net_%03d.pth' % (args.outf, index + 1))
        print('EPOCH: {:04d} Iter {:04d} | TRAIN [LOSS|ACC.]: PRI {:.4f} {:.4f} COSSIM {:.4f} || '
              'META [LOSS|ACC.]: PRE {:.4f} {:.4f} AFTER {:.4f} {:.4f} || TEST: {:.4f} {:.4f}'
              .format(index, k, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2], avg_cost[index][3],
                      avg_cost[index][4], avg_cost[index][5], avg_cost[index][6], avg_cost[index][7],
                      avg_cost[index][8]))
        f.write('EPOCH: {:04d} Iter {:04d} | TRAIN [LOSS|ACC.]: PRI {:.4f} {:.4f} COSSIM {:.4f} || '
              'META [LOSS|ACC.]: PRE {:.4f} {:.4f} AFTER {:.4f} {:.4f} || TEST: {:.4f} {:.4f}'
              .format(index, k, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2], avg_cost[index][3],
                  avg_cost[index][4], avg_cost[index][5], avg_cost[index][6], avg_cost[index][7], avg_cost[index][8]))
        f.write('\n')
        f.flush()

