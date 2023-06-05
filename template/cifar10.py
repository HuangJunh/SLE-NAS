"""
# !/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import load_dataset.data_loader as data_loader
import os
import numpy as np
import math
import copy
from datetime import datetime
import multiprocessing
from utils import Utils
from torchprofile import profile_macs
from template.drop import drop_path


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

    def __repr__(self):
        return 'Hswish()'

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def sigmoid(x, inplace=False):
    return x.sigmoid_() if inplace else x.sigmoid()
class SELayer(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25):
        super(SELayer, self).__init__()
        reduce_chs = max(1, int(in_chs * se_ratio))
        self.act_fn = F.relu
        self.gate_fn = sigmoid
        reduced_chs = reduce_chs or in_chs
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        # NOTE adaptiveavgpool can be used here, but seems to cause issues with NVIDIA AMP performance
        x_se = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x_se = self.conv_reduce(x_se)
        x_se = self.act_fn(x_se, inplace=True)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

class MBConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, expansion=3, stride=1, dilation=1, act_func='h_swish', attention=False, drop_connect_rate=0.0, dense=False, affine=True):
        super(MBConv, self).__init__()
        interChannels = expansion * C_out
        self.op1 = nn.Sequential(
            nn.Conv2d(C_in, interChannels, kernel_size=1, stride=1, padding=0, bias=False, groups=1),
            nn.BatchNorm2d(interChannels, affine=affine)
        )
        self.op2 = nn.Sequential(
                   nn.Conv2d(interChannels, interChannels, kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) / 2) * dilation, bias=False, dilation=dilation, groups=interChannels),
                   nn.BatchNorm2d(interChannels, affine=affine)
        )
        self.op3 = nn.Sequential(
            nn.Conv2d(interChannels, C_out, kernel_size=1, stride=1, padding=0, bias=False, groups=1),
            nn.BatchNorm2d(C_out, affine=affine)
        )

        if act_func == 'relu':
            self.act_func = nn.ReLU(inplace=True)
        else:
            self.act_func = Hswish(inplace=True)
        if attention:
            self.se = SELayer(interChannels)
        else:
            self.se = nn.Sequential()
        self.drop_connect_rate = drop_connect_rate
        self.stride = stride
        self.dense = int(dense)
        self.C_in = C_in
        self.C_out = C_out

    def forward(self, x):
        out = self.op1(x)
        out = self.act_func(out)
        out = self.op2(out)
        out = self.act_func(out)
        out = self.se(out)
        out = self.op3(out)

        if self.drop_connect_rate > 0:
            out = drop_path(out, drop_prob=self.drop_connect_rate, training=self.training)
        if self.stride == 1 and self.dense:
            out = torch.cat([x, out], dim=1)
        elif self.stride == 1 and self.C_in == self.C_out:
            out = out + x
        return out

class DenseBlock(nn.Module):
    def __init__(self, layer_types, in_channels, out_channels, kernel_sizes, expansions, strides, act_funcs, attentions, drop_connect_rates, dense):
        super(DenseBlock, self).__init__()
        self.layer_types = list(map(int, layer_types.split()))
        self.in_channels = list(map(int, in_channels.split()))
        self.out_channels = list(map(int, out_channels.split()))
        self.kernel_sizes = list(map(int, kernel_sizes.split()))
        self.expansions = list(map(int, expansions.split()))
        self.attentions = list(map(bool, map(int, attentions.split())))
        self.strides = list(map(int, strides.split()))
        self.act_funcs = list(map(str, act_funcs.split()))
        self.drop_connect_rates = list(map(float, drop_connect_rates.split()))
        self.dense = int(dense)

        self.layer = self._make_dense(len(self.out_channels))

    def _make_dense(self, nDenseBlocks):
        layers = []
        for i in range(int(nDenseBlocks)):
            if self.layer_types[i] == 0:
                layers.append(Identity())
            else:
                layers.append(MBConv(self.in_channels[i], self.out_channels[i], self.kernel_sizes[i], self.expansions[i], self.strides[i], 1, self.act_funcs[i], self.attentions[i], self.drop_connect_rates[i], self.dense))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        out = self.layer(out)
        return out

class EvoCNNModel(nn.Module):
    def __init__(self):
        super(EvoCNNModel, self).__init__()
        self.Hswish = Hswish(inplace=True)
        #generated_init


    def forward(self, x):
        out_aux = None
        #generate_forward

        out = self.evolved_block2(out)
        #out = self.Hswish(self.bn_end1(self.conv_end1(out)))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = self.Hswish(self.conv_end1(out))
        out = out.view(out.size(0), -1)
        # out = self.Hswish(self.dropout(self.linear1(out)))

        out = F.dropout(out, p=0.2, training=self.training)
        out = self.linear(out)

        return out, out_aux


class TrainModel(object):
    def __init__(self, is_test, particle, batch_size, weight_decay):
        if is_test:
            full_trainloader = data_loader.get_train_loader('../datasets/CIFAR10_data', batch_size=batch_size, augment=True,shuffle=True, random_seed=2312391, show_sample=False,num_workers=4, pin_memory=True)
            testloader = data_loader.get_test_loader('../datasets/CIFAR10_data', batch_size=batch_size, shuffle=False,num_workers=4, pin_memory=True)
            self.full_trainloader = full_trainloader
            self.testloader = testloader
        else:
            trainloader, validate_loader = data_loader.get_train_valid_loader('../datasets/CIFAR10_data', batch_size=256,augment=True, subset_size=1,valid_size=0.1, shuffle=True,random_seed=2312390, show_sample=False,num_workers=4, pin_memory=True)
            self.trainloader = trainloader
            self.validate_loader = validate_loader

        net = EvoCNNModel()
        cudnn.benchmark = True
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0
        self.net = net
        self.criterion = criterion
        self.best_acc = best_acc
        self.flops = 10e9
        self.best_epoch = 0
        self.file_id = os.path.basename(__file__).split('.')[0]
        self.weight_decay = weight_decay
        self.particle = copy.deepcopy(particle)

        if not is_test:
            self.log_record('loading weights from the weight pool', first_time=True)
        else:
            self.log_record('testing, not need to load weights from the weight pool', first_time=True)
        self.net = self.net.cuda()

    def load_weights_from_pool(self):
        pool_path = './trained_models/'
        subparticle_length = Utils.get_params('PSO', 'particle_length') // 3
        subParticles = [self.particle[0:subparticle_length], self.particle[subparticle_length:2 * subparticle_length],
                        self.particle[2 * subparticle_length:]]
        weight_pool = self.get_flist(pool_path)
        # print(weight_pool)

        for op_name in ['begin1', 'end1', 'linear', 'tranLayer0', 'tranLayer1']:
            is_found = False
            for op in weight_pool:
                if op_name in op:
                    is_found = True
                    self.log_record('Inheriting the weights of '+op_name + ' from the weight pool.')
                    # op_weights = torch.load(pool_path+op, map_location='cuda')
                    op_weights = torch.load(pool_path + op, map_location=torch.device('cpu'))
                    adjusted_op_weights = self.adjust_weights(op_name, op_weights)
                    self.net.load_state_dict(adjusted_op_weights, strict=False)
            if not is_found:
                self.log_record(op_name+' is not found in the weight pool.')

        for j, subParticle in enumerate(subParticles):
            for number, dimen in enumerate(subParticle):
                type = int(dimen)
                if not type == 0:
                    is_found = False
                    op_name = 'evolved_block'+str(j)+'.layer.'+str(number)
                    for op in weight_pool:
                        if op_name+'-'+str(type)+'-' in op:
                            is_found = True
                            self.log_record('Inheriting the weights of ' + op_name+'-'+str(type) + ' from the weight pool.')
                            # op_weights = torch.load(pool_path + op, map_location='cuda')
                            op_weights = torch.load(pool_path + op, map_location=torch.device('cpu'))
                            adjusted_op_weights = self.adjust_weights(op_name, op_weights)
                            self.net.load_state_dict(adjusted_op_weights, strict=False)
                    if not is_found:
                        self.log_record(op_name+'-'+str(type)+' is not found in the weight pool.')

    def adjust_weights(self, op_name, op_weights):
        net_items = self.net.state_dict().items()
        adjusted_op_weights = {}

        for curr_key, curr_val in net_items:
            if op_name+'.' in curr_key:
                curr_val = curr_val.numpy()
                pool_weight = op_weights[curr_key].numpy()
                shape_pool_op = np.shape(pool_weight)
                shape_curr_op = np.shape(curr_val)
                shape_inherit = np.min([shape_pool_op, shape_curr_op], axis=0)
                _, subset_inherit = self.get_subset(pool_weight, shape_inherit)
                adjusted_op_weight = self.inherit_weight(curr_val, subset_inherit)
                adjusted_op_weights[curr_key] = torch.from_numpy(adjusted_op_weight)
        return adjusted_op_weights

    def get_subset(self, a, bshape):
        slices = []
        for i, dim in enumerate(a.shape):
            center = dim // 2
            start = center - bshape[i] // 2
            stop = start + bshape[i]
            slices.append(slice(start, stop))
        return slices, a[tuple(slices)]

    def inherit_weight(self, curr_val, subset_inherit):
        inherit_shape = np.shape(subset_inherit)
        slices = []
        # curr_val = copy.deepcopy(curr_val)
        for i, dim in enumerate(curr_val.shape):
            center = dim // 2
            start = center - inherit_shape[i] // 2
            stop = start + inherit_shape[i]
            slices.append(slice(start, stop))
        curr_val[tuple(slices)] = subset_inherit
        return curr_val

    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime( '%Y-%m-%d %H:%M:%S' )
        if first_time:
            file_mode = 'w'
        else:
            file_mode = 'a+'
        f = open('./log/%s.txt'%(self.file_id), file_mode)
        f.write('[%s]-%s\n'%(dt, _str))
        f.flush()
        f.close()

    def train(self, epoch, optimizer):
        self.net.train()
        running_loss = 0.0
        total = 0
        correct = 0
        # flops = 10e9
        for ii, data in enumerate(self.trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs, out_aux = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 5)
            optimizer.step()
            running_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

            if epoch==0 and ii==0:
                inputs = torch.randn(1, 3, 32, 32)
                inputs = Variable(inputs.cuda())
                params = sum(p.numel() for p in self.net.parameters())
                #flops1, params1 = profile(self.net, inputs=(inputs,))
                self.flops = profile_macs(copy.deepcopy(self.net), inputs)
                self.log_record('#parameters:%d, #FLOPs:%d' % (params, self.flops))
        self.log_record('Train-Epoch:%4d,  Loss: %.4f, Acc:%.4f'% (epoch+1, running_loss/total, (correct/total)))


    def final_train(self, epoch, optimizer):
        self.net.train()
        running_loss = 0.0
        total = 0
        correct = 0
        # flops = 10e9
        for ii, data in enumerate(self.full_trainloader, 0):
            inputs, labels = data
            # inputs = F.interpolate(inputs, size=40, mode='bicubic', align_corners=False)
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            optimizer.zero_grad()
            outputs, out_aux = self.net(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 5)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()

            if epoch==0 and ii==0:
                inputs = torch.randn(1, 3, 32, 32)
                inputs = Variable(inputs.cuda())
                params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
                #flops1, params1 = profile(self.net, inputs=(inputs,))
                self.flops = profile_macs(copy.deepcopy(self.net), inputs)
                self.log_record('#parameters:%d, #FLOPs:%d' % (params, self.flops))
        self.log_record('Train-Epoch:%4d,  Loss: %.4f, Acc:%.4f' % (epoch + 1, running_loss / total, (correct / total)))

    def get_flist(self, path):
        for root, dirs, files in os.walk(path):
            # self.log_record(files)
            pass
        return files

    def update_op(self, op_name, curr_acc, type=None):
        pool_path = './trained_models/'
        net_items = self.net.state_dict().items()
        updated_weights = {}

        for sub_key, val in net_items:
            if op_name in sub_key:
                updated_weights[sub_key] = val

        if type:
            updated_weights_path = pool_path + op_name + '-' + str(type) + '-' + str(curr_acc) + '.pt'
        else:
            updated_weights_path = pool_path+op_name+'-'+str(curr_acc)+'.pt'
        torch.save(updated_weights, updated_weights_path)

    def update_weight_pool(self, curr_acc):
        pool_path = './trained_models/'
        subparticle_length = Utils.get_params('PSO', 'particle_length') // 3
        subParticles = [self.particle[0:subparticle_length], self.particle[subparticle_length:2 * subparticle_length],
                        self.particle[2 * subparticle_length:]]
        weight_pool = self.get_flist(pool_path)
        curr_acc = curr_acc.cpu().numpy()
        curr_acc = np.around(curr_acc,4)
        for op_name in ['begin1', 'end1', 'linear', 'tranLayer0', 'tranLayer1']:
            is_found = False
            for op in weight_pool:
                if op_name in op:
                    is_found = True
                    op_acc = float(op.split('.pt')[0].split('-')[1])
                    if curr_acc > op_acc:
                        self.update_op(op_name, curr_acc)
                        os.remove(pool_path + op)
                        self.log_record('update new op: %s, and remove old op: %s'%(op_name+'-'+str(curr_acc), op))
            if not is_found:
                self.update_op(op_name, curr_acc)
                self.log_record('update new op: %s' % (op_name + '-' + str(curr_acc)))

        for j, subParticle in enumerate(subParticles):
            for number, dimen in enumerate(subParticle):
                type = int(dimen)
                if not type == 0:
                    is_found = False
                    op_name = 'evolved_block'+str(j)+'.layer.'+str(number)
                    for op in weight_pool:
                        if op_name+'-'+str(type)+'-' in op:
                            is_found = True
                            op_acc = float(op.split('.pt')[0].split('-')[-1])
                            if curr_acc > op_acc:
                                self.update_op(op_name, curr_acc, type)
                                os.remove(pool_path + op)
                                self.log_record('update new op: %s, and remove old op: %s' % (op_name + '-'+ str(type)+'-' + str(curr_acc), op))
                    if not is_found:
                        self.update_op(op_name, curr_acc, type)
                        self.log_record('update new op: %s' % (op_name + '-' + str(type) + '-' + str(curr_acc)))

    def validate(self, epoch):
        self.net.eval()
        test_loss = 0.0
        total = 0
        correct = 0
        # is_terminate = 0
        for _, data in enumerate(self.validate_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs,_ = self.net(inputs)
            loss = self.criterion(outputs, labels)
            test_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        if correct / total > self.best_acc:
            self.best_epoch = epoch
            self.best_acc = correct / total
            self.update_weight_pool(self.best_acc)
        self.log_record('Validate-Epoch:%4d,  Validate-Loss:%.4f, Acc:%.4f'%(epoch + 1, test_loss/total, correct/total))
        # return is_terminate

    def process(self):
        total_epoch = Utils.get_params('SEARCH', 'epoch_test')
        min_epoch_eval = Utils.get_params('SEARCH', 'min_epoch_eval')

        lr_rate = 0.03
        optimizer = optim.SGD(self.net.parameters(), lr=lr_rate, momentum=0.9, weight_decay=5e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, min_epoch_eval)

        params = sum(p.numel() for p in self.net.parameters())
        self.validate(0)
        for p in range(min_epoch_eval):
            self.train(p, optimizer)
            scheduler.step()
            self.validate(p)
        return self.best_acc, params, self.flops

    def process_test(self):
        params = sum(p.numel() for n,p in self.net.named_parameters() if p.requires_grad and not n.__contains__('auxiliary'))
        total_epoch = Utils.get_params('SEARCH', 'epoch_test')
        lr_rate = 0.1
        optimizer = optim.SGD(self.net.parameters(), lr=lr_rate, momentum=0.9, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)

        self.test(0)
        for p in range(total_epoch):
            if p < 5:
                optimizer_ini = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9, weight_decay=self.weight_decay)
                self.final_train(p, optimizer_ini)
                self.test(p)
            else:
                self.final_train(p, optimizer)
                self.test(p)
                scheduler.step()
        return self.best_acc, params, self.flops

    def test(self,p):
        self.net.eval()
        test_loss = 0.0
        total = 0
        correct = 0
        for _, data in enumerate(self.testloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs,_ = self.net(inputs)
            loss = self.criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.data).sum()
        if correct / total > self.best_acc:
            torch.save(self.net.state_dict(), './trained_models/best_CNN.pt')
            self.best_acc = correct / total
        self.log_record('Test-Loss:%.4f, Acc:%.4f' % (test_loss / total, correct / total))

class RunModel(object):
    def do_work(self, gpu_id, curr_gen, file_id, is_test, particle=None, batch_size=None, weight_decay=None):
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        best_acc = 0.0
        params = 1e9
        flops = 10e9
        try:
            m = TrainModel(is_test, particle, batch_size, weight_decay)
            m.log_record('Used GPU#%s, worker name:%s[%d]' % (gpu_id, multiprocessing.current_process().name, os.getpid()))
            if is_test:
                best_acc, params, flops = m.process_test()
            else:
                best_acc, params, flops = m.process()
            #import random
            #best_acc = random.random()
        except BaseException as e:
            print('Exception occurs, file:%s, pid:%d...%s'%(file_id, os.getpid(), str(e)))
            m.log_record('Exception occur:%s'%(str(e)))
        finally:
            m.log_record('Finished-Acc:%.4f'%best_acc)
            m.log_record('Finished-Err:%.4f' % (1-best_acc))

            f1 = open('./populations/err_%02d.txt'%(curr_gen), 'a+')
            f1.write('%s=%.5f\n'%(file_id, 1-best_acc))
            f1.flush()
            f1.close()

            f2 = open('./populations/params_%02d.txt' % (curr_gen), 'a+')
            f2.write('%s=%d\n' % (file_id, params))
            f2.flush()
            f2.close()

            f3 = open('./populations/flops_%02d.txt' % (curr_gen), 'a+')
            f3.write('%s=%d\n' % (file_id, flops))
            f3.flush()
            f3.close()
"""


