from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
from numpy.lib.stride_tricks import as_strided

#Example Sigmoid
#这个类中包含了forward和backward函数
class Sigmoid():
    def __init__(self):
        pass

    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self, z):
        return self.forward(z) * (1 - self.forward(z))


##在原LeNet-5上进行少许修改后的网路结构
"""
conv1: in_channels: 1, out_channel:6, kernel_size=(5x5), pad=0, stride=1, activation: relu
avgpool1: in_channels: 6, out_channels:6, kernel_size = (2x2), stride=2
conv2: in_channels: 6, out_channel:16, kernel_size=(5x5), pad=0, stride=1, activation: relu
avgpool2: in_channels: 16, out_channels:16, kernel_size = (2x2), stride=2
flatten
fc1: in_channel: 256, out_channels: 128, activation: relu
fc2: in_channel: 128, out_channels: 64, activation: relu
fc3: in_channel: 64, out_channels: 10, activation: relu
softmax:

tensor: (1x28x28)   --conv1    -->  (6x24x24)
tensor: (6x24x24)   --avgpool1 -->  (6x12x12)
tensor: (6x12x12)   --conv2    -->  (16x8x8)
tensor: (16x8x8)    --avgpool2 -->  (16x4x4)
tensor: (16x4x4)    --flatten  -->  (256)
tensor: (256)       --fc1      -->  (128)
tensor: (128)       --fc2      -->  (64)
tensor: (64)        --fc3      -->  (10)
tensor: (10)        --softmax  -->  (10)
"""
##Conv类中的部分函数
def im2col(inputs, filter_size, stride=1, pad=(0, 0)):
    N, C, H, W = inputs.shape
    FH, FW = filter_size
    out_h = (H + 2 * pad[0] - FH) // stride + 1
    out_w = (W + 2 * pad[1] - FW) // stride + 1
    img = np.pad(inputs, ((0, 0), (0, 0), (pad[0], pad[0]), (pad[1], pad[1])), 'constant')
    col = np.zeros((N, C, FH, FW, out_h, out_w))
    for y in range(FH):
        y_max = y + stride * out_h
        for x in range(FW):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col, out_h, out_w

def col2im(col, input_shape, filter_shape, out_shape, stride=1, pad=(0,0)):
    N, C, H, W = input_shape
    out_h ,out_w = out_shape
    filter_h, filter_w = filter_shape
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2*pad[0] + stride - 1, W + 2*pad[1] + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    return img[:, :, pad[0]:H + pad[0], pad[1]:W + pad[1]]

def conv(inputs, kernel, bias, stride=1, pad=(0, 0)):
    N, _, _, _ = inputs.shape
    FN, _, FH, FW = kernel.shape
    col, OH, OW = im2col(inputs, [FH, FW], stride, pad)
    ker = kernel.reshape((FN, -1)).T  
    dot = np.dot(col, ker)  
    result = dot + bias
    result = result.T.reshape((FN, N, OH, OW))
    result = result.transpose(1, 0, 2, 3)
    return result

def flip180(arr):
    FC, FN, FH, FW = arr.shape
    new_arr = arr.reshape((FC, FN, -1))
    new_arr = new_arr[..., ::-1]
    new_arr = new_arr.reshape((FC, FN, FH, FW))
    return new_arr

def reconv(delta_in, kernel, stride=1, pad=(0, 0)):
    _, _, OH, OW = delta_in.shape
    _, _, FH, FW = kernel.shape
    kernel0 = kernel.transpose(1, 0, 2, 3)
    kernel0 = flip180(kernel0)
    if stride > 1:
        hid = np.repeat(np.arange(1,OH), stride-1)
        wid = np.repeat(np.arange(1,OW), stride-1)
        delta_in = np.insert(delta_in, hid, 0, axis=2)
        delta_in = np.insert(delta_in, wid, 0, axis=3)
    delta_out = conv(delta_in, kernel0, 0, pad=(FH-1, FW-1))
    _, _, H1, W1 = delta_out.shape
    delta_out = delta_out[..., pad[0]:H1 - pad[0], pad[1]:W1 - pad[1]]
    return delta_out

##卷积层
class Conv(object):
    def __init__(self, kernel_size, stride=1, pad=(0,0)):
        self.pad = pad
        self.stride = stride
        self.kernel_size = kernel_size

    def init_weight(self):
        FN, FC, FH, FW = self.kernel_size
        std = np.sqrt(2 / (FN*FC*FH*FW))
        kernel = np.random.normal(0, std, (FN, FC, FH, FW))
        bias = np.random.normal(0, 0.01, (1, FN))
        self.kernel = kernel
        self.bias = bias

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = conv(self.inputs, self.kernel, self.bias, self.stride, self.pad)
        return self.outputs

    def backward(self, delta_in, lr):
        if len(delta_in.shape) < 4:
            delta_in = np.expand_dims(delta_in, axis=(2,3))
        self.delta_in = delta_in
        self.delta_out = reconv(self.delta_in, self.kernel, self.stride, self.pad)
        temp = np.sum(self.delta_in, axis=(0, 2, 3))
        temp=temp.reshape((1,-1))
        self.bias -= lr * temp
        delta_in0 = self.delta_in.swapaxes(0,1) 
        inputs0 = self.inputs.swapaxes(0,1)     
        kernel_gra = conv(inputs0, delta_in0, 0, self.stride, self.pad)    
        kernel_gra = kernel_gra.swapaxes(0,1)
        self.kernel = self.kernel - lr * kernel_gra
        return self.delta_out

##ReLU激活函数     
class ReLU(object):
    def __init__(self):
        pass
    def forward(self, inputs):
        self.Data_in = inputs
        outputs = inputs.copy()
        outputs[outputs < 0] *= 0.01
        return outputs
    def backward(self, delta):
        dx = np.ones_like(self.Data_in)
        dx[self.Data_in <= 0] = 0.01
        return dx * delta

##池化层
class Avgpool(object):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        self.I = in_channel
        self.O = out_channel
        self.stride = stride       
    def forward(self, inputs, OH, OW):
        B, _, _, _ = inputs.shape
        div = self.stride ** 2
        self.div = div
        shape = (B, self.O, OH, OW, self.stride, self.stride)
        strides = (*inputs.strides[:-2], inputs.strides[-2] * self.stride, inputs.strides[-1] * self.stride, *inputs.strides[-2:])
        return np.mean(as_strided(inputs, shape=shape, strides=strides), axis=(-2, -1))   
    def backward(self,delta):
        return np.repeat(np.repeat(delta, self.stride, axis=-1), self.stride, axis=-2) / self.div

class Flatten(object):
    def __init__(self):
        pass
    def forward(self, inputs):
        self.sha = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)
    def backward(self, delta):
        return delta.reshape(self.sha)

##全连接层
class Fc(object):
    def __init__(self, fin, fout):
        self.fin = fin
        self.fout = fout
        self.weight = np.zeros((fin, fout))
    def init_weight(self):
        self.timestep = 0
        self.m1 = 0
        self.m2 = 0
        self.v1 = 0
        self.v2 = 0
        self.b1 = 0.9
        self.b2 = 0.999
        self.e = 1e-8
        r = np.sqrt(6 / (self.fin + self.fout))
        self.weight = np.random.uniform(-r, r,(self.fin, self.fout))
        self.bias = np.random.uniform(-r, r, self.fout)
    def forward(self, inputs):
        self.Data_in = inputs.reshape(-1, self.fin)
        return np.dot(inputs, self.weight) + self.bias
    def backward(self,delta,lr):
        dx = np.tensordot(delta, self.weight, axes=[(1,), (1,)])
        dw = np.tensordot(self.Data_in, delta, axes=[(0,), (0,)])
        db = np.sum(delta, axis=0)
        self.timestep += 1
        self.m1 = self.b1 * self.m1 + (1 - self.b1) * dw
        self.m2 = self.b1 * self.m2 + (1 - self.b1) * db
        self.v1 = self.b2 * self.v1 + (1 - self.b2) * dw ** 2
        self.v2 = self.b2 * self.v2 + (1 - self.b2) * db ** 2
        mtt1 = self.m1 / (1 - self.b1 ** (self.timestep))
        vtt1 = self.v1 / (1 - self.b2 ** (self.timestep))
        mtt2 = self.m2 / (1 - self.b1 ** (self.timestep))  
        vtt2 = self.v2 / (1 - self.b2 ** (self.timestep))
        self.weight -= lr * mtt1 / (self.e + np.sqrt(vtt1))
        self.bias -= lr * mtt2 / (self.e + np.sqrt(vtt2))
        return dx
##SoftmaxLoss       
class Softmax(object):
    def __init__(self, size):
        self.size = size
    def forward(self, inputs):
        B, _ = inputs.shape
        self.batch = B
        outputs = np.zeros_like(inputs)
        for b in range(B):
            outputs[b] = np.exp(inputs[b]) / np.sum(np.exp(inputs[b]))
        self.result = outputs
        return outputs
    def backward(self, ind):
        Ind = ind.reshape(self.batch, -1)
        dx = self.result.copy()
        for b in range(self.batch):
            dx[b,Ind[b]] -= 1
        return dx            
##LeNet       
class LeNet(object):
    def __init__(self):
        '''
        初始化网路，在这里你需要，声明各Conv类， AvgPool类，Relu类， FC类对象，SoftMax类对象
        并给 Conv 类 与 FC 类对象赋予随机初始值
        注意： 不要求做 BatchNormlize 和 DropOut, 但是有兴趣的可以尝试
        '''
        self.conv1 = Conv((6, 1, 5, 5))
        self.relu1 = ReLU()
        self.avgpool1 = Avgpool(6, 6, (2, 2), 2)
        self.conv2 = Conv((16, 6, 5, 5))
        self.relu2 = ReLU()
        self.avgpool2 = Avgpool(16, 16, (2, 2), 2)
        self.flatten = Flatten()
        self.fc1 = Fc(256, 128)
        self.relu3 = ReLU()
        self.fc2 = Fc(128, 64)
        self.relu4 = ReLU()
        self.fc3 = Fc(64, 10)
        self.relu5 = ReLU()
        self.softmax = Softmax(10)
        self.init_weight()          
        print("initialize")
    def init_weight(self):
        self.conv1.init_weight()
        self.conv2.init_weight()
        self.fc1.init_weight()
        self.fc2.init_weight()
        self.fc3.init_weight()
    def forward(self, x):
        """前向传播
        x是训练样本， shape是 B,C,H,W
        这里的C是单通道 c=1 因为 MNIST中都是灰度图像
        返回的是最后一层 softmax后的结果
        也就是 以 One-Hot 表示的类别概率

        Arguments:
            x {np.array} --shape为 B，C，H，W
        """
        x1 = self.relu1.forward(self.conv1.forward(x))
        x2 = self.avgpool1.forward(x1, 12, 12)
        x3 = self.relu2.forward(self.conv2.forward(x2))
        x4 = self.avgpool2.forward(x3, 4, 4)
        x4_flat = self.flatten.forward(x4)
        x5 = self.relu3.forward(self.fc1.forward(x4_flat))
        x6 = self.relu4.forward(self.fc2.forward(x5))
        x7 = self.relu5.forward(self.fc3.forward(x6))
        result = self.softmax.forward(x7)
        return result
    def backward(self, index_true, lr=1.0e-3):
        """根据error，计算梯度，并更新model中的权值
        Arguments:
            error {np array} -- 即计算得到的loss结果
            lr {float} -- 学习率，可以在代码中设置衰减方式
        """
        delta1 = self.softmax.backward(index_true)
        delta2 = self.fc3.backward(self.relu5.backward(delta1), lr = lr)
        delta3 = self.fc2.backward(self.relu4.backward(delta2), lr = lr)
        delta4 = self.fc1.backward(self.relu3.backward(delta3), lr = lr)
        delta4_flat = self.flatten.backward(delta4)
        delta5 = self.avgpool2.backward(delta4_flat)
        delta6 = self.conv2.backward(self.relu2.backward(delta5), lr = lr)
        delta7 = self.avgpool1.backward(delta6)
        self.conv1.backward(self.relu1.backward(delta7), lr = lr)
    def evaluate(self, inputs, labels):
        """
        x是测试样本， shape 是BCHW
        labels是测试集中的标注， 为one-hot的向量
        返回的是分类正确的百分比

        在这个函数中，建议直接调用一次forward得到pred_labels,
        再与 labels 做判断

        Arguments:
            x {np array} -- BCWH
            labels {np array} -- B x 10
        """
        B, H, W = inputs.shape
        inputs = inputs.reshape(B, 1, H, W)
        pred = self.forward(inputs)
        label = np.argmax(labels, axis = 1)
        result = np.argmax(pred, axis = 1)
        accuracy = np.zeros_like(result)
        accuracy[result == label] = 1
        return np.sum(accuracy) / np.size(accuracy)
    def data_augmentation(self, images):
        '''
        数据增强，可选操作，非强制，但是需要合理
        一些常用的数据增强选项：ramdom scale，translate，color(grayscale) jittering，rotation, gaussian noise,
        这一块儿允许使用 opencv 库或者 PIL image 库
        比如把6旋转90度变成了9，但是仍然标签为6就不合理了
        '''
        return images
    def compute_loss(self,result,labels):
        loss = 0
        for b in range(result.shape[0]):
            index = np.argmax(labels[b])
            loss -= np.log(result[b,index])
        return loss
    def fit(self, train_image, train_label, test_image=None, test_label=None, epoches=10, batch_size=16, lr=1.0e-3):
        sum_time = 0
        accuracies = []
        for epoch in range(epoches):
            ##可选操作，数据增强
            train_image = self.data_augmentation(train_image)
            ##随机打乱train_image 的顺序，但是注意train_image和test_label仍需对应
            '''
            # 1.一次forward，bachword肯定不能是所有的图像一起,
            因此需要根据batch_size将train_image,和 train_label分成:[ batch0 | batch1 | ... | batch_last]
            '''
            batch_images = train_image.reshape(-1,batch_size,1,28,28) #请实现step #1
            batch_labels = train_label.reshape(-1,batch_size,10) #请实现step #1

            last = time.time() #计时开始
            for imgs, labels in zip(batch_images, batch_labels):
                '''
                这里我只是给了一个范例，大家在实现上可以不一定要按照这个严格的2,3,4步骤
                我在验证大家的模型时，只会在main中调用fit函数和evaluate 函数。
                2.做一次forward，得到pred结果 eg. pred = self.forward(imgs)
                3.pred和labels做一次loss eg. error = self.compute_loss(pred, labels)
                4.做一次backward，更新网络权值 eg. self.backward(error, lr=1e-3)
                '''
                self.forward(imgs)
                index_true = np.argmax(labels, axis=1)
                self.backward(index_true, lr)
            duration = time.time() - last
            sum_time += duration

            if epoch % 5 == 0:
                accuracy = self.evaluate(test_image, test_label)
                print("epoch{} accuracy{}".format(epoch, accuracy))
                accuracies.append(accuracy)

        avg_time = sum_time / epoches
        return avg_time, accuracies


