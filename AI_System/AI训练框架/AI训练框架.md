# PyTorch

>**该笔记为作者通过TUM(Technische Universität München)的Introduction to Deep Leraning课程、B站Up主“我是土堆”的“PyTorch”课程以及PyTorch官方Documentation"总结的PyTorch学习笔记**

## *Dataset*

### 基本使用方法及结构

* Map style

    ```python
    def Dataset():
        def __init__(sefl,*args,**kwds):
        
        def __getitem__(self, index):

        def __len__(self):
    ```

* Iteration style

    ```python
    def IterableDataset():
        def __init__():
        
        def __iter__(self): #构造迭代器
    ```

## *Dataloader*



### 重要参数

* `dataset`: datasets from wihcih to load the data
* `batch_size`: how many samples per batch to load
* `shuffle`: 当设置为True时每一个Epoch中sample的顺序都不相同
* `num_workers`
  * 当默认设置为0时只使用主进程加载数据
  * Win环境下用多个进程可能会出现`BrokenPipeError`的错误，此时考虑设置为0
* `drop_last`：当设置为True且#samples/batch_size有余数时舍去最后一组batch

### 基本使用方法及结构

```python
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Prepare test data
test_data = torchvision.datasets.CIFAR10("./dataset", train = False,download=True, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

img, target = test_data[0] # CIFAR10数据集中__getitem__规定返回 img和target
print(img.shape) # torch.Size([3, 32, 32])
print(target) # 3 target就是label

writer = SummaryWriter("./logs")
# batch_size=4 就相当于每4张图片为一组，将这4张图片的img和target分别打包成两个list，喂给神经网络
for epoch in range(2):
    step = 0
    print("Start training of epoch #:{}".format(epoch))
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape) # torch.Size([4, 3, 32, 32])
        # print(targets) # tensor([1, 8, 2, 6]) 4张图片分别所属的target
        writer.add_images("Epoch:{}".format(epoch), imgs, step)
        step = step + 1

writer.close()
```

## *TensorBoard*



### 基本使用方法及结构

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

writer.add_image(tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')

#tag是Data identifier；scalar_value是图像的y轴；global_step是x轴
writer.add_scaler(tag, scalar_value, global_step=None, walltime=None, new_style=False, double_presicion=False)

writer.close()
```

* add_image中的img_tensor参数可以是torch.Tensor，numpy.array，or string/blobname
  * 通过PIL包中PIL.open打开的图片类型是不符合img_tensor的，需要用`np.array()`进行转换后使用
  * 使用opencv读取的图片数据类型是numpy.array，可以直接被使用
* 需要注意add_image中的dataformats

## *Transforms*

### Transforms的import模块

```python
from torchvision import transforms
```

### Documentation及模块作用

当使用PyTorch训练用于图片的神经网络之前，需要对图片进行Pre-processing。tranforms.py中定义了很多对图像的预处理工具，最常用的有如ToTensor、Normalize、Rescale、CenterCrop等

### Transform工具的基本结构和使用方法（以ToTensor为例）

* ToTensor的结构

    ```python
    class ToTensor(object):
        def __call__(self, pic):
            return F.to_tensor(pic)
        
        def __repr__(self):
            return self.__class__.__name__+'()'
    ```

* ToTensor的使用

    ```python
    from PIL import Image
    img_path = ""
    img = Image.open(img_path) #用Image.open打开的图片类型为PIL.JpegImagePlugin.JpegImageFile Class
    
    tensor_trans = transforms.ToTensor() #首先要具体化给的工具，因为如Normalize之类的预处理还需要指定参数
    tesnsor_img = tensor_trans(img) #使用制定好的工具后再进行预处理
    ```

### 通过ToTensor解决两个问题

* transforms该如何使用

    ```python
    trans_norm = transforms.Normalize([1, 3, 5], [3, 2, 1])
    img_norm = trans_norm(img_tensor)
    ```

  * 首先具体化预处理工具
  * 将图片输入定制好的预处理工具
* 为什么需要Tensor数据类型：Tensor和numpy.array是很类似的数据结构，但他是专门针对GPU训练所设计的多维矩阵，有着很多深度学习需要的参数

### 组合图片预处理

```python
trans_compose = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([1, 3, 5], [3, 2, 1])
    ])
```

## *Torchvision自带数据集的使用*

### 使用示例

```python
import torchvision

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    ...
]) #创建预处理模块，可以在下载数据集时顺便完成预处理，很方便

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)

test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True) 
```

* `downlaod`一直设置为True比较方便，还可以自动解压缩数据集

## *神经网络的实现*

### 神经网络的基本骨架-`torch.nn.Module`

* Documentation

>**Base class for all neural network modules.**  
Your models should also subclass this class.  
Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes

`nn.Module`是所有自定义神经网络的骨架，即所有自定义class的父类

* 使用实例

    ```python
    import torch
    from torch import nn
    
    class TestNetwort(nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, input):
            output = input + 1
            return output
    
    my_network = TestNetwort() #实例化
    
    x = torch.tensor(1.0)
    output = my_network(x)
    print(output)
    ```

### 卷积操作和卷积层

* `torch.nn`是对`torch.nn.functional`的一种封装，便于使用，但实现细节如`nn.Conv1d`等在`torch.nn.functional`之中
* `torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros'...)`的参数
  * `input` - shape (minibatch, in_channels, iH, iW)
  * `weight` - filters of shape (out_channels, $\frac{in\_channels}{groups}$, kH, kW)
  * `bias`
  * `stride`
  * `padding`
  * `dilation`：空洞卷积，一般默认为1
  * `groups`
  * `padding_mode`
* shape
  * input:$(N,\ C_{in},\ H_{in},\ W_{in})$
  * output:$(N,\ C_{out},\ H_{out},\ W_{out})$
  * $H_{out}=\left[\frac{H_{in}+2\times padding[0]-dilation[0]\times (kernel_size[0]-1)-1}{stride[0]}+1\right]$
  * $W_{out}=\left[\frac{W_{in}+2\times padding[1]-dilation[1]\times (kernel_size[1]-1)-1}{stride[1]}+1\right]$
* 卷积操作

    ```python
    import torch
    import torch.nn.functional as F

    input = torch.tensor([[1, 2, 0, 3, 1],
                        [0, 1, 2, 3, 1],
                        [1, 2, 1, 0, 0],
                        [5, 2, 3, 1, 1],
                        [2, 1, 0, 1, 1]])

    kernel = torch.tensor([[1, 2, 1],
                        [0, 1, 0],
                        [2, 1, 0]])

    print(input.shape)
    print(kernel.shape)
    # input和kernel很明显并不满足定义的size，要进行resize
    input = torch.reshape(input, (1, 1, 5, 5))
    kernel = torch.reshape(kernel, (1, 1, 3, 3))

    output1 = F.conv2d(input, kernel, stride=1)
    print(output1)

    output2 = F.conv2d(input, kernel, stride=2)
    print(output2)

    output3 = F.conv2d(input, kernel, stride=1, padding=1)
    print(output3)
    ```

* `torch`中`conv2d`函数的应用

    ```python
    import torch
    import torchvision
    from torch import nn
    from torch.nn import Conv2d
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    
    dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)
    dataloader = DataLoader(dataset, batch_size=64)
    
    class TestNetwork(nn.Module):
        def __init__(self):
            super(TestNetwork, self).__init__()
            self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
    
        def forward(self, x):
            x = self.conv1(x)
            return x
    
    my_Network = TestNetwork()
    print(my_Network)
    
    writer = SummaryWriter("./logs")
    step = 0
    for data in dataloader:
        imgs, targets = data
        output = my_Network(imgs)
        print(imgs.shape)
        print(output.shape)
        # torch.Size([64, 3, 32, 32])
        writer.add_images("input", imgs, step, dataformats='NCHW')
        # torch.Size([64, 6, 30, 30])
        output = torch.reshape(output, (-1, 3, 30, 30)) #写-1会自动计算尺寸
        writer.add_images("output", output, step)
    
        step = step + 1
    
    writer.close()
    ```

### 最大池化/下采样 Max Pooling `torch.nn.MaxPool2d`

* 函数定义

    ```python
    class torch.nn.MaxPool2d(kernel_size, stride=None, 
    padding=0, dilation=1, return_indices=False, 
    ceil_mode=False)
    ```

* 参数
  * input size: $(N,\ C,\ H_{in},\ W_{in})$
  * output 
  * ceil_mode：边缘小于kernel_size的是否要舍去

* 使用示范

    ```python
    import torch
    import torchvision
    from torch import nn
    from torch.nn import MaxPool2d
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    
    dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)
    dataloader = DataLoader(dataset, batch_size=64)
    
    class TestNetwork(nn.Module):
        def __init__(self):
            super(TestNetwork, self).__init__()
            self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)
    
        def forward(self, input):
            output = self.maxpool1(input)
            return output
    
    my_Network = TestNetwork()
    
    writer = SummaryWriter("./logs")
    step = 0
    for data in dataloader:
        imgs, targets = data
        writer.add_image("input", imgs, step, dataformats="NCHW")
        output = my_Network(imgs)
        writer.add_image("output", output, step, dataformats="NCHW")
        setp = step + 1
    
    writer.close()
    ```

### Nonlinear activation

* ReLU
  * 实现

    ```python
    import torch
    from torch import nn
    from torch.nn import ReLU
    
    class TestNetwork(nn.Module):
        def __init__(self):
            super(TestNetwork, self).__init__()
            self.relu1 = ReLU()
    
        def forward(self, input):
            output = self.relu1(input)
            return output
    
    my_Network = TestNetwork()
    print(my_Network)
    ```
  
    

### Lienar and Other laylers

* Linear layer
  * Documentaiton
    > Applies a linear transformation to the incoming data: $y=xA^T+b$

    ```python
    torch.nn.Linear(in_features, out_features, 
    bias=True, device=None, dtype=None)
    ```

  * 线性变换层的weight和bias取决于指定的in_features和out_features，通过$\mu(-\sqrt{k},\ \sqrt{k}), where\ k=\frac{1}{in\_features}$初始化
  * 应用

    ```python
    import torch
    import torchvision
    from torch import nn
    from torch.nn import Linear
    from torch.utils.data import DataLoader
    
    dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)
    
    dataloader = DataLoader(dataset, batch_size=64, drop_last=True)
    # 因为linear层制定了input feature的维度，所以要droplast掉最后一组batch，否则计算维度不匹配出错
    
    class TestNetwork(nn.Module):
        def __init__(self):
            super(TestNetwork, self).__init__()
            self.linear1 = Linear(196608, 10)
    
        def forward(self, input):
            output = self.linear1(input)
            return output
    
    my_Network = TestNetwork()
    
    for data in dataloader:
        imgs, targets = data
        print(imgs.shape)# torch.Size([64, 3, 32, 32])
        # output = torch.reshape(imgs, (1, 1, 1, -1))
        output = torch.flatten(imgs)# 展成列向量，reshape涵盖了flatten的功能
        print(output.shape)# torch.Size([196608])
    
        output = my_Network(output)
        print(output.shape)#torch.Size([10])
    ```

* Dropout layer
* Padding layer
* Normalization layer
* Recurrent layer
* Transformer layer
* Sparse layer(NLP)

### 完整的前向网络搭建（以CIFAR10-quick model为例并采用`nn.Sequential`简化代码）

<img src="CIFAR10_Model_Structure.png">

```python
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter

''' Too complicated!
class TestNetwork(nn.Module):
    def __init__(self):
        super(TestNetwork, self).__init__()
        self.conv1 = Conv2d(3, 32, 5, padding=2)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
'''


class TestNetwork(nn.Module):
    def __init__(self):
        super(TestNetwork, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )

    def forward(self, x):
        x = self.model1(x)
        return x

# 测试网络的正确性
my_Network = TestNetwork()
print(my_Network)
input = torch.ones((64, 3, 32, 32))
output = my_Network(input)
print(output.shape)

# 创建计算图检查网络结构
writer = SummaryWriter("./logs")
writer.add_graph(my_Network, input)
writer.close()
```

### Loss function and Back propagation

* L1
* MSE
* Cross-entropy

```python
import torch
from torch.nn import L1Loss
from torch import nn

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss(reduction='sum')
result = loss(inputs, targets)

loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targets)

print(result)  # tensor(2.)
print(result_mse)  # tensor(1.3333)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)  # tensor(1.1019)
```

### 优化器 Optimizer `toorch.optim`

```python
import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=1)

class TestNetwork(nn.Module):
    def __init__(self):
        super(TestNetwork, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )

    def forward(self, x):
        x = self.model1(x)
        return x


loss = nn.CrossEntropyLoss()
my_Network = TestNetwork()
optim = torch.optim.SGD(my_Network.parameters(), lr=0.01)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = my_Network(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()  # 前次梯度置零
        result_loss.backward()  # 计算反向梯度
        optim.step()  # 执行反向传播
        running_loss = running_loss + result_loss
    print(running_loss)
```

## *现有网络模型的使用及修改* `torchvision.models`：以VGG16为例

### Documentation

```python
torchvision.models.vgg16(pretrained: bool = False, 
progress: bool = True, **kwargs: Any)
```

* pretrained：是否使用训练好的参数
* progress：显示进度条

### 使用

```python
import torchvision

# VGG16是针对ImageNet数据集训练的
from torch import nn

vgg16_not_pretrained = torchvision.models.vgg16(pretrained=False)
vgg16_pretrained = torchvision.models.vgg16(pretrained=True)

print(vgg16_pretrained)

train_data = torchvision.datasets.CIFAR10('./dataset', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

# 迁移学习 Transfer Learning

vgg16_pretrained.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_pretrained)

print(vgg16_not_pretrained)
vgg16_not_pretrained[6] = nn.Linear(4096, 10)
print(vgg16_not_pretrained)
```

### 网络模型的保存与读取

* 保存

    ```python
    import torch
    import torchvision

    vgg16 = torchvision.models.vgg16(pretrained=False)

    # Method 1 保存模型结构+模型参数
    torch.save(vgg16, "vgg16_method1.pth")
    # Method 2 将模型参数保存为字典
    torch.save(vgg16.state_dict(), "vgg16_method2.pth")
    ```

* 读取

    ```python
    # Method 1
    model = torch.load("vgg16_method1.pth")
    # Method 2
    vgg16.load_state_dict()
    model = torch.load("vgg16_method2.pth")
    ```

## *完整的训练过程（以CIFAR10数据集为例）*

* 大纲：准备数据->加载数据->准备模型->设置损失函数->设置优化器->开始训练->验证->Tensorboard展示

* 分开构造Modell

```python
# Construct Neural Network
import torch
from torch import nn

class TestNetwork(nn.Module):
    def __init__(self):
        super(TestNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    my_network = TestNetwork()
    input = torch.ones((64, 3, 32, 32))
    output = my_network(input)
    print(output)

```

* 主体

```python
import torch.optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Prepare dataset
train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("The length of train dataset is:{}".format(train_data_size))
print("The length of test dataset is:{}".format(test_data_size))

# Use Dataloader to load data
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# Construct Neural Network
class TestNetwork(nn.Module):
    def __init__(self):
        super(TestNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# Loss Function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
learning_rate = 1e-2
optimizer = torch.optim.SGD(TestNetwork.parameters(), lr=learning_rate)

# Set Network Parameters
total_train_step = 0
total_test_step = 0
epoch = 10

# Tensorboard
writer = SummaryWriter("../logs_train")

# Training   
for i in range(epoch):
    print("------Strat to train #{} epoch------".format(i+1))
    for data in train_dataloader:
        imgs, targets = data
        outputs = TestNetwork(imgs)
        loss = loss_fn(outputs, targets)

        # Set up optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0: #避免无用信息
            print("# Training:{}, Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # Test
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = TestNetwork(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
    print("Loss of the test dataset:{}".format(total_test_loss))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(TestNetwork, "TestNetwork_{}.pth".format(i))
    print("Model saved.")
    

writer.close()
```

## *利用cuda进行GPU加速训练*

### 对网络模型、数据（输入）和损失函数使用`.cuda()`

```python
if torch.cuda.is_available():
    my_network.cuda()
    loss_fn = loss_fn.cuda()

    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
```

### 使用`.to(device)`

```python
# 定义训练的设备 
# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_availaable() else "cpu")
my_network.to(device)    
loss_fn.to(device)


imgs, targets = data
imgs = imgs.to(device)
targets = targets.to(device)
```

# PyTorch

Documentation：https://pytorch.org/docs/stable/index.html

## *PyTorch基础*

### 不同深度学习框架的区别

PyTorch是一个建立在Torch库之上的Python包，Torch本身就是一个科学计算框架，一开始并不支持Python，后来是由facebook将其用Python实现了

PyTorch 用来加速深度学习，它提供一种类似numpy的抽象方法来表示张量 Tensor（一种特殊设计的高效的多维数组结构），可以利用GPU来加速学习

### PyTorch结构 & 架构

下面的结构图展示了 PyTorch 主要模块之间的关系

* `torch`：类似于numpy的通用数组库，主要提供对于tensor的处理，可将tensor转换为 `torch.cuda.TensorFloat` 用来在GPU上进行计算
* `torch.autograd`：用于构建计算图并自动获取梯度的包
* `torch.nn`：具有共享层和损失函数的神经网络库
* `torch.optim`：具有通用优化算法的包，如SGD、Adam等

### PyTorch workflow

<img src="pytorch_workflow.png" width="75%">

## *Tensor数据结构*

### Tensor介绍

Tensor是PyTorch中经过专门设计的，可用于GPU计算的特殊数据结构，它和Numpy中的ndarray非常相似，两者可以共享内存，并且之间可以进行高效的转换。区别是Numpy只能用于CPU计算

### 创建Tensor

* 直接构造，支持list容器
  * `torch.tensor()`：从数据中推断数据类型
  * `torch.Tensor()`：传入数据时，默认使用全局默认dtype（FloatTensor），返回一个大小为1的张量，它是随机初始化的值
* Tensor和ndarray的转换，这两个方法都是对原对象进行操作，不会生成拷贝
  * 从ndarray中构造tensor `torch.from_numpy(array_np) `
  * 转换回ndarray `array_ts_2.numpy()`

### 修改Tensor形状

* `size()` 返回tensor形状
* `view(shape)`，修改tensor形状，修改的是原对象。常用 `view(-1)` 相当于flatten拉平数组
* `reshape(shape)` 修改tensor形状，但会创建新tensor拷贝
* `resize(shape)` 和view相似，但在size超出时会重新分配内存空间
* `squeeze` 和 `unsqueeze` 指定维度降维或升维

## *PyTorch数据处理工具箱*

PyTorch的数据处理部分的工具在Data Preparation部分已经模拟实现过，可以回顾具体的代码实现

### `torch.utils.data` 结构

<img src="torch_utils_data工具包.png" width="75%">

* `Dataset` 是一个抽象类，其他数据集都需要继承这个父类，并重写其中的两个方法 `__getitem__()` 和 `__len__()`
* `DataLoader` 定义一个新的迭代器，来为神经网络提供自动转换为tensor的mini-Batch
* `random_split` 把数据集随机拆分为给定长度的非重叠的新数据集
* `*Sampler` 多种采样函数

### torchvision 视觉处理工具包

torchvision 是独立于PyTorch的视觉处理包，需要独立安装，类似的还有用于语言处理的 torchaudio 等。torchvision包括了下面四个类

* `datasets` 提供常用视觉数据集，设计上继承自 `torch.utils.data.Dataset`，主要包括MNIST，cifar-10，ImageNet和COCO等
* `models` 提供DL中各种经典的网络结构以及预训练好的模型，比如AlexNet，VGG系列等
* `transforms` 常用的数据预处理操作，主要包括对Tenosr及PIL Image对象的操作
  * 对PIL Image对象
    * `ToTensor` 把一个取值范围是 [0, 255] 的形状为 (H,W,C) 的PIL.Image ndarray转换为取值范围是 [0, 1.0] 的形状为 (C,H,W) 的torch.FloatTensor
  * 对Tensor对象
    * `ToPILImage` 将torch.FloatTensor转换为 PIL Image ndarray
  * 可以用 `transforms.Compose([ ])` 将多个预处理连接起来
* `utils` 包含两个函数
  * `make_grid()` 能将多张图片拼接在一个网格中
  * `save_img()` 能将Tensor保存成图片

### TensorBoard可视化工具

TensorBoard 是 Google TensorFlow 的可视化工具，被 PyTorch 借用了

## *PyTorch神经网络工具箱*

<img src="torch_nn神剧网络工具包.png" width="60%">

### 构建网络以及前向传播

自定义网络需要继承 `nn.Module`

```python
class Net(nn.Module):
    def __init__(self, activation=nn.Sigmoid(),
                 input_size=1*28*28, hidden_size=100, classes=10):
        
        super(Net, self).__init__()
        self.input_size = input_size

        # Here we initialize our activation and set up our two linear layers
        self.activation = activation
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, classes)
 
    def forward(self, x):
        x = x.view(-1, self.input_size) # flatten
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x
net = Net() #实例化网络
```

一般都利用函数 `nn.sequentail()` 来搭建网络

```python
model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )

# Using Sequential with OrderedDict. This is functionally the
# same as the above code
model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))
```

### 常用的层

```python
# Affine/Linear
torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
# Batch Normalization
torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
```

### 反向传播

Thanks to the <b>autograd</b> package, we just have to define the <b>forward()</b> function. We can use any of the Tensor operations in the <b>forward()</b>  function.
 The <b>backward()</b> function (where gradients are computed through back-propagation) is automatically defined by PyTorch.

### 优化器 Optimizer

```python
# 举例
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

### 训练模型

使用for循环进行迭代训练

```python
train_loss_history = [] # loss
train_acc_history = [] # accuracy

for epoch in range(2):

    running_loss = 0.0
    correct = 0.0
    total = 0    
    # Iterating through the minibatches of the data
    for i, data in enumerate(fashion_mnist_dataloader, 0):
        # data is a tuple of (inputs, labels)
        X, y = data

        X = X.to(device)
        y = y.to(device)

        # Reset the parameter gradients  for the current  minibatch iteration 
        optimizer.zero_grad()
        
        y_pred = net(X)             # Perform a forward pass on the network with inputs
        loss = criterion(y_pred, y) # calculate the loss with the network predictions and ground Truth
        loss.backward()             # Perform a backward pass to calculate the gradients
        optimizer.step()            # Optimize the network parameters with calculated gradients
        
        # Accumulate the loss and calculate the accuracy of predictions
        running_loss += loss.item()
        _, preds = torch.max(y_pred, 1) #convert output probabilities of each class to a singular class prediction
        correct += preds.eq(y).sum().item()
        total += y.size(0)

        # Print statistics to console
        if i % 1000 == 999: # print every 1000 mini-batches
            running_loss /= 1000
            correct /= total
            print("[Epoch %d, Iteration %5d] loss: %.3f acc: %.2f %%" % (epoch+1, i+1, running_loss, 100*correct))
            train_loss_history.append(running_loss)
            train_acc_history.append(correct)
            running_loss = 0.0
            correct = 0.0
            total = 0

print('FINISH.')
```

## *PyTorch Lightning*

PyTorch Lightning 是对 PyTorch 的进一步第三方封装

Documentation: https://pytorch-lightning.rtfd.io/en/latest/

### Define a Lightning module

自定义Net要继承 `pl.lightning`，可以将最重要的网络结果、前向传播、train都写在一个class内

具体代码可参考 i2dl exercise 7 lightning_models.py

```python
class TwoLayerNet(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # This sets self.hparams the the dict or namespace
        self.save_hyperparameters(hparams)

        # We can access the parameters here
        self.model = nn.Sequential(
            nn.Linear(self.hparams.input_size,
                      self.hparams.hidden_size),
            nn.Sigmoid(),
            nn.Linear(self.hparams.hidden_size,
                      self.hparams.num_classes),
        )
    def forward(self, x):
        # flatten the image  before sending as input to the model
        N, _, _, _ = x.shape
        x = x.view(N, -1)

        x = self.model(x)

        return x
    
    def training_step(self, batch, batch_idx):
        pass
    def validation_step(self, batch, batch_idx):
        pass
    def validation_epoch_end(self, outputs):
        pass
    def configure_optimizers(self):
        pass
    def visualize_predictions(self, images, preds, targets):
        pass
```

### Data pipeline

* Define Dataset：自定义的 DataModule 要继承 `pl.LightningDataModule`

  ```python
  class FashionMNISTDataModule(pl.LightningDataModule):
      def __init__(self, batch_size=4):
          super().__init__()
          self.batch_size = batch_size
      def prepare_data(self):
          # Define the transform
          transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))])
          # Download the Fashion-MNIST dataset
          fashion_mnist_train_val = torchvision.datasets.FashionMNIST(root='../datasets', train=True,
                                                                     download=True, transform=transform)
          self.fashion_mnist_test = torchvision.datasets.FashionMNIST(root='../datasets', train=False,
                                                                   download=True, transform=transform)
          # Apply the Transforms
          transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))])
          # Perform the training and validation split
          self.train_dataset, self.val_dataset = random_split(
              fashion_mnist_train_val, [50000, 10000])
  ```

* Define dataloader in `pl.LightningDataModule` for each data split

  ```python
      def train_dataloader(self):
          return DataLoader(self.train_dataset, batch_size=self.batch_size)
  
      def val_dataloader(self):
          return DataLoader(self.val_dataset, batch_size=self.batch_size)
  
      def test_dataloader(self):
          return DataLoader(self.fashion_mnist_test, batch_size=self.batch_size)
  ```

# Tensorflow

# MindSpore



