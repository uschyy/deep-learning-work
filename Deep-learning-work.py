import io
import math, json
import numpy as np
import pandas as pd
from PIL import Image

import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt
import paddle
import paddle.nn.functional as F
import paddle.vision.transforms as T
from paddle.io import DataLoader, Dataset

import warnings
warnings.filterwarnings("ignore")

#读取数据集标注

# 读取json文件
train_json = pd.read_json('C:\\Users\\15982\\PycharmProjects\\pythonProject4\\deep_work\\DEEP_XUN\\train_dataset\\train.json')

# 对json文件中annotations列中的“\\”替换成“//”，保存在train_json中
train_json['filename'] = train_json['annotations'].apply(lambda x: x['filename'].replace('\\', '//'))

# print("train_json:",train_json)

# 提取annotations中的period和weather两列数据，，并作为新的列放进train_json中
train_json['period'] = train_json['annotations'].apply(lambda x: x['period'])
train_json['weather'] = train_json['annotations'].apply(lambda x: x['weather'])

# 查看前5行
# train_json.head()
# print(train_json.head())


# 用factorize将标签进行编码，这里需要记住编码的次序。
train_json['period'], period_dict = pd.factorize(train_json['period'])
train_json['weather'], weather_dict = pd.factorize(train_json['weather'])
#将period和weather进行编码便于后期的分类



#统计标签
train_json['period'].value_counts()

train_json['weather'].value_counts()

print(train_json['period'].value_counts())
print('---------------------------------------')
print(train_json['weather'].value_counts())

# 自定义数据集
class WeatherDataset(Dataset):
    def __init__(self, df):
        super(WeatherDataset, self).__init__()
        self.df = df
        # WeatherDataSet继承自DateSet 然后df=传过来的参数
        # 创建一个可调用的Compose对象，它将依次调用每个给定的 transforms
        self.transform = T.Compose([
            # 大小调整、随机裁剪、旋转、水平/垂直翻转以及将图像转换为张量和归一化。
            T.Resize(size=(340, 340)),
            T.RandomCrop(size=(256, 256)),
            T.RandomRotation(10),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5)
        ])

    # 设置增加图像的方法
    def __getitem__(self, index):
        file_name = self.df['filename'].iloc[index]
        # print(file_name)
        file_name='C://Users//15982//PycharmProjects//pythonProject4//deep_work//DEEP_XUN//train_dataset//'+file_name
        # print(file_name)

        img = Image.open(file_name)
        img = self.transform(img)
        return img, \
            paddle.to_tensor(self.df['period'].iloc[index]), \
            paddle.to_tensor(self.df['weather'].iloc[index])

    def __len__(self):
        return len(self.df)



# 加载数据集和验证集
# 训练集
# 留500张进行验证
# 参数1传入的数据集，参数二每一批有多少样本,在每个epoch开始的时候，对数据进行重新排序
# 训练集构建
train_dataset = WeatherDataset(train_json.iloc[:-500])
print(train_json.iloc[:-500])
# 训练数据器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 验证集构建
val_dataset = WeatherDataset(train_json.iloc[-500:])
# 验证数据器
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)


from paddle.vision.models import resnet18


# 选择resnet18模型
class WeatherModel(paddle.nn.Layer):
    def __init__(self):
        super(WeatherModel, self).__init__()
        backbone = resnet18(pretrained=True)  # 加载CNN网络模型
        backbone.fc = paddle.nn.Identity()  # 定义全连接层
        self.backbone = backbone

        # 分类1
        self.fc1 = paddle.nn.Linear(512, 4)  # 对于时间的分类，根据统计的得到的数据种类有4种

        # 分类2
        self.fc2 = paddle.nn.Linear(512, 3)  # 对于天气的分类，根据统计的得到的数据种类有3种

    # 定义前向激活函数
    def forward(self, x):
        out = self.backbone(x)

        # 同时完成类别1 和 类别2 分类
        logits1 = self.fc1(out)
        logits2 = self.fc2(out)
        return logits1, logits2
# 创建网络
model = WeatherModel()
model(paddle.to_tensor(np.random.rand(10, 3, 256, 256).astype(np.float32)))
# torch.save(model.state_dict(), 'model_fen.pt')

# 定义损失函数和优化器
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.0001)
criterion = paddle.nn.CrossEntropyLoss()

Train_Loss, Val_Loss = [], []   #两个损失
Train_ACC1, Train_ACC2 = [], [] #两个训练正确率
Val_ACC1, Val_ACC2 = [], []     #两个验证正确率
epoch_train_list=[]                   #轮数
epoch_val_list=[]
for epoch in range(15):
    print(epoch)
    # 模型训练
    model.train()
    for i, (x, y1, y2) in enumerate(train_loader):
        # 得到天气和时间的预测
        pred1, pred2 = model(x)
        # print('training...')
        # print(pred1,pred2)
        # 类别1的loss + 类别2的loss为总共的loss，也就是天气和时间的损失
        loss = criterion(pred1, y1) + criterion(pred2, y2)
        # 添加损失，方便后续可视化
        Train_Loss.append(loss.item())
        # 反向传播
        loss.backward()
        #更新权重
        optimizer.step()
        #清除梯度
        optimizer.clear_grad()
        # print(Train_Loss[-1])
        # 记录正确率
        Train_ACC1.append((pred1.argmax(1) == y1.flatten()).numpy().mean())
        Train_ACC2.append((pred2.argmax(1) == y2.flatten()).numpy().mean())
        # print(Train_ACC1)
        # print(i)
        epoch_train_list.append(epoch)
    # 模型验证
    model.eval()
    # 读取验证集
    for i, (x, y1, y2) in enumerate(val_loader):
        pred1, pred2 = model(x)#将测试集中的数据丢入训练好的模型进行预测
        loss = criterion(pred1, y1) + criterion(pred2, y2)#计算天气分类和时间分类的损失
        Val_Loss.append(loss.item())#得到每次验证的损失，方便验证集的可视化
        Val_ACC1.append((pred1.argmax(1) == y1.flatten()).numpy().mean())#
        Val_ACC2.append((pred2.argmax(1) == y2.flatten()).numpy().mean())#
        epoch_val_list.append(i)

    if epoch % 1 == 0:#每10轮打印一次损失和正确率
        # print(epoch)
        print(f'Loss {np.mean(Train_Loss):3.5f}/{np.mean(Val_Loss):3.5f}')
        print(f'period.ACC {np.mean(Train_ACC1):3.5f}/{np.mean(Val_ACC1):3.5f}')
        print(f'weather.ACC {np.mean(Train_ACC2):3.5f}/{np.mean(Val_ACC2):3.5f}')

plt.plot(epoch_train_list,Train_Loss)
plt.xlabel('number')
plt.ylabel('Loss')
plt.title('Train_SUM_LOSS')
plt.show()

plt.plot(epoch_val_list,Val_ACC1)
plt.xlabel('number')
plt.ylabel('acc')
plt.title('ValAcc_1')
plt.show()

plt.plot(epoch_val_list,Val_ACC2)
plt.xlabel('number')
plt.ylabel('acc')
plt.title('ValAcc_2')
plt.show()
# 将预测的东西写入对应的格式当中，用于上传结果
import glob

#获取测试集数据路径
test_df = pd.DataFrame({'filename': glob.glob('train_dataset//test_images/*.jpg')})

# 所有标签都是未知的
test_df['period'] = 0
test_df['weather'] = 0

#排序
test_df = test_df.sort_values(by='filename')

# 读取测试集
test_dataset = WeatherDataset(test_df)

# 构建测试集数据器
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 评估模式
model.eval()

# 存储预测结果
period_pred = []
weather_pred = []
epoch_test_list=[]
#测试集进行预测
for i, (x, y1, y2) in enumerate(test_loader):
    pred1, pred2 = model(x)
    period_pred += period_dict[pred1.argmax(1).numpy()].tolist()
    weather_pred += weather_dict[pred2.argmax(1).numpy()].tolist()
    print("period_pred:",period_pred)
    print("weather_pred:",weather_pred)



test_df['period'] = period_pred
test_df['weather'] = weather_pred

# 初始化一个json对象，用与存储最终的提交结果
submit_json = {
    'annotations':[]
}

#生成测试集结果
for row in test_df.iterrows():
    submit_json['annotations'].append({
        'filename': 'test_images\\' + row[1].filename.split('/')[-1],
        'period': row[1].period,
        'weather': row[1].weather,
    })

with open('submit.json', 'w') as up:
    json.dump(submit_json, up)