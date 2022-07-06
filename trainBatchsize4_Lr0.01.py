import torch, argparse
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from tqdm import tqdm
from torch.backends.cudnn import deterministic
from ResNetModel import resnet50
from PIL import Image


# 设置训练使用的设备
if torch.cuda.is_available():
    硬件设备 = torch.device("cuda:0")
    # 保证每次返回的卷积算法将是确定的，如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 为每层搜索适合的卷积算法实现，加速计算
else:
    硬件设备 = torch.device("cpu")
print("训练使用设备", 硬件设备)


# 设置一个参数解析器
命令行参数解析器 = argparse.ArgumentParser(description='使用3维数据进行自监督训练，之后再用2维图像微调')
# 添加无标签数据训练时的参数
命令行参数解析器.add_argument('--max_epoch', type=int, default=6000, help="无标签训练的最大迭代周期")
命令行参数解析器.add_argument('--batch_size', default=4, type=int, help="有标签数据训练时的批量大小")
无标签训练命令行参数 = 命令行参数解析器.parse_args() # 获取命令行传入的参数


def 灰度图读取(path):
    # datasets.ImageFolder默认是以RGB形式读图像
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L') # 以灰度形式读


def 训练模型(命令行参数):
    随机图像变换 = {
        "训练集": transforms.Compose([
            # TODO 选择合适的图像大小。是否需要随机高斯滤波
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
            transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
            # 修改亮度、对比度和饱和度
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # 随机应用添加的各种图像变换
            transforms.ToTensor(),  # 转换为张量且维度是[C, H, W]
            # 三通道归一化
            transforms.Normalize([0.4914], [0.2023])]),
        "测试集": transforms.Compose([
            transforms.ToTensor(),
            # 三通道归一化
            transforms.Normalize([0.4914], [0.2023])])
    }

    # 加载训练数据集和测试数据集
    有标签训练数据集 = datasets.ImageFolder(root="../Dataset/FMLabeledDataset/train", transform=随机图像变换["训练集"], loader=灰度图读取)
    有标签训练数据 = torch.utils.data.DataLoader(有标签训练数据集, batch_size=命令行参数.batch_size, shuffle=True,
                                          num_workers=0, pin_memory=True)
    有标签测试数据集 = datasets.ImageFolder(root="../Dataset/FMLabeledDataset/Validate", transform=随机图像变换["测试集"], loader=灰度图读取)
    有标签测试数据 = torch.utils.data.DataLoader(有标签测试数据集, batch_size=命令行参数.batch_size, shuffle=False,
                                          num_workers=0, pin_memory=True)

    分类模型 = resnet50(2).to(硬件设备) # 生成模型，如果使用预训练权重这里就不需要传入分类数目
    损失函数 = torch.nn.CrossEntropyLoss()
    优化器 = torch.optim.Adam(分类模型.parameters(), lr=0.005, weight_decay=1e-6)
    学习率调整器 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(优化器, 5, 2)

    # 开始训练
    保存模型数量 = 0
    for 当前训练周期 in range(1, 命令行参数.max_epoch + 1):
        分类模型.train()
        当前周期全部损失 = 0.0
        训练正确总数目 = 0
        # 每一批数据训练。enumerate可以在遍历元素的同时输出元素的索引
        训练循环 = tqdm(enumerate(有标签训练数据), total=len(有标签训练数据), leave=True)
        for 当前批次, (图像数据, 标签) in 训练循环:
            图像数据, 标签 = 图像数据.to(硬件设备), 标签.to(硬件设备)
            训练集预测概率 = 分类模型(图像数据)
            预测类别 = torch.max(训练集预测概率, dim=1)[1]
            训练正确总数目 += torch.eq(预测类别, 标签).sum().item()  # 累加每个批次预测正确的数目
            训练损失 = 损失函数(训练集预测概率, 标签)  # 每一批的训练损失
            优化器.zero_grad()
            训练损失.backward()
            优化器.step()
            学习率调整器.step(当前训练周期 + 当前批次 / len(有标签训练数据))  # 调整学习率
            当前周期全部损失 += 训练损失.detach().item()
        当前迭代周期训练准确率 = 训练正确总数目 / len(有标签训练数据集)

        分类模型.eval()  # 每一批数据训练完成后测试模型效果
        测试正确的总数目 = 0
        # 下方代码块不反向计算梯度
        with torch.no_grad():
            测试循环 = tqdm(enumerate(有标签测试数据), total=len(有标签测试数据), leave=True)
            for 当前批次, (图像数据, 标签) in 测试循环:
                图像数据, 标签 = 图像数据.to(硬件设备), 标签.to(硬件设备)
                测试集预测概率 = 分类模型(图像数据)
                # torch.max(a,1)返回行最大值和列索引。结果中的第二个张量是列索引
                预测类别 = torch.max(测试集预测概率, dim=1)[1]
                测试正确的总数目 += torch.eq(预测类别, 标签).sum().item()  # 累加每个批次预测正确的数目
        当前迭代周期测试准确率 = 测试正确的总数目 / len(有标签测试数据集)
        print("[周期%d] 平均训练损失:%.8f 训练准确率:%.4f 测试准确率:%.4f 学习率%.8f" % (
            当前训练周期, 当前周期全部损失 / len(有标签训练数据集) * 命令行参数.batch_size, 当前迭代周期训练准确率, 当前迭代周期测试准确率, 优化器.param_groups[0]['lr']))

        if 当前迭代周期训练准确率 >= 0.82 and 当前迭代周期测试准确率 >= 0.82:
            保存模型数量 += 1
            if 保存模型数量 > 12:
                break
            torch.save(分类模型.state_dict(), str(当前训练周期) + "ModelBatch4Lr001.pth")


if __name__ == '__main__':
    训练模型(无标签训练命令行参数)