import os, argparse, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.backends.cudnn import deterministic
from PIL import Image
from ResNetModel import resnet50
import torchmetrics


# 设置训练使用的设备
if torch.cuda.is_available():
    硬件设备 = torch.device("cuda:0")
    # 保证每次返回的卷积算法将是确定的，如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 为每层搜索适合的卷积算法实现，加速计算
else:
    硬件设备 = torch.device("cpu")
print("使用设备", 硬件设备)


# 设置一个参数解析器
命令行参数解析器 = argparse.ArgumentParser(description='使用3维数据进行自监督训练，之后再用2维图像微调')
# 添加无标签数据训练时的参数

命令行参数解析器.add_argument('--model_path', default="Batchsize-6_Lr-0.01_Acc-81.pth", type=str, help="输入模型的路径")
命令行参数解析器.add_argument('--batch_size', default=6, type=int, help="批量大小")
无标签训练命令行参数 = 命令行参数解析器.parse_args() # 获取命令行传入的参数


def 灰度图读取(path):
    # datasets.ImageFolder默认是以RGB形式读图像
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L') # 以灰度形式读


def 训练模型(命令行参数):
    随机图像变换 = {
        "测试集": transforms.Compose([
            transforms.ToTensor(),
            # 三通道归一化
            transforms.Normalize([0.4914], [0.2023])])
    }

    # 加载训练数据集和测试数据集
    有标签测试数据集 = datasets.ImageFolder(root="../Dataset/FMLabeledDataset/Validate", transform=随机图像变换["测试集"], loader=灰度图读取)
    有标签测试数据 = torch.utils.data.DataLoader(有标签测试数据集, batch_size=命令行参数.batch_size, shuffle=True,
                                          num_workers=0, pin_memory=True)

    分类模型 = resnet50(2).to(硬件设备)  # 生成模型，需传入分类数目
    权重路径 = 命令行参数.model_path
    assert os.path.exists(权重路径), "模型权重{}不存在.".format(权重路径)
    分类模型.load_state_dict(torch.load(权重路径, map_location=硬件设备))  # 加载模型参数
    召回率 = torchmetrics.Recall(average='none', num_classes=2).to(硬件设备)
    精确度 = torchmetrics.Precision(average='none', num_classes=2).to(硬件设备)


    分类模型.eval()  # 每一批数据训练完成后测试模型效果
    测试正确的总数目 = 0
    # 下方代码块不反向计算梯度
    with torch.no_grad():
        # 每一批数据训练。enumerate可以在遍历元素的同时输出元素的索引
        测试循环 = tqdm(enumerate(有标签测试数据), total=len(有标签测试数据), leave=True)
        for 当前批次, (图像数据, 标签) in 测试循环:
            图像数据, 标签 = 图像数据.to(硬件设备), 标签.to(硬件设备)
            测试集预测概率 = 分类模型(图像数据)
            # torch.max(a,1)返回行最大值和列索引。结果中的第二个张量是列索引
            预测类别 = torch.max(测试集预测概率, dim=1)[1]
            测试正确的总数目 += torch.eq(预测类别, 标签).sum().item()  # 累加每个批次预测正确的数目
            召回率(测试集预测概率.argmax(1), 标签)
            精确度(测试集预测概率.argmax(1), 标签)
    总召回率 = 召回率.compute()
    总精确度 = 精确度.compute()
    当前迭代周期测试准确率 = 测试正确的总数目 / len(有标签测试数据集)
    print("测试准确率: %.4f" % 当前迭代周期测试准确率)
    print("召回率", 总召回率)
    print("精确度", 总精确度)


if __name__ == '__main__':
    训练模型(无标签训练命令行参数)