import torch, argparse, os, random
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.backends.cudnn import deterministic
from ResNetModel import resnet50


# 模型训练
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用 {}.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    net = resnet34()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnet34-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 5)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 3
    best_acc = 0.0
    save_path = './resNet34.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()


def 有标签训练(命令行参数):
    #  init seed 初始化随机种子
    全部随机数种子 = 222

    # 下面似乎都是控制生成相同的随机数
    random.seed(全部随机数种子)
    np.random.seed(全部随机数种子)  # todo
    torch.manual_seed(全部随机数种子)
    torch.cuda.manual_seed_all(全部随机数种子)
    torch.cuda.manual_seed(全部随机数种子)
    np.random.seed(全部随机数种子)  # todo

    # 禁止哈希随机化，使实验可复现
    os.environ['PYTHONHASHSEED'] = str(全部随机数种子)

    # 设置训练使用的设备
    if torch.cuda.is_available():
        硬件设备 = torch.device("cuda:0")
        # 保证每次返回的卷积算法将是确定的，如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的。
        torch.backends.cudnn.deterministic = True
        if torch.backends.cudnn.deterministic:
            print("确定卷积算法")
        torch.backends.cudnn.benchmark = False  # 为每层搜索适合的卷积算法实现，加速计算
    else:
        硬件设备 = torch.device("cpu")
    print("训练使用设备", 硬件设备)

    随机图像变换 = {
        "训练集": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "测试集": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    # 加载训练数据集和测试数据集
    数据集路径 = "LabeledDataset"
    有标签训练数据集 = datasets.ImageFolder(root=os.path.join(数据集路径, "Train"), transform=随机图像变换["测试集"])

    类别 = 有标签训练数据集.class_to_idx

    # win可能多线程报错，num_workers最多和CPU的超线程数目相同，若报错设为0
    # todo 线程数 = min([os.cpu_count(), 命令行参数.batch_size if 命令行参数.batch_size > 1 else 0, 8])  # number of workers
    有标签训练数据 = torch.utils.data.DataLoader(有标签训练数据集, batch_size=命令行参数.labeled_data_batch_size, shuffle=True,
                                          num_workers=0, pin_memory=True)
    有标签测试数据集 = datasets.ImageFolder(root=os.path.join(数据集路径, "Validate"), transform=随机图像变换["测试集"])
    有标签测试数据 = torch.utils.data.DataLoader(有标签测试数据集, batch_size=命令行参数.labeled_data_batch_size, shuffle=False,
                                          num_workers=0, pin_memory=True)

    分类模型 = resnet50()  # 生成模型，如果使用预训练权重这里就不需要传入分类数目
    残差模型权重路径 = "Weight/resnet50-19c8e357.pth"
    assert os.path.exists(残差模型权重路径), "残差模型权重{}不存在.".format(残差模型权重路径)
    分类模型.load_state_dict(torch.load(残差模型权重路径, map_location=硬件设备) )  # 加载模型参数

    ''' 冻结残差预训练权重
    for 参数 in 分类模型.parameters():
        参数.requires_grad = False
    '''
    输入通道数目 = 分类模型.fc.in_features
    分类模型.fc = nn.Linear()
    分类模型.to(硬件设备)
    损失函数 = torch.nn.CrossEntropyLoss()
    优化器 = torch.optim.Adam(分类模型.全连接.parameters(), lr=1e-3, weight_decay=1e-6)

    最高测试准确率 = 0.0
    # 开始训练
    for 当前训练周期 in range(1, 命令行参数.labeled_train_max_epoch + 1):
        分类模型.train()
        当前周期全部损失 = 0.0
        # 每一批数据训练。enumerate可以在遍历元素的同时输出元素的索引
        训练循环 = tqdm(enumerate(有标签训练数据), total=len(有标签训练数据), leave=True)
        for 当前批次, (图像数据, 标签) in 训练循环:
            图像数据, 标签 = 图像数据.to(硬件设备), 标签.to(硬件设备)
            训练集预测概率 = 分类模型(图像数据)
            训练损失 = 损失函数(训练集预测概率, 标签)  # 每一批的训练损失
            优化器.zero_grad()
            训练损失.backward()
            优化器.step()
            当前周期全部损失 += 训练损失.detach().item()
            训练循环.desc = "训练迭代周期 [{}/{}] 当前损失：{:.8f}".format(当前训练周期, 命令行参数.labeled_train_max_epoch,
                                                            训练损失.detach().item())  # 设置进度条描述

        # 记录每个周期的平均每批损失值
        with open(os.path.join("Weight", "stage2_loss.txt"), "a") as f:
            f.write(str(当前周期全部损失 / len(有标签训练数据集) * 命令行参数.labeled_data_batch_size) + "\n")

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
                测试循环.desc = "测试迭代周期 [{}/{}]".format(当前训练周期, 命令行参数.labeled_train_max_epoch)  # 设置进度条描述
        当前迭代周期测试准确率 = 测试正确的总数目 / len(有标签测试数据集)
        print("[迭代周期 %d] 平均训练损失: %.8f  测试准确率: %.4f" % (
            当前训练周期, 当前周期全部损失 / len(有标签训练数据集) * 命令行参数.labeled_data_batch_size, 当前迭代周期测试准确率))
        with open(os.path.join("Weight", "TestAccuracy .txt"), "a") as f:
            f.write(str(当前迭代周期测试准确率) + "\n")

        # 以最高测试准确率作为模型保存的标准
        if 当前迭代周期测试准确率 > 最高测试准确率:
            最高测试准确率 = 当前迭代周期测试准确率
            torch.save(分类模型.state_dict(), os.path.join("Weight", "LabeledModel" + ".pth"))
