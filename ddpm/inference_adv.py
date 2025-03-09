import torch
import os
import argparse
from net import UNetModel
from diffusion_adv import GaussianDiffusion
from torchvision.utils import save_image

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import random

# import sys
# resnet_path = os.path.expanduser('~/DiffusionModel/pytorch-cifar/models')
# sys.path.append(resnet_path)

from classifier import Classifier


# from resnet import ResNet50


transform = transforms.Compose([
    # transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.1307], std=[0.3081])  # 单通道的标准化
])

# train_dataset = datasets.MNIST(root='root/advdiff_impl/data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='~/advdiff_impl/data', train=False, download=False, transform=transform)

# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)

def inference(args):
    batch_size = args.batch_size
    timesteps = args.timesteps
    datasets_type = args.datasets_type

    if datasets_type:
        in_channels = 3
        image_size = 32
        save_image_name = "cifar10"
    else:
        in_channels = 1
        image_size = 28
        save_image_name = "mnist"

    # define model and diffusion
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = UNetModel(
        in_channels=in_channels,
        model_channels=96,
        out_channels=in_channels,
        channel_mult=(1, 2, 2),
        attention_resolutions=[]
    )

    map_location = None if torch.cuda.is_available() else lambda storage, loc: storage
    model.to(device)
    # model = torch.nn.DataParallel(model)
    model.load_state_dict((torch.load(args.pth_path, map_location=map_location)))
    model.eval()

    # classifier = ResNet50()
    # classifier = classifier.to(device)
    # classifier = torch.nn.DataParallel(classifier)
    # checkpoint = torch.load('ckpt.pth')
    # classifier.load_state_dict(checkpoint['net'])
    # classifier.eval()

    classifier = Classifier().to(device)
    classifier.train_model(device, test_loader, test_loader, epochs=10)
    # classifier.eval()

    gaussian_diffusion = GaussianDiffusion(timesteps=timesteps, classifier = classifier, device = device)
    # generated_images = gaussian_diffusion.sample(model, image_size, batch_size=batch_size, channels=in_channels)

    

    
    # generate new images
    if not os.path.exists("adv_photos"):
        os.mkdir("adv_photos")

    def get_samples_per_class(n, test_loader):
        # 用于存储每个类别的样本
        class_samples = {i: [] for i in range(10)}
        
        # 遍历测试数据集，将每个样本按类别存储
        for images, labels in test_loader:
            for i in range(len(labels)):
                class_samples[labels[i].item()].append(images[i])
        
        # 从每个类别中随机选择5张样本
        selected_samples = []
        selected_labels = []
        
        for class_idx, samples in class_samples.items():
            if len(samples) >= n:  # 确保每个类别有至少5张样本
                selected_samples.extend(random.sample(samples, n))
                selected_labels.extend([class_idx] * n)
        
        # 将选中的样本转化为TensorDataset
        adv_ori_images = torch.stack(selected_samples)
        adv_labels = torch.tensor(selected_labels)
    
        return adv_ori_images, adv_labels

    # for j, (img, label) in enumerate(test_loader):
    img, label = get_samples_per_class(1, test_loader)
    img, label = img.to(device), label.to(device)
    generated_images = gaussian_diffusion.sample(model, image_size, batch_size=batch_size, channels=in_channels, img=img, label=label)
    imgs = generated_images[-1].reshape(batch_size, in_channels, image_size, image_size)
    img = torch.tensor(imgs)

    
    pred = classifier(img.detach().to(device))
    print(pred.argmax(dim=1))


    # if j % 30 == 0:
    save_image(torch.tensor(img), f'adv_photos/{save_image_name}_batch.png', nrow=10, normalize=True)
    
    imgs_time = []
    for n_row in range(10):
        for n_col in range(10):
            t_idx = (timesteps // 10) * n_col if n_col < 9 else -1
            img = torch.tensor(generated_images[t_idx][n_row].reshape(in_channels, image_size, image_size))
            imgs_time.append(img)

    imgs = torch.stack(imgs_time).reshape(-1, in_channels, image_size, image_size)
    save_image(imgs, f'adv_photos/{save_image_name}_2.png', nrow=10, normalize=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', default=10, type=int, help="batch size")
    parser.add_argument('-d', '--datasets_type', default=0, type=int, help="datasets type,0:MNISI,1:cifar-10")
    parser.add_argument('-t', '--timesteps', default=500, type=int, help="timesteps")
    parser.add_argument('-p', '--pth_path', default="models/mnist-500-20-0.0005.pth", type=str, help="path of pth file")

    args = parser.parse_args()
    print(args)
    inference(args)
