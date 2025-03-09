import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F

# 假设classifier是你训练好的分类器模型
# 假设gaussian_diffusion.sample返回的图片已经在[0, 1]范围内

def evaluate_and_save_images(test_loader, model, classifier, device, save_image_name, log_num=10):
    all_generated_images = []  # 存储所有生成的图片
    all_labels = []  # 存储所有标签
    all_preds = []  # 存储所有预测标签
    all_preds_labels = []  # 存储图片上显示的真实标签和预测标签
    
    model.eval()  # 将生成模型切换到评估模式
    classifier.eval()  # 将分类器切换到评估模式
    
    with torch.no_grad():  # 评估时不需要计算梯度
        for j, (img, label) in enumerate(test_loader):
            img, label = img.to(device), label.to(device)

            # 使用生成模型生成图片
            generated_images = gaussian_diffusion.sample(model, image_size, batch_size=img.size(0), channels=in_channels, img=img, label=label)
            imgs = generated_images[-1].reshape(img.size(0), in_channels, image_size, image_size)

            all_generated_images.append(imgs)  # 保存生成的图片
            all_labels.append(label)  # 保存真实标签
            
            # 分类器预测
            outputs = classifier(imgs)
            _, preds = torch.max(outputs, 1)  # 获取预测的标签
            all_preds.append(preds)  # 保存预测标签

            # 将真实标签和预测标签保存到一个列表中
            all_preds_labels.append((label, preds))

            # 每隔log_num次保存一个batch
            if j % log_num == 0:
                save_image(imgs, f'adv_photos/{save_image_name}_batch{j}.png', nrow=4, normalize=True)

        # 将所有生成的图片拼接起来用于评估
        all_generated_images = torch.cat(all_generated_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_preds = torch.cat(all_preds, dim=0)

        # 计算准确率
        correct = (all_preds == all_labels).sum().item()
        total = all_labels.size(0)
        accuracy = correct / total

        print(f'Accuracy: {accuracy * 100:.2f}%')

        # 可视化并保存真实标签和预测标签
        for i in range(min(10, len(all_preds_labels))):  # 可视化前10张图片
            real_label, pred_label = all_preds_labels[i]
            img = all_generated_images[i].cpu().numpy().transpose(1, 2, 0)  # 转换为HWC格式

            # 在图片上标注真实标签和预测标签
            plt.imshow(img)
            plt.title(f"True: {real_label.item()} | Pred: {pred_label.item()}")
            plt.axis('off')
            plt.savefig(f'heatmap_photos/{save_image_name}_eval_{i}.png')
            plt.close()

    return accuracy