import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
from torchvision.utils import save_image


def linear_beta_schedule(timesteps):
    """
    beta schedule
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            timesteps=1000,
            beta_schedule='linear',
            classifier = None,
            device = None
    ):
        super(GaussianDiffusion, self).__init__()
        self.timesteps = timesteps

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1 = (
                self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * torch.sqrt(self.alphas)
                / (1.0 - self.alphas_cumprod)
        )
        if classifier is not None:
            self.classifier = classifier
        if device is not None:
            self.device = device
        self.target_layer = self.classifier.conv2

    def forward(self, x):
        return self.classify(x)

    def classify(self, x):

        return self.classifier(x)
            
    def _extract(self, a, t, x_shape):
        # get the param of given timestep t
        # print(t.shape)
        # print(t)
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    def q_sample(self, x_start, t, noise=None):
        # forward diffusion (using the nice property): q(x_t | x_0)
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_mean_variance(self, x_start, t):
        # Get the mean and variance of q(x_t | x_0).
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_start, x_t, t):
        # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
        posterior_mean = (
                self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        # compute x_0 from x_t and pred noise: the reverse of `q_sample`
        return (
                self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def p_mean_variance(self, model, x_t, t, clip_denoised=True):
        # compute predicted mean and variance of p(x_{t-1} | x_t)
        # predict noise using model
        pred_noise = model(x_t, t)
        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, model, x_t, t, clip_denoised=True, heatmap = None, label = None, iterations = 5, oriimg = None, noise = None):
        # denoise_step: sample x_{t-1} from x_t and pred_noise
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # 攻击条件判断
        if t[0] < self.timesteps * 0.5 and t[0] > self.timesteps * 0.1:
            
            # 获取对抗参数
            sigma = 0.2  # 扰动强度系数
            beta_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)  # 噪声调度系数
            # 逐样本处理
            for batch_idx in range(x_t.size(0)):
                # 初始化对抗样本
                z_t = pred_img[batch_idx].unsqueeze(0).detach().clone()  # 保持维度 [1,C,H,W]
                z_0 = z_t.clone()
    
                # 迭代扰动
                if t.min() > self.timesteps * 0:
                    for _ in range(iterations):
                        # 前向分类
                        z_t.requires_grad_(True)
                        logits = self.classifier(z_t)
                        pred_class = logits.argmax()
                        self.classifier.zero_grad()
                        
                        # 停止条件：分类错误
                        if pred_class != label[batch_idx]:
                            print(pred_class, label[batch_idx])
                            break
        
                        # 计算对抗梯度
                        loss = F.cross_entropy(logits, label[batch_idx].unsqueeze(0))
                        self.classifier.zero_grad()
                        loss.backward()
        
                        # 符号梯度攻击
                        with torch.no_grad():
                            grad = z_t.grad.sign()
                            perturbation = grad
        
                            # 投影梯度约束（确保在epsilon球内）
                            delta = (z_t + perturbation) - z_0
                            delta_norm = torch.norm(delta, p=2)
                            # if delta_norm > sigma * beta_t[batch_idx]:
                            #     perturbation = perturbation / delta_norm.clamp(min=1e-8) * sigma * beta_t[batch_idx]
                            if(t.min() < self.timesteps * 0.3):
                                while(delta_norm > sigma * beta_t[batch_idx] and perturbation.max() > 1e-2):
                                    perturbation /= 2
                                    delta = (z_t + perturbation) - z_0
                                    delta_norm = torch.norm(delta, p=2)
                            z_t = z_t + perturbation
    
                # 热力图融合
                
                with torch.no_grad():
                    # 获取正向过程采样
                    x_sample = self.q_sample(oriimg[batch_idx].unsqueeze(0), t[batch_idx].unsqueeze(0), noise = noise[batch_idx])
                    
                    inverse_heatmap = 1 - heatmap[batch_idx]
                    fused_img = (x_sample * heatmap[batch_idx] * beta_t[batch_idx] + z_t * inverse_heatmap) if t.min() > self.timesteps * 0 else z_t
    
                    pred_img[batch_idx] = fused_img.squeeze(0)
        return pred_img

    # @torch.no_grad()
    def p_sample_loop(self, model, shape, oriimg = None, label = None):
        # denoise: reverse diffusion
        batch_size = shape[0]
        device = self.device
        # start from pure noise (for each example in the batch)

        heatmap = self.generate_grad_cam_heatmap(oriimg, label)
        noise = torch.randn_like(oriimg)
        img = self.q_sample(oriimg, torch.full((batch_size,), self.timesteps-1, device=device, dtype=torch.long), noise = noise)
            
        imgs = []
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self.p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long), oriimg=oriimg, heatmap=heatmap, label = label, noise = noise)
            imgs.append(img.detach().cpu().numpy())
        return imgs

    # @torch.no_grad()
    def sample(self, model, image_size, batch_size=8, channels=3, img = None, label = None):
        # sample new images
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), oriimg = img, label = label)

    def train_losses(self, model, x_start, t):
        # compute train losses
        # generate random noise
        noise = torch.randn_like(x_start)
        # get x_t
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = model(x_noisy, t)
        loss = F.mse_loss(noise, predicted_noise)
        return loss


    def generate_grad_cam_heatmap(self, x, class_idx=None):
        # 确保输入梯度追踪
        self.classifier.eval()
        x = x.detach().requires_grad_(True)
        # print(x.shape, x)
        # 获取目标层（示例为分类器的第二个卷积层）
        target_layer = self.target_layer
        activations = []
        gradients = []
        print(x[0])
        # 修正后的钩子定义
        def forward_hook(module, input, output):
            activations.append(output)  # 保存激活值时解除梯度追踪
    
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])  # 正确获取输出梯度
    
        # 注册钩子
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)
    
        # print(f"x requires_grad: {x.requires_grad}")  # 应为 True
        logits = self(x)
        # print(f"logits requires_grad: {logits.requires_grad}")  # 应为 True
        
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
    
        # 创建one-hot梯度引导
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, class_idx.unsqueeze(1), 1)
    
        # 反向传播（关键修正步骤）
        self.classifier.zero_grad()
        logits.backward(gradient=one_hot, retain_graph=True)
    
        # 立即移除钩子
        forward_handle.remove()
        backward_handle.remove()
    
        # 检查梯度是否捕获成功
        if len(gradients) == 0 or len(activations) == 0:
            raise RuntimeError("未能捕获梯度或激活值，请检查钩子注册")
    
        # Grad-CAM计算
        activation = activations[0]
        grad = gradients[0]
    
        # 通道维度加权平均
        weights = grad.mean(dim=(2,3), keepdim=True)
        grad_cam = torch.sum(activation * weights, dim=1, keepdim=True)
        
        # 后处理
        grad_cam = F.relu(-grad_cam)
        # grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-8)
        grad_cam = grad_cam / grad_cam.max()
        grad_cam = F.softmax(grad_cam, dim = -1)
        

        if not os.path.exists("grad_photos"):
            os.mkdir("grad_photos")

        save_image(grad_cam, 'grad_photos/grad.png', nrow=1, normalize=True)

        # print(grad_cam[0])
        # print(1/0)
    
        return grad_cam
