import random

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

from DataModule import DataModule


class Sampler:
    def __init__(self, datamodule: DataModule, buffer_size: int, n_steps: int, sgld_lr: int, reinit_freq: int, sgld_std: float, device: str):
        super().__init__()
        self.n_steps = n_steps
        self.sgld_lr = sgld_lr
        self.sgld_std = sgld_std
        self.reinit_freq = reinit_freq
        self.buffer_size = buffer_size
        self.device = device

        self.img_shape = tuple(datamodule.img_shape)
        self.batch_size = datamodule.batch_size
        self.dataset = datamodule.dataset
        self.n_classes = datamodule.n_classes

        self.conditionals = []
        self.replay_buffer = self.init_from_centers()
        self.replay_buffer = list(self.replay_buffer.chunk(self.batch_size, dim=0))

    def init_random(self, batch_size, img_shape, n_classes):
        new = torch.zeros(batch_size, img_shape[0], img_shape[1], img_shape[2])

        for i in range(batch_size):
            index = np.random.randint(n_classes)
            dist = self.conditionals[index]
            new[i] = dist.sample().view(img_shape)

        return torch.clamp(new, -1, 1).cpu()

    def init_from_centers(self):
        with open(f"weights/{self.dataset}_mean.pt", "rb") as file:
            centers = torch.load(file)

        with open(f"weights/{self.dataset}_cov.pt", "rb") as file:
            covs = torch.load(file)

        buffer = []
        for i in range(self.n_classes):
            mean = centers[i].to(self.device)
            cov = covs[i].to(self.device)
            dist = MultivariateNormal(mean, covariance_matrix=cov + 1e-4 * torch.eye(int(np.prod(self.img_shape))).to(self.device))
            self.conditionals.append(dist)
            tensor = dist.sample((self.batch_size // self.n_classes,)).view((self.batch_size // self.n_classes,) + self.img_shape).cpu()
            buffer.append(tensor)

        images = torch.clamp(torch.cat(buffer), -1, 1)

        return images

    def sample_q(self, f: nn.Module):
        n_new = np.random.binomial(self.batch_size, self.reinit_freq)

        rand_imgs = self.init_random(n_new, self.img_shape, self.n_classes)
        old_imgs = torch.cat(random.choices(self.replay_buffer, k=self.batch_size - n_new), dim=0)
        inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach().to(self.device)
        inp_imgs, loss = self.generate_samples(f, inp_imgs)

        self.replay_buffer = list(inp_imgs.to(torch.device("cpu")).chunk(self.batch_size, dim=0)) + self.replay_buffer
        self.replay_buffer = self.replay_buffer[: self.buffer_size]

        return loss

    def generate_samples(self, f, inp_imgs, return_img_per_step=False):
        inp_imgs.requires_grad = True

        imgs_per_step = []

        for _ in range(self.n_steps):
            energies = f(inp_imgs)
            e_x = energies.sum()
            eta = torch.autograd.grad(e_x, [inp_imgs], retain_graph=True)[0]
            tmp_inp = inp_imgs + torch.clamp(eta, -1, 1) * self.sgld_lr

            inp_imgs.data = tmp_inp.data

            if self.sgld_std > 0.0:
                inp_imgs.data += self.sgld_std * torch.randn_like(inp_imgs)

            inp_imgs.data = torch.clamp(inp_imgs.data, -1, 1)

            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())

        f.train()

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs
