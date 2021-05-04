import os
import math
import json

import streamlit as st
from matplotlib import image
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import datasets

num_mnist_generated_imgs = 14

dataset_to_method = {
    'MNIST': datasets.MNIST,
    'FashionMNIST': datasets.FashionMNIST,
    'KMNIST': datasets.KMNIST
}


def plot_grayscale_img(imgs, labels):
    fig = plt.figure()
    for i in range(15):
        plt.subplot(3, 5, i + 1)
        plt.imshow(imgs[i][0], cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.xticks([])
        plt.yticks([])
    return fig


# @st.cache
def load_sample_data(dataset, transform):
    load_method = dataset_to_method[dataset]
    data = load_method(root='./data', train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(data, batch_size=15, shuffle=False)
    return loader


def rolling_avg(data, roll_range=50):
    ravg = []
    for i in range(len(data)):
        last_range = data[max(0, i - (roll_range - 1)):i + 1]
        ravg.append(sum(last_range) / len(last_range))
    return ravg


def smooth_stats(data, upperbound):
    if upperbound == -1:
        return data
    return list(map(lambda x: x if x < upperbound else upperbound, data))


# @st.cache
def load_pretrain_stats(lr, latent_dim, dataset):
    with open(f'data/pretrained_{dataset.lower()}_stats.json') as f:
        data = json.loads(f.read())

    for d in data:
        config = d['config']
        if config['lr'] == lr and config['latent_dim'] == latent_dim:
            batch_step = d['batches_done']
            g_loss = d['g_loss']
            d_loss = d['d_loss']
            break

    return batch_step, g_loss, d_loss


def get_img_paths(img_dir):
    paths = []
    for root, _, files in os.walk(img_dir):
        for f in files:
            if f == '.DS_Store':
                continue
            paths.append(os.path.join(root, f))
    return sorted(paths)


def load_img(img_dir):
    paths = get_img_paths(img_dir)
    imgs = [image.imread(img) for img in paths]
    return imgs


def plot_generated_img(imgs, step):
    idx = math.ceil(step / 500)
    fig = plt.figure()
    plt.imshow(imgs[idx])
    plt.title(f'After: {step} steps')
    return fig
