import os
import math
import json
import numpy as np
import pandas as pd

import streamlit as st
import altair as alt
from matplotlib import image
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import datasets

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


def smooth_stats(data, upperbound):
    if upperbound == -1:
        return data
    return list(map(lambda x: x if x < upperbound else upperbound, data))


def rolling_avg(data, roll_range=50):
    ravg = []
    for i in range(len(data)):
        last_range = data[max(0, i - (roll_range - 1)):i + 1]
        ravg.append(sum(last_range) / len(last_range))
    return ravg


def plot_loss_figs(batch_step, g_loss, d_loss, upperbound, roll_range):
    g_loss = smooth_stats(g_loss, upperbound)
    d_loss = smooth_stats(d_loss, upperbound)
    loss_df = pd.DataFrame({
        'Batch Step': batch_step,
        'Generator': g_loss,
        'Discriminator': d_loss,
    })
    loss_df = loss_df.melt('Batch Step')
    loss_plot = alt.Chart(
        loss_df,
        title="Generator and Discriminator's Loss during Training",
    ).mark_line().encode(
        x=alt.X("Batch Step:Q", title="Batch Step"),
        y=alt.Y(
            "value:Q",
            title="Loss",
        ),
        color=alt.Y(
            "variable",
            title="Category",
        ),
        tooltip=[
            alt.Tooltip('Batch Step:Q', title="Batch Step"),
            alt.Tooltip('variable', title="Category"),
            alt.Tooltip('value', title="Loss"),
        ],
    ).properties(width=600, height=400)

    g_loss_ravg = smooth_stats(rolling_avg(g_loss, roll_range), upperbound)
    d_loss_ravg = smooth_stats(rolling_avg(d_loss, roll_range), upperbound)
    loss_ravg_df = pd.DataFrame({
        'Batch Step': batch_step,
        'Generator': g_loss_ravg,
        'Discriminator': d_loss_ravg,
    })
    loss_ravg_df = loss_ravg_df.melt('Batch Step')

    step_select = alt.selection_interval(encodings=['x'])
    loss_ravg_plot = alt.Chart(
        loss_ravg_df,
        title="Generator and Discriminator's Rolling Average Loss during Training",
    ).mark_line().encode(
        x=alt.X("Batch Step:Q", title="Batch Step"),
        y=alt.Y(
            "value:Q",
            title="Rolling Avg Loss",
        ),
        color=alt.Y(
            "variable",
            title="Category",
        ),
        tooltip=[
            alt.Tooltip('Batch Step:Q', title="Batch Step"),
            alt.Tooltip('variable', title="Category"),
            alt.Tooltip('value', title="Rolling Avg Loss"),
        ],
    ).add_selection(step_select).properties(width=600, height=400)

    scaled_loss_plot = loss_plot.encode(
        alt.X("Batch Step:Q", title="Batch Step", scale=alt.Scale(domain=step_select)))

    return scaled_loss_plot, loss_ravg_plot


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


def plot_pretrain_generated_img(imgs, step):
    idx = math.ceil(step / 500)
    fig = plt.figure()
    plt.imshow(imgs[idx])
    plt.title(f'After: {step} steps')
    plt.xticks([])
    plt.yticks([])
    return fig


def plot_generated_img(imgs, step, sample_interval):
    idx = int(step / sample_interval)
    imgs = imgs[idx]
    fig = plt.figure()
    fig.suptitle(f'Step: {step}')
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(imgs[i][0], cmap='gray')
        plt.xticks([])
        plt.yticks([])
    fig
    return fig
