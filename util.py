import torch
import torchvision
from torchvision import datasets

import matplotlib.pyplot as plt

dataset_to_method = {
  'MNIST': datasets.MNIST,
  'FashionMNIST': datasets.FashionMNIST
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
