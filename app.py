import numpy as np
import pandas as pd

import streamlit as st
import altair as alt
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import datasets, transforms

import util

def show_overview_page():
  st.title('Overview of Datasets')
  st.text('By Jiajun Bao and Zixu Chen')

  st.header('Note to First Time User')
  st.write('If this is the first time you open the application, please wait a bit for the dataset to be downloaded 🙂')

  dataloaders = {}

  # TODO: show dataset introduction and statistics
  # TODO: show 'loading data' in the app?

  st.header('MNIST Dataset')
  if st.checkbox('Display sample MNIST images'):
    dataloaders['MNIST'] = util.load_sample_data('MNIST', torchvision.transforms.ToTensor())
    data, labels = next(iter(dataloaders['MNIST']))
    fig = util.plot_grayscale_img(data, labels)
    fig

  st.header('FashionMNIST Dataset')
  if st.checkbox('Display sample FashionMNIST images'):
    dataloaders['FashionMNIST'] = util.load_sample_data('FashionMNIST', torchvision.transforms.ToTensor())
    data, labels = next(iter(dataloaders['FashionMNIST']))
    fig = util.plot_grayscale_img(data, labels)
    fig

  st.header('KMNIST Dataset')
  if st.checkbox('Display sample KMNIST images'):
    dataloaders['KMNIST'] = util.load_sample_data('KMNIST', torchvision.transforms.ToTensor())
    data, labels = next(iter(dataloaders['KMNIST']))
    fig = util.plot_grayscale_img(data, labels)
    fig

  st.header('QuickDraw Dataset')
  if st.checkbox('Display sample QuickDraw images'):
    st.write('Still working on this 🚧')

def show_preprocessing_page():
  st.title('Data Preprocessing and Augmentation')
  st.text('By Jiajun Bao and Zixu Chen')

  st.header('Explore Data Augmentation Techniques')
  st.write('Note: common data augmentation techniques are typically used in non-GAN tasks, such as classification. However, here we want you to explore how applying data augmentation could affect the results of GAN')

  st.sidebar.subheader('Data Augmentation Configurations')
  dataset = st.sidebar.selectbox('Select dataset:', (
    'MNIST',
    'FashionMNIST',
    'KMNIST'
  ))

  dataloaders = {}

  # Original
  st.subheader('Original Dataset')
  st.write('This is the original images in the dataset without any augmentations')

  dataloaders['Original'] = util.load_sample_data(dataset, transforms.ToTensor())
  data, labels = next(iter(dataloaders['Original']))
  fig = util.plot_grayscale_img(data, labels)
  fig

  # Crop In the Center
  st.subheader('Crop In the Center')
  st.write('Crops the given image at the center. With small crop size, the image is zoomed in to the center. With large crop size, the image is zoomed out from the center')

  crop_size = int(st.text_input('Crop size (int):', '15'))
  if crop_size < 1:
    crop_size = 1

  transform = transforms.Compose([
    transforms.CenterCrop(crop_size),
    transforms.ToTensor()
  ])
  dataloaders['CenterCrop'] = util.load_sample_data(dataset, transform)
  data, labels = next(iter(dataloaders['CenterCrop']))
  fig = util.plot_grayscale_img(data, labels)
  fig

  # Color Jitter
  st.subheader('Color Jitter')
  st.write('Randomly change the brightness, contrast, saturation and hue of an image')

  brightness = float(st.text_input('Brightness (float):', '0'))
  contrast = float(st.text_input('Contrast (float):', '0'))
  saturation = float(st.text_input('Saturation (float):', '0'))
  hue = float(st.text_input('Hue (float, [0, 0.5]):', '0'))

  transform = transforms.Compose([
    transforms.ColorJitter(brightness, contrast, saturation, hue),
    transforms.ToTensor()
  ])
  dataloaders['ColorJitter'] = util.load_sample_data(dataset, transform)
  data, labels = next(iter(dataloaders['ColorJitter']))
  fig = util.plot_grayscale_img(data, labels)
  fig

  # Random Rotation
  st.subheader('Random Rotation')
  st.write('Rotate the image by angle. The rotation range will be [-Max Degree, +Max Degree]')

  degrees = int(st.text_input('Max Degree (int):', '0'))

  transform = transforms.Compose([
    transforms.RandomRotation(degrees),
    transforms.ToTensor()
  ])
  dataloaders['RandomRotation'] = util.load_sample_data(dataset, transform)
  data, labels = next(iter(dataloaders['RandomRotation']))
  fig = util.plot_grayscale_img(data, labels)
  fig

  # Gaussian Blur
  st.subheader('Gaussian Blur')
  st.write('Blurs image with randomly chosen Gaussian blur')

  kernel_size = int(st.text_input('Kernel Size (odd and positive int):', '5'))

  transform = transforms.Compose([
    transforms.GaussianBlur(kernel_size),
    transforms.ToTensor()
  ])
  dataloaders['GaussianBlur'] = util.load_sample_data(dataset, transform)
  data, labels = next(iter(dataloaders['GaussianBlur']))
  fig = util.plot_grayscale_img(data, labels)
  fig

def show_training_page():
  st.title('Model Training')
  st.text('By Jiajun Bao and Zixu Chen')

  st.sidebar.subheader('Hyperparameter Configurations')

  dataset = st.sidebar.selectbox('Training dataset:', (
    'MNIST',
    'FashionMNIST',
    'KMNIST'
  ))
  lr = float(st.sidebar.text_input('Learning Rate (float):', '0.01'))
  epochs = int(st.sidebar.text_input('Training Epochs (int):', '10'))
  da = st.sidebar.multiselect('Data Augmentations:', [
        'CenterCrop',
        'ColorJitter',
        'RandomRotation',
        'GaussianBlur'
       ], [])
  
  st.subheader('Current Training Configurations')
  st.write(f'Dataset = {dataset}')
  st.write(f'Learning Rate = {lr}')
  st.write(f'Epochs = {epochs}')
  st.write(f'Data Augmentations = {da}')

  st.subheader("Ready? Let's get started!")
  if st.button('Start Training'):
    fig = util.plot_fake_loss()
    fig

def show_inference_page():
  st.title('Model Inference')
  st.text('By Jiajun Bao and Zixu Chen')

  st.write('Still working on this 🚧')

  # TODO: show generated image
  # TODO: add pre-trained models for further exploration


st.sidebar.title('GAN Visualizer')
st.sidebar.write('Help beginners to learn GAN more easily')

st.sidebar.subheader('Page Navigation')
pages = [
    'Dataset Overview',
    'Data Preprocessing',
    'Model Training',
    'Model Inference'
]

page = st.sidebar.selectbox('Choose a stage to explore:', pages)

if page == pages[0]:
    show_overview_page()
elif page == pages[1]:
    show_preprocessing_page()
elif page == pages[2]:
    show_training_page()
elif page == pages[3]:
    show_inference_page()
