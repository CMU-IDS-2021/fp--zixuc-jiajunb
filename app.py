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

  dataloaders = {}

  # TODO: show dataset introduction and statistics
  # TODO: show "loading data" in the app

  st.header('MNIST Dataset')
  dataloaders['MNIST'] = util.load_sample_data('MNIST', torchvision.transforms.ToTensor())
  data, labels = next(iter(dataloaders['MNIST']))
  if st.checkbox('Display sample MNIST images'):
    fig = util.plot_grayscale_img(data, labels)
    fig

  st.header("Fashion-MNIST Dataset")
  dataloaders['FashionMNIST'] = util.load_sample_data('FashionMNIST', torchvision.transforms.ToTensor())
  data, labels = next(iter(dataloaders['FashionMNIST']))
  if st.checkbox('Display sample Fashion-MNIST images'):
    fig = util.plot_grayscale_img(data, labels)
    fig

  st.header("Quick-Draw Dataset")

def show_preprocessing_page():
  st.title("Data Preprocessing and Augmentation")
  st.text('By Jiajun Bao and Zixu Chen')

  st.header("Explore Data Augmentation Techniques")

  '''
  Note: common data augmentation techniques are typically used in non-GAN tasks, such as classification. However, here we want you to explore how applying data augmentation could affect the results of GAN
  '''

  dataloaders = {}

  st.subheader("Crop In the Center")
  transform = transforms.Compose([
    transforms.CenterCrop(15),
    transforms.ToTensor()
  ])
  dataloaders['MNIST'] = util.load_sample_data('MNIST', transform)
  data, labels = next(iter(dataloaders['MNIST']))
  if st.checkbox('Display sample MNIST images'):
    fig = util.plot_grayscale_img(data, labels)
    fig
  
  # TODO: allow switching dataset
  # TODO: allow tuning data augmentation parameters

def show_training_page():
  st.title("Model Training")
  st.text('By Jiajun Bao and Zixu Chen')

  # TODO: let users specify the data augmentation techniques to incorporate
  # TODO: specify hyperparameters like learning rate

def show_inference_page():
  st.title("Model Inference")
  st.text('By Jiajun Bao and Zixu Chen')

  # TODO: show generated image
  # TODO: add pre-trained models for further exploration


st.sidebar.title('GAN Visualizer')

pages = [
    'Dataset Overview',
    'Data Preprocessing',
    'Model Training',
    'Model Inference'
]

page = st.sidebar.selectbox('Choose a stage to explore', pages)

if page == pages[0]:
    show_overview_page()
elif page == pages[1]:
    show_preprocessing_page()
elif page == pages[2]:
    show_training_page()
elif page == pages[3]:
    show_inference_page()
