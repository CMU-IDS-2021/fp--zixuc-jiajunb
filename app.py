import os
import math
import numpy as np
import pandas as pd

import streamlit as st
import altair as alt
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import datasets, transforms

import util
import SessionState
import modeling

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Prevent warning in MacOS


def show_overview_page():
    st.title('Overview of Datasets')
    st.text('By Jiajun Bao and Zixu Chen')

    st.header('Note to First Time Users')
    st.write(
        'If this is the first time you open the application, please wait a bit for the dataset to be downloaded 🙂'
    )

    dataloaders = {}

    # TODO: show dataset introduction

    st.header('MNIST Dataset')
    if st.checkbox('Display sample MNIST images'):
        dataloaders['MNIST'] = util.load_sample_data('MNIST', torchvision.transforms.ToTensor())
        data, labels = next(iter(dataloaders['MNIST']))
        fig = util.plot_grayscale_img(data, labels)
        fig

    st.header('FashionMNIST Dataset')
    if st.checkbox('Display sample FashionMNIST images'):
        dataloaders['FashionMNIST'] = util.load_sample_data('FashionMNIST',
                                                            torchvision.transforms.ToTensor())
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


def show_gan_intro_page():
    st.title('Introduction of GAN')
    st.text('By Jiajun Bao and Zixu Chen')

    st.header('What is GAN?')
    st.write(
        'Generative adversarial network (GAN) is a deep neural network architecture that can output new data that resembles input data. At a high level, GAN consists of a generator and a discriminator. The generator learns from the training set to generate outputs with similar distribution as the original data, while the discriminator tries to distinguish generated outputs from the authentic data. The two have to compete with each other in a zero-sum game fashion to avoid suffering loss.'
    )

    st.header('How can GAN Visualizer help you understand GAN?')
    st.write(
        '''You can use the "Pre-trained Models", "Model Training", and "Model Inference" page to play around with training GAN and using GAN. In those pages, you will be able to tune some of the most essential hyperparameters to see how they can affect the generated images. In this way, you don't need to be confused with the abstract visualization of the gradients while getting the flexibility to tune the model.'''
    )

    st.header('Explanation of the hyperparameters we provide')
    st.write(
        'These are the parameters that you will be able to tune in the model training pages. Having a good understanding of what they are can help you get a sense of how to train a good model.'
    )
    st.markdown('**Dataset**: the dataset you will be training on')
    st.markdown('**Learning Rate**: step size in the gradient descent optimization')
    st.markdown('**Latent Variable Dimension**: size of the hidden weight')
    st.markdown(
        '**Batch Size**: number of examples to work though in each optimization iteration. It is the subset of the entire training dataset)'
    )
    st.markdown('**Epochs**: each epoch means going through the entire training dataset once')
    st.markdown('**Sample Interval**: how frequently we sample the loss and generated images')

    st.header('How does GAN differ from CNN?')
    st.write(
        'Both GAN and Convolutional Neural Network (CNN) are deep learning neural network architectures that mainly target the computer vision tasks. CNN is often used in discriminative tasks like image classification, while GAN is often used in generative tasks like generating new images.'
    )

    st.header("What's the relationship between GAN, CNN, and data augmentation?")
    st.write(
        'Data augmentation is a common technique used with CNN. In real-world scenario, we are often lack of training data, so data augmentation is a way to generate new training data. Additionally, with the added noise from data augmentation, the technique can typically make CNN models work more robustly.'
    )
    st.write(
        'Regarding GAN, we mentioned that GAN can be used to generate new images, what can you think of its relationship to data augmentation? We can use GAN to generate hallucinating images used in data augmentation!'
    )

    st.header('Explore Data Augmentation Techniques')
    st.write(
        '''Here you can play around with a few common data augmentation techniques to get a "visualized" sense of what data augmentation (mainly for CNN) is. You will get GAN's data augmentation once you train the model.'''
    )

    st.sidebar.subheader('Data Augmentation Configurations')
    dataset = st.sidebar.selectbox('Select dataset:', ('MNIST', 'FashionMNIST', 'KMNIST'))

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
    st.write(
        'Crops the given image at the center. With small crop size, the image is zoomed in to the center. With large crop size, the image is zoomed out from the center'
    )

    crop_size = int(st.text_input('Crop size (int):', '15'))
    if crop_size < 1:
        crop_size = 1

    transform = transforms.Compose([transforms.CenterCrop(crop_size), transforms.ToTensor()])
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

    transform = transforms.Compose(
        [transforms.ColorJitter(brightness, contrast, saturation, hue),
         transforms.ToTensor()])
    dataloaders['ColorJitter'] = util.load_sample_data(dataset, transform)
    data, labels = next(iter(dataloaders['ColorJitter']))
    fig = util.plot_grayscale_img(data, labels)
    fig

    # Random Rotation
    st.subheader('Random Rotation')
    st.write('Rotate the image by angle. The rotation range will be [-Max Degree, +Max Degree]')

    degrees = int(st.text_input('Max Degree (int):', '0'))

    transform = transforms.Compose([transforms.RandomRotation(degrees), transforms.ToTensor()])
    dataloaders['RandomRotation'] = util.load_sample_data(dataset, transform)
    data, labels = next(iter(dataloaders['RandomRotation']))
    fig = util.plot_grayscale_img(data, labels)
    fig

    # Gaussian Blur
    st.subheader('Gaussian Blur')
    st.write('Blurs image with randomly chosen Gaussian blur')

    kernel_size = int(st.text_input('Kernel Size (odd and positive int):', '5'))

    transform = transforms.Compose([transforms.GaussianBlur(kernel_size), transforms.ToTensor()])
    dataloaders['GaussianBlur'] = util.load_sample_data(dataset, transform)
    data, labels = next(iter(dataloaders['GaussianBlur']))
    fig = util.plot_grayscale_img(data, labels)
    fig


# Reference: https://discuss.streamlit.io/t/is-there-any-working-example-for-session-state-for-streamlit-version-0-63-1/4551/2
session_state = SessionState.get(
    start_pretrain=False,
    pretrain_dataset='MNIST',
    pretrain_lr=0.0001,
    pretrain_latent_dim=25,
    start_train=False,
    finish_train=False,
    dataset='MNIST',
    lr=0.0001,
    latent_dim=25,
    epochs=20,
    sample_interval=10,
    batch_step=[],
    g_loss=[],
    d_loss=[],
    imgs=[],
    saved_model=None,
    saved_latent_dim=25,
)


def show_pretrain_page():
    st.title('Pre-trained Models')
    st.text('By Jiajun Bao and Zixu Chen')
    st.write(
        'To avoid the long training wait time, we provide pre-trained models with less flexibility in hyperparameter configurations.'
    )
    st.write(
        'We have recorded the logs of the loss data and generated image when training these models, so you can simulate a real training process without having to acutally wait.'
    )

    st.sidebar.subheader('Hyperparameter Configurations')
    dataset = st.sidebar.selectbox('Training dataset:', ('MNIST', 'FashionMNIST', 'KMNIST'))
    lr = st.sidebar.selectbox('Learning Rate:', (0.0001, 0.0002, 0.0004, 0.0008, 0.001, 0.002), 2)
    latent_dim = st.sidebar.selectbox('Latent Variable Dimension:', (25, 50, 75, 100, 125), 3)
    batch_size = st.sidebar.selectbox('Batch Size:', ([64]))
    epochs = st.sidebar.selectbox('Training Epochs:', ([20]))

    st.sidebar.subheader("Ready? Let's get started!")
    start_train = st.sidebar.button('Start Training')
    if dataset != session_state.pretrain_dataset or lr != session_state.pretrain_lr or latent_dim != session_state.pretrain_latent_dim:
        session_state.start_pretrain = False

    st.header('Training Loss Results')
    if start_train or session_state.start_pretrain:
        session_state.start_pretrain = True
        session_state.pretrain_dataset = dataset
        session_state.pretrain_lr = lr
        session_state.pretrain_latent_dim = latent_dim

        # Configuring upperbound
        st.markdown('**Configuring Upperbound**')
        st.write(
            'If you look at the loss figure below, for some hyperparameter configurations, the loss result may have outliers that are very high, which can impact the overall scale of the plots. The below field allows you to set an upperbound on the loss to smooth the plots. The default -1 means no upperbound is set.'
        )
        st.write(
            'This typically happens when we have a high learning rate. If we look at the generated image after the outlier appears, they will be a mess.'
        )
        upperbound = int(st.text_input('Smoothing Upperbound (int):', '-1'))

        # Configuring rolling range
        st.markdown('**Configuring Range for Calculating Rolling Average**')
        st.write(
            'The vanilla loss figure is zig-zagging a lot and hard to see the trend. Therefore, we can apply rolling average for a range of data to smooth the curves. Typically, the larger the range is, the smoother the curve is.'
        )
        roll_range = st.selectbox('Rolling Average Range:', (10, 25, 50, 100, 200), 2)

        # Plotting figures
        st.markdown('**Loss Figures**')
        batch_step, g_loss, d_loss = util.load_pretrain_stats(lr, latent_dim, dataset)

        scaled_loss_plot, loss_ravg_plot = util.plot_loss_figs(batch_step, g_loss, d_loss,
                                                               upperbound, roll_range)
        scaled_loss_plot & loss_ravg_plot

        st.write(
            "You may slide on the rolling average loss figure to select a range of batch steps. This allows you to zoom in to the corresponding range in the above vanilla loss figure to see the details."
        )
    else:
        st.write("Waiting for training to start or finish...")

    st.header('Generated Images')
    if start_train or session_state.start_pretrain:
        st.write(
            'You may use the slider below to see how the images generated by the GAN changes through the training process.'
        )

        step = st.slider(
            'After How Many Batch Steps',
            min_value=0,
            max_value=6250,
            value=3000,
            step=500,
        )

        imgs = util.load_img(f'data/{dataset.lower()}_generated/{lr}_{latent_dim}')
        fig = util.plot_pretrain_generated_img(imgs, step)
        fig
    else:
        st.write("Waiting for training to start or finish...")


def show_training_page():
    st.title('Model Training')
    st.text('By Jiajun Bao and Zixu Chen')

    st.sidebar.subheader('Hyperparameter Configurations')

    dataset = st.sidebar.selectbox('Training dataset:', ('MNIST', 'FashionMNIST', 'KMNIST'))
    lr = float(st.sidebar.text_input('Learning Rate (float):', '0.0004'))
    latent_dim = int(st.sidebar.text_input('Latent Variable Dimension (int):', '100'))
    batch_size = st.sidebar.selectbox('Batch Size:', ([32]))
    epochs = int(st.sidebar.text_input('Training Epochs (int):', '20'))
    sample_interval = st.sidebar.selectbox('Sample Interval:', (10, 20, 50, 100, 500, 1000), 0)
    show_progress = st.sidebar.checkbox("Show Intermediate Generated Images", value=True)

    st.sidebar.subheader("Ready? Let's get started!")
    start_train = st.sidebar.button('Start Training')
    if dataset != session_state.dataset or lr != session_state.lr or latent_dim != session_state.latent_dim or epochs != session_state.epochs or sample_interval != session_state.sample_interval:
        session_state.start_train = False
        session_state.finish_train = False

    st.header('Reminder')
    st.write(
        "You can reduce the number of training epochs to a small number to finish the training faster. However, the generated images might look very bad if we don't train enough."
    )

    st.header('Training Progress')
    batch_step = []
    g_loss = []
    d_loss = []
    imgs = []
    total_steps = 63 * epochs

    if start_train or session_state.start_train:
        session_state.start_train = True
        session_state.dataset = dataset
        session_state.lr = lr
        session_state.latent_dim = latent_dim
        session_state.epochs = epochs
        session_state.sample_interval = sample_interval
        progress_bar = st.progress(0)

        if not session_state.finish_train:
            for ret in modeling.train(lr, latent_dim, epochs, sample_interval, dataset):
                batch_step.append(ret['batches_done'])
                g_loss.append(ret['g_loss'])
                d_loss.append(ret['d_loss'])
                imgs.append(ret['first_25_images'])
                g_model = ret['g_model']
                progress = int(batch_step[-1] / total_steps * 100)
                if show_progress:
                    fig = plt.figure()
                    fig.suptitle(f'Step: {batch_step[-1]} | Progress: {progress}%')
                    for i in range(25):
                        plt.subplot(5, 5, i + 1)
                        plt.imshow(imgs[-1][i][0], cmap='gray')
                        plt.xticks([])
                        plt.yticks([])
                    fig
                progress_bar.progress(progress)
            session_state.finish_train = True
            session_state.batch_step = batch_step
            session_state.g_loss = g_loss
            session_state.d_loss = d_loss
            session_state.imgs = imgs
            session_state.saved_model = g_model
            session_state.saved_latent_dim = latent_dim
        else:
            batch_step = session_state.batch_step
            g_loss = session_state.g_loss
            d_loss = session_state.d_loss
            imgs = session_state.imgs

        progress_bar.progress(100)
        st.write("Training finished! Model is automatically saved.")
    else:
        st.write("Waiting for training to start or finish...")

    st.header('Training Loss Results')
    if start_train or session_state.start_train:
        # Configuring upperbound
        st.markdown('**Configuring Upperbound**')
        st.write(
            'If you look at the loss figure below, for some hyperparameter configurations, the loss result may have outliers that are very high, which can impact the overall scale of the plots. The below field allows you to set an upperbound on the loss to smooth the plots. The default -1 means no upperbound is set.'
        )
        st.write(
            'This typically happens when we have a high learning rate. If we look at the generated image after the outlier appears, they will be a mess.'
        )
        upperbound = int(st.text_input('Smoothing Upperbound (int):', '-1'))

        # Configuring rolling range
        st.markdown('**Configuring Range for Calculating Rolling Average**')
        st.write(
            'The vanilla loss figure is zig-zagging a lot and hard to see the trend. Therefore, we can apply rolling average for a range of data to smooth the curves. Typically, the larger the range is, the smoother the curve is.'
        )
        roll_range = st.selectbox('Rolling Average Range:', (10, 25, 50, 100, 200), 2)

        # Plotting figures
        st.markdown('**Loss Figures**')

        scaled_loss_plot, loss_ravg_plot = util.plot_loss_figs(batch_step, g_loss, d_loss,
                                                               upperbound, roll_range)
        scaled_loss_plot & loss_ravg_plot

        st.write(
            "You may slide on the rolling average loss figure to select a range of batch steps. This allows you to zoom in to the corresponding range in the above vanilla loss figure to see the details."
        )
    else:
        st.write("Waiting for training to start or finish...")

    st.header('Generated Images')
    if start_train or session_state.start_train:
        st.write(
            'You may use the slider below to see how the images generated by the GAN changes through the training process.'
        )

        step = st.slider(
            'After How Many Batch Steps',
            min_value=0,
            max_value=total_steps,
            value=0,
            step=sample_interval,
        )

        fig = util.plot_generated_img(imgs, step, sample_interval)
        fig
    else:
        st.write("Waiting for training to start or finish...")


def show_inference_page():
    st.title('Model Inference')
    st.text('By Jiajun Bao and Zixu Chen')

    st.write('Still working on this 🚧')

    # Load saved model
    if session_state.saved_model != None:
        generator = modeling.Generator(session_state.saved_latent_dim)
        generator.load_state_dict(session_state.saved_model)
        generator.eval()
        st.write(generator.state_dict())

    # TODO: generate images with saved and pre-trained models


st.sidebar.title('GAN Visualizer')
st.sidebar.write('Help beginners to learn GAN more easily')

st.sidebar.subheader('Page Navigation')
pages = [
    'Dataset Overview', 'GAN Introduction', 'Pre-trained Models', 'Model Training',
    'Model Inference'
]

page = st.sidebar.selectbox('Choose a stage to explore:', pages)

if page == pages[0]:
    show_overview_page()
elif page == pages[1]:
    show_gan_intro_page()
elif page == pages[2]:
    show_pretrain_page()
elif page == pages[3]:
    show_training_page()
elif page == pages[4]:
    show_inference_page()
