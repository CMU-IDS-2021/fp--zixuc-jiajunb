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

from torch.autograd import Variable
import util
import SessionState
import modeling

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Prevent warning in MacOS


def show_overview_page():
    st.title('Overview of Datasets')
    st.text('By Jiajun Bao and Zixu Chen')

    st.write(
        "We use two datasets in the application, the MNIST and FashionMNIST datasets. Since they are all relatively small datasets, it's easier to train than larger datasets like human face datasets. Additionally, these datasets are easy to understand, so we can easily evaluate the end results."
    )

    dataloaders = {}

    st.header('MNIST Dataset')
    st.write(
        'MNIST (Modified National Institute of Standards and Technology) dataset is a set of handwritten digits from 0-9 (a total of 10 classes). The dataset images are all 28×28 pixel grayscale images. There are a total of 60,000 training images and 10,000 testing images.'
    )
    st.markdown('[MNIST Source](http://yann.lecun.com/exdb/mnist/)')

    st.subheader('MNIST Labels')
    st.image('assets/mnist-labels.png')
    st.text('Fig1: MNIST labels [ref: https://m-alcu.github.io/blog/2018/01/13/nmist-dataset/]')

    st.subheader('MNIST Sample Images')
    dataloaders['MNIST'] = util.load_sample_data('MNIST', torchvision.transforms.ToTensor())
    data, labels = next(iter(dataloaders['MNIST']))
    fig = util.plot_grayscale_img(data, labels)
    fig

    st.header('FashionMNIST Dataset')
    st.write(
        'Similar to MNIST, FashionMNIST is dataset of 10 classes of fashion items. The images are also 28×28 pixel grayscale images, and the dataset also consists of 60,000 training images and 10,000 testing images.'
    )
    st.markdown('[FashionMNIST Source](https://github.com/zalandoresearch/fashion-mnist)')

    st.subheader('FashionMNIST Labels')
    st.image('assets/fashionmnist-labels.png')
    st.text(
        'Fig2: FashionMNIST labels [ref: https://www.researchgate.net/figure/Fashion-MNIST-Dataset-Images-with-Labels-and-Description-II-LITERATURE-REVIEW-In-image_fig1_340299295]'
    )

    st.subheader('FashionMNIST Sample Images')
    dataloaders['FashionMNIST'] = util.load_sample_data('FashionMNIST',
                                                        torchvision.transforms.ToTensor())
    data, labels = next(iter(dataloaders['FashionMNIST']))
    fig = util.plot_grayscale_img(data, labels)
    fig


def show_gan_intro_page():
    st.title('Introduction of GAN')
    st.text('By Jiajun Bao and Zixu Chen')

    st.header('What Is GAN?')
    st.image('assets/gan.png')
    st.text('Fig1: GAN architecture [ref: https://sthalles.github.io/intro-to-gans/]')
    st.write(
        'Generative adversarial network (GAN) is a deep neural network architecture that can output new data that resembles input data. At a high level, GAN consists of a generator and a discriminator. The generator learns from the training set to generate outputs with similar distribution as the original data, while the discriminator tries to distinguish generated outputs from the authentic data. The two have to compete with each other in a zero-sum game fashion to avoid suffering loss.'
    )

    st.header('How Can GAN Visualizer Help You Understand GAN?')
    st.write(
        '''You can use the "Trained Model Logs", "Model Training", and "Model Inference" page to play around with training GAN and using GAN. In those pages, you will be able to tune some of the most essential hyperparameters to see how they can affect the generated images. In this way, you don't need to be confused with the abstract visualization of the gradients while getting the flexibility to tune the model.'''
    )
    st.write(
        'GAN Visualizer also gives you a way of training your own GAN models without having to know how to implement the complex GAN architecture in PyTorch or TensorFlow. We abstract the model details away and allow you to directly tune the training hyperparameters in a GUI. You may freely play around with it to explore GAN. Have fun 😉'
    )

    st.header('Explanation of Provided Hyperparameters')
    st.write(
        'These are the parameters that you will be able to tune in the model training pages. Having a good understanding of what they are can help you get a sense of how to train a good model.'
    )
    st.markdown('**Dataset**: the dataset you will be training on')
    st.markdown(
        '**Dataset Classes**: how many classes to look at. Using 1 class can give a good result in a short time'
    )
    st.markdown('**Learning Rate**: step size in the gradient descent optimization')
    st.markdown('**Latent Variable Dimension**: size of the hidden weight')
    st.markdown(
        '**Batch Size**: number of examples to work though in each optimization iteration. It is the subset of the entire training dataset)'
    )
    st.markdown('**Epochs**: each epoch means going through the entire training dataset once')
    st.markdown('**Sample Interval**: how frequently we sample the loss and generated images')

    st.header('How Does GAN Differ From CNN?')
    st.image('assets/lenet5.png')
    st.text(
        'Fig2: example CNN architecture [ref: https://www.researchgate.net/figure/The-architecture-of-LeNet-5-23-a-CNN-used-for-digits-recognition-for-the-MNIST-dataset_fig2_321665783]'
    )
    st.write(
        'Both GAN and Convolutional Neural Network (CNN) are deep learning neural network architectures that mainly target the computer vision tasks. CNN is often used in discriminative tasks like image classification, while GAN is often used in generative tasks like generating new images. For example, with the MNIST dataset, CNN can recognize the digits while GAN can generate random new handwritten digit images.'
    )

    st.header("What's The Relationship Between GAN, CNN, And Data Augmentation?")
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
    dataset = st.sidebar.selectbox('Select dataset:', ('MNIST', 'FashionMNIST'))

    dataloaders = {}

    # Original
    st.subheader('Original Dataset')
    st.write('This is the original images in the dataset without any augmentations')

    dataloaders['Original'] = util.load_sample_data(dataset, transforms.ToTensor())
    data, labels = next(iter(dataloaders['Original']))
    orig_fig = util.plot_grayscale_img(data, labels)
    orig_fig

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

    if st.checkbox('Compare Crop In the Center With No Augmentation'):
        orig_fig

    # Color Jitter
    st.subheader('Color Jitter')
    st.write('Randomly change the brightness, contrast, saturation and hue of an image')

    brightness = float(st.text_input('Brightness (float):', '50'))
    contrast = float(st.text_input('Contrast (float):', '50'))
    saturation = float(st.text_input('Saturation (float):', '0'))
    hue = float(st.text_input('Hue (float, [0, 0.5]):', '0'))

    transform = transforms.Compose(
        [transforms.ColorJitter(brightness, contrast, saturation, hue),
         transforms.ToTensor()])
    dataloaders['ColorJitter'] = util.load_sample_data(dataset, transform)
    data, labels = next(iter(dataloaders['ColorJitter']))
    fig = util.plot_grayscale_img(data, labels)
    fig

    if st.checkbox('Compare Color Jitter With No Augmentation'):
        orig_fig

    # Random Rotation
    st.subheader('Random Rotation')
    st.write('Rotate the image by angle. The rotation range will be [-Max Degree, +Max Degree]')

    degrees = int(st.text_input('Max Degree (int):', '90'))

    transform = transforms.Compose([transforms.RandomRotation(degrees), transforms.ToTensor()])
    dataloaders['RandomRotation'] = util.load_sample_data(dataset, transform)
    data, labels = next(iter(dataloaders['RandomRotation']))
    fig = util.plot_grayscale_img(data, labels)
    fig

    if st.checkbox('Compare Random Rotation With No Augmentation'):
        orig_fig

    # Gaussian Blur
    st.subheader('Gaussian Blur')
    st.write('Blurs image with randomly chosen Gaussian blur')

    kernel_size = int(st.text_input('Kernel Size (odd and positive int):', '5'))

    transform = transforms.Compose([transforms.GaussianBlur(kernel_size), transforms.ToTensor()])
    dataloaders['GaussianBlur'] = util.load_sample_data(dataset, transform)
    data, labels = next(iter(dataloaders['GaussianBlur']))
    fig = util.plot_grayscale_img(data, labels)
    fig

    if st.checkbox('Compare Gaussian Blur With No Augmentation'):
        orig_fig


# Reference: https://discuss.streamlit.io/t/is-there-any-working-example-for-session-state-for-streamlit-version-0-63-1/4551/2
session_state = SessionState.get(
    start_pretrain=False,
    pretrain_dataset='MNIST',
    pretrain_lr=0.0001,
    pretrain_latent_dim=25,
    start_train=False,
    finish_train=False,
    dataset='MNIST',
    dataset_classes=1,
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
    st.title('Trained Model Logs')
    st.text('By Jiajun Bao and Zixu Chen')

    st.write(
        'To avoid the long training wait time, we provide a set of pre-trained model logs with less flexibility in hyperparameter configurations.'
    )
    st.write(
        'We have recorded the logs of the loss data and generated image when training these models, so you can simulate a real training process without having to acutally wait.'
    )
    st.markdown(
        '**Hint**: you may check the "GAN Introduction" page for explanation of the provided hyperparameters.'
    )

    st.sidebar.subheader('Hyperparameter Configurations')
    dataset = st.sidebar.selectbox('Training Dataset:', ('MNIST', 'FashionMNIST'))
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

    st.write(
        "You can reduce the number of training epochs to a small number to finish the training faster. However, the generated images might look very bad if we don't train enough."
    )
    st.markdown('**One Class Dataset Label**: digit 8 (MNIST), bag (FashionMNIST)')
    st.markdown(
        '**Hint**: you may check the "GAN Introduction" page for explanation of the provided hyperparameters.'
    )

    st.sidebar.subheader('Hyperparameter Configurations')

    dataset = st.sidebar.selectbox('Training Dataset:', ('MNIST', 'FashionMNIST'))
    dataset_classes = st.sidebar.selectbox('Dataset Classes:', (1, 10), 0)
    lr = float(st.sidebar.text_input('Learning Rate (float):', '0.0004'))
    latent_dim = int(st.sidebar.text_input('Latent Variable Dimension (int):', '100'))
    batch_size = st.sidebar.selectbox('Batch Size:', ([32]))
    epochs = int(st.sidebar.text_input('Training Epochs (int):', '20'))
    sample_interval = st.sidebar.selectbox('Sample Interval:', (10, 20, 50, 100, 500, 1000), 0)
    show_progress = st.sidebar.checkbox("Show Intermediate Generated Images", value=True)

    st.sidebar.subheader("Ready? Let's get started!")
    start_train = st.sidebar.button('Start Training')
    if dataset != session_state.dataset or dataset_classes != session_state.dataset_classes or lr != session_state.lr or latent_dim != session_state.latent_dim or epochs != session_state.epochs or sample_interval != session_state.sample_interval:
        session_state.start_train = False
        session_state.finish_train = False

    st.header('Training Progress')
    batch_step = []
    g_loss = []
    d_loss = []
    imgs = []

    if dataset_classes == 1:
        total_steps = 63 * epochs
        dataset_path = 'data/small'
    elif dataset_classes == 10:
        total_steps = 625 * epochs
        dataset_path = 'data/ten_classes'

    if start_train or session_state.start_train:
        session_state.start_train = True
        session_state.dataset = dataset
        session_state.dataset_classes = dataset_classes
        session_state.lr = lr
        session_state.latent_dim = latent_dim
        session_state.epochs = epochs
        session_state.sample_interval = sample_interval
        progress_bar = st.progress(0)

        if not session_state.finish_train:
            for ret in modeling.train(lr, latent_dim, epochs, sample_interval, dataset,
                                      dataset_path):
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

        # Note: sometimes the step will exceed max_value?
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

    st.write(
        'You may generate new images with your saved model from "Model Training" page or the pre-trained model we provide'
    )

    model = st.sidebar.selectbox('Inference Model:', ('My Model', 'Pre-trained Model'))

    latent_dim = 100
    generator = None

    if model == 'My Model':
        st.header('Inference With The Model You Just Trained')

        # Load saved model
        if session_state.saved_model != None:
            latent_dim = session_state.saved_latent_dim
            generator = modeling.Generator(latent_dim)
            # generator = modeling.Loaded_Generator(10, latent_dim, 28)
            generator.load_state_dict(session_state.saved_model)
            generator.eval()
        else:
            st.write(
                'No saved trained model found... Please train one at the "Model Training" page')
    else:
        st.header('Inference With Pre-trained Model')

        model_name = st.sidebar.selectbox('Trained-on:', ('MNIST', 'FashionMNIST'))
        ckpt = torch.load(os.path.join('ckpts', f'{model_name}_generator.pth.tar'))
        latent_dim = ckpt['config']['latent_dim']
        args = ckpt["generater_args"]
        n_classes, latent_dim, img_shape = args['n_classes'], args['latent_dim'], args['img_shape']
        generator = modeling.Loaded_Generator(n_classes, latent_dim, img_shape)
        generator.load_state_dict(ckpt['generator'])
        generator.eval()

    if generator != None:
        value = st.slider(
            'Set the initial input vector',
            min_value=0,
            max_value=10,
            value=3,
            step=1,
        )

        if st.button('Start Inferencing'):
            st.write(generator)
            n_row = 5
            z = Variable(
                torch.FloatTensor(value / 10 * np.random.normal(0, 1, (n_row**2, latent_dim))))

            # Get labels ranging from 0 to n_classes for n rows
            labels = np.array([num for _ in range(n_row) for num in range(n_row)])
            labels = Variable(torch.LongTensor(labels))
            gen_imgs = generator.eval_forward(z, labels)
            st.write(gen_imgs.shape)


st.sidebar.title('GAN Visualizer')
st.sidebar.write('Help beginners to learn GAN more easily')

st.sidebar.subheader('Page Navigation')
pages = [
    'Dataset Overview', 'GAN Introduction', 'Trained Model Logs', 'Model Training',
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
