import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.sidebar.title('GAN Visualizer')
page = st.sidebar.selectbox('Choose a stage to explore', [
    'Dataset Overview',
    'Data Preprocessing',
    'Model Training',
    'Model Inference'
])

st.title("Generative Adversarial Network (GAN)")

st.text('By Jiajun Bao and Zixu Chen')
