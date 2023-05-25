from os import listdir
from os.path import join

import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

st.title("Bienvenue !")

st.header("Classification d'images - Deep Learning & CNN")

st.write(
"""Développer une application qui permet de détecter automatiquement des images d'animaux Chiens et Chats.
L'utilisateur doit pouvoir uploader une photo et l'application doit préciser de quel animal il s'agit ainsi que la probabilité de la classification.
Le classifieur sera développé avec Keras."""
)

st.header("Résultats des modèles")

folder_name = 'models'

for model_name in sorted(listdir(folder_name)):

    df = pd.read_csv(join(folder_name, model_name, 'stats.csv'))

    fig, ax = plt.subplots(layout='constrained')
    ax.plot(df['epoch'], df['loss'], label='loss')
    ax.plot(df['epoch'], df['val_loss'], label='val_loss')

    ax.plot(df['epoch'], df['accuracy'], label='accuracy')
    ax.plot(df['epoch'], df['val_accuracy'], label='val_accuracy')

    df.set_index("epoch")

    ax.vlines(df['val_accuracy'].idxmax(), ymin=0, ymax=1, label="max val_accuracy", color="black", linestyles='dotted')
    ax.hlines(0.9, xmin=df['epoch'].min(), xmax=df['epoch'].max(), label="90% accuracy", color="black", linestyles='dashed')

    ax.set_title(model_name)
    ax.legend()

    st.subheader(model_name)

    st.pyplot(fig)

    st.write(f"**Best val_accuracy :** {round(df['val_accuracy'].max(), 5)} (epoch {df['val_accuracy'].idxmax()})")
    st.write(f"**Best val_loss :** {round(df['val_loss'].min(), 5)} (epoch {df['val_loss'].idxmin()})")
    st.write(f"**Last val_accuracy :** {round(df['val_accuracy'][df.index.max()], 5)}")
    st.write(f"**Last val_loss :** {round(df['val_loss'][df.index.max()], 5)}")