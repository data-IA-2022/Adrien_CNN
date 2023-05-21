import os

import streamlit as st

st.title("Prédiction d'image")

st.header("Choisir une image")

img = st.file_uploader("Choisir une image pour la prédiction", label_visibility = "collapsed")

if img:
    st.image(img)

    with st.spinner("Traitement de l'image..."):

        import tensorflow as tf
        from PIL import Image
        from tensorflow import keras

        PILimg = Image.open(img)
        PILimg.save("img_tmp.jpg")
        PILimg.close()

        model = keras.models.load_model("saves/save_at_25.keras")

        image_size = (180, 180)

        img = keras.utils.load_img(
            "img_tmp.jpg", target_size=image_size
        )
        img_array = keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis
        predictions = model.predict(img_array)
        score = float(predictions[0])

        os.remove("img_tmp.jpg")

        cat = (1 - score)
        dog = score

    st.header("Résultat")

    if cat >= dog:
        st.write(f"Cette image est un chat ({cat:.2%}).")
    else:
        st.write(f"Cette image est un chien ({dog:.2%}).")