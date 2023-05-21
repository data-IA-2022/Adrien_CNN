import streamlit as st

st.title("Explication d'un réseau de neurones")

st.header("Qu'est-ce qu'un neurone ?")

st.write("Dans un CNN un neurone est une version simplifié de la réalité. C'est une simple fonction mathématique.")

st.image("pages/images/neurone.png")

st.write("La sortie du neurone correspond à la somme des entrées, plus un biais, chacunes multiplié par leurs poids respectifs, le tout passés dans une fonction d'activation.")

st.latex(r'''h(bw_0 + x_1w_1 + x_2w_2 + \dots + x_pw_p) = h \left ( \sum^{i = 0}_p x_iw_i \right ) = y''')

st.header("Qu'est-ce qu'une fonction d'activation ?")

st.write("Une fonction d'activation permet de restreindre le résultat à une borne, par exemple entre 0 et 1 avec une sigmoid.")

st.image("pages/images/sigmoid.png")

st.header("Réseau de neurones")

st.write("Pour créer un réseau de neurones on va agencer les neurones en 'couche'. Reliant les neurones les uns aux autres, comme par exemple :")

st.image("pages/images/neural_network.jpeg")

st.write("Ensuite on va entrainer le modèle en lui donnant en entré des données dont on a le résultat attendue. Ceci afin de modifier les poids de chacun des neurones petit-à-petit juqu'à ce que le réseau ait des résultats suffisamments satisfaisants.")
