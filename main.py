# main_app.py (ホームページ) のイメージ
import streamlit as st
st.title("AI体験パークへようこそ！")
st.write("ここでは、AIが学習する過程を直感的に体験できます。")

col1, col2 = st.columns(2)
with col1:
    st.subheader("じゃんけんAI")
    if st.button("じゃんけんを教えに行く"):
        st.switch_page("pages/janken.py") # 指定したページへ移動

with col2:
    st.subheader("画像認識AI")
    if st.button("画像認識を体験する"):
        st.switch_page("pages/neural_experience.ipynb")
