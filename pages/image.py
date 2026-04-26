from streamlit_drawable_canvas import st_canvas
import streamlit as st

# スマホでも書きやすいキャンバスを設置
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

import numpy as np
import pandas as pd
import scipy.special
import cv2
import torch

def preprocess_image(image_data):
    # 1. RGBAからRGBへ変換し、さらにグレースケール（白黒）へ
    img = image_data[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2. 28x28ピクセルにリサイズ
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # 3. 0~255の数値を 0.0~1.0 の範囲に正規化
    img = img.astype(np.float32) / 255.0
    
    # 4. 1次元のベクトルに変換
    img = img.flatten()

class neuralNetwork:

    # ニューラルネットワークの初期化
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 入力層、隠れ層、出力層のノード数の設定
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 学習率の設定
        self.lr = learningrate

        # リンクの重み行列wih とwho
        # 行列内の重み w_i_j, ノードiからノードjへの重み
        # w11 w21
        # w12 w22 などq
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 活性化関数はシグモイド関数
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # ニューラルネットワークの学習
    def train(self, inputs_list, targets_list):
        # 入力リストを行列に変換
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # 隠れ層に入ってくる信号の計算
        hidden_inputs = np.dot(self.wih, inputs)
        # 隠れ層で結合された信号を活性化関数により出力
        hidden_outputs = self.activation_function(hidden_inputs)
        # 出力層に入ってくる信号の計算
        final_inputs = np.dot(self.who, hidden_outputs)
        # 出力層で結合された信号を活性化関数により出力
        final_outputs = self.activation_function(final_inputs)

        # 出力層の誤差 = (目標値 - 最終出力)
        output_errors = targets - final_outputs

        # 隠れ層の誤差は、出力層の誤差をリンクの重みの割合で分配
        hidden_errors = np.dot(self.who.T, output_errors)

        # 隠れ層と出力層の間のリンクの重みを更新
        self.who += self.lr * np.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            np.transpose(hidden_outputs))
        
        # 入力層と隠れ層の間のリンクの重みを更新
        self.wih += self.lr * np.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            np.transpose(inputs))
        pass

    # ニューラルネットワークへの照会
    def query(self, inputs_list):
        #　入力リストを行列に変換
        inputs = np.array(inputs_list, ndmin=2).T

        # 隠れ層に入ってくる信号の計算
        hidden_inputs = np.dot(self.wih, inputs)
        # 隠れ層で結合された信号を活性化関数により出力
        hidden_outputs = self.activation_function(hidden_inputs)

        # 出力層に入ってくる信号の計算
        final_inputs = np.dot(self.who, hidden_outputs)
        # 出力層で結合された信号を活性化関数により出力
        final_outputs = self.activation_function(final_inputs)

        if final_outputs[0] > final_outputs[1] and final_outputs[0] > final_outputs[2]:
            answer = "1"
        elif final_outputs[1] > final_outputs[0] and final_outputs[1] > final_outputs[2]:
            answer = "4"
        else:
            answer = "9"
            

        return answer
    
# モデルがまだ保存されていない場合のみ、新しく作る
if 'model' not in st.session_state:
    # クラスのインスタンスを生成して保存
    st.session_state.model = neuralNetwork(
        inputnodes=784, 
        hiddennodes=70, 
        outputnodes=3, 
        learningrate=0.3
    )

# 以降、この「保存されたモデル」を使って学習や予測を行う
n = st.session_state.model


if canvas_result.image_data is not None:
    # ユーザーが「学習開始」ボタンを押した時
    if st.button("この数字をAIに教える"):
        # 1. 画像を整形
        input = preprocess_image(canvas_result.image_data)
        
        # 2. モデルに入力して予測
        output = n.query(input)
        
        st.write(f"AIの予測: {output}")

     
        
        # 3. 重みの更新（学習）
        # ここで正解ラベル（1, 4, 9のどれか）を渡して誤差逆伝播を行います
else:
    st.write("キャンバスに数字を描いて、学習開始ボタンを押してください！")

user_ans = st.radio("あなたの書いた数字を教えてください（AIはこの結果を基に学習します）", ["1", "4", "9"])

if user_ans == "1":
    target = [0.99, 0.01, 0.01]
elif user_ans == "4":
    target = [0.01, 0.99, 0.01]
else:
    target = [0.01, 0.01, 0.99]

n.train(input, target)