from streamlit_drawable_canvas import st_canvas
import streamlit as st

# 解説ポップアップの定義
@st.dialog("画像処理の操作方法")
def show_explanation():
    st.write("### 🎨 1, 4, 9のどれかを書いて予測！")
    st.image("https://thumb.ac-illust.com/d5/d54404572b4528dfc071e79b57413724_w.jpeg", caption="MNISTデータセットの例")
    
    st.markdown("""
    1. **1, 4, 9のどれかを書く！**: キャンバスに数字を描いて、AIに予測させましょう。
    2. **『この数字をAIに予測させる』をクリック！**: AIがあなたの描いた数字を予測します。
    3. **自分の入力した数字を選択して『正解を学習させる』をクリック！**:だいたい10~20回くらい学習させると、AIの予測精度が上がってくる
    4. **もう一度数字を書いて何回も繰り返そう！**:
    
    """)

    if st.button("理解しました！"):
        st.session_state.explanation_closed = True
        st.rerun()

# 2. フラグの初期化
if "explanation_closed" not in st.session_state:
    st.session_state.explanation_closed = False

# 3. フラグがFalse（閉じていない）ならダイアログを表示
if not st.session_state.explanation_closed:
    show_explanation()

# スマホでも書きやすいキャンバスを設置
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=20,
    stroke_color="#000000",
    background_color="#FFFFFF",
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
import matplotlib.pyplot as plt
import networkx as nx

st.session_state.weights = [[], []]  # 初期重みをセッションステートに保存

def preprocess_image(image_data):
    # 1. RGBAからRGBへ変換し、さらにグレースケール（白黒）へ
    img = image_data[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2. 14x14ピクセルにリサイズ
    img = cv2.resize(img, (14, 14), interpolation=cv2.INTER_AREA)
    
    # 3. 0~255の数値を 0.0~1.0 の範囲に正規化
    img = (img.astype(np.float32) / 255.0)  # 0.01を足して0を避ける
    img  = (1.0 - img)*0.98 + 0.01
    # 4. 1次元のベクトルに変換
    img = img.flatten()
    
    # 最後に 14x14 を 196 に一列に並べ替え、NumPy形式にする
    img_flattened = img.reshape(196) 
    return img_flattened

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
        

        st.session_state.weights = [self.wih, self.who]  # 重みをセッションステートに保存
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
        inputnodes=196, 
        hiddennodes=10, 
        outputnodes=3, 
        learningrate=0.3
    )

# 以降、この「保存されたモデル」を使って学習や予測を行う
n = st.session_state.model

# 1. 状態の初期化 (前回の回答で触れた session_state を活用)
if 'last_input' not in st.session_state:
    st.session_state.last_input = None

if 'output' not in st.session_state:
    st.session_state.output = None

if 'user_ans' not in st.session_state:
    st.session_state.user_ans = None

if 'count' not in st.session_state:
    st.session_state.count = 0

if canvas_result.image_data is not None:
    # ユーザーが「学習開始」ボタンを押した時
    if st.button(" この数字をAIに予測させる"):
        
        # 1. 画像を整形
        st.session_state.last_input = preprocess_image(canvas_result.image_data)
        
        # 2. モデルに入力して予測
        st.session_state.output = n.query(st.session_state.last_input)
        
        st.write(f"<p style='text-align: center; font-size: 30px;'>AIの予測: <b font-size='40px'>{st.session_state.output}</b></p>", unsafe_allow_html=True)

     
        
        # 3. 重みの更新（学習）
        # ここで正解ラベル（1, 4, 9のどれか）を渡して誤差逆伝播を行います
else:
    st.write("キャンバスに数字を描いて、学習開始ボタンを押してください！")

if st.session_state.last_input is not None:
    st.divider() # 区切り線
    st.write("#### AIに正解を教えて学習させましょう")
    
    st.session_state.user_ans = st.radio("あなたが書いた数字はどれですか？", ["1", "4", "9"], horizontal=True)
    
    # 正解ラベルの数値化
    label_map = {"1": [0.99, 0.01, 0.01], "4": [0.01, 0.99, 0.01], "9": [0.01, 0.01, 0.99]}
    target = label_map[st.session_state.user_ans]

    if st.button("正解を学習させる"):
        # 保存しておいた入力を使い、重みを更新
        for e in range(6):  # 6回学習させる（任意）
            n.train(st.session_state.last_input, target)

        
        st.success(f"数字 '{st.session_state.user_ans}' として学習が完了しました！もう一度予測させると結果が変わるかもしれません。")

        st.session_state.count += 1
        
        # 学習が終わったら、入力をクリアして連続学習を防ぐ（任意）
        # st.session_state.last_input = None

        st.divider()
        st.write("### AIの予測結果")

        if 'results' not in st.session_state:
            st.session_state.results = []

        # 平均予測の計算
        if 'sum' not in st.session_state:
            st.session_state.sum = 0.0
        st.session_state.sum += 1.0 if st.session_state.user_ans == st.session_state.output else 0.0


        mean = st.session_state.sum / st.session_state.count if st.session_state.count > 0 else 0.0

        # データの追加（タプルで保存されている前提）
        st.session_state.results.insert(0, {
            "あなたの入力": st.session_state.user_ans,
            "AIの予測": st.session_state.output,
            "判定": "正解" if st.session_state.user_ans == st.session_state.output else "不正解",
            "平均精度": f"{mean:.2f}"
        })

        # データフレームとして表示
        df = pd.DataFrame(
    st.session_state.results, 
    columns=["あなたの入力", "AIの予測", "判定", "平均精度"]
)
        st.table(df)

def draw_nn(weights, layers):
    # 重みがまだ空（初期状態）の場合は描画しない
    if not isinstance(weights[0], np.ndarray):
        st.warning("まだ学習が行われていないため、重みを可視化できません。")
        return
    G = nx.DiGraph()
    pos = {}
    node_idx = 0
    
    # 描画上の最大高さを決める（入力層の数に合わせると綺麗です）
    max_height = max(layers) 
    
    for layer_idx, layer_size in enumerate(layers):
        # この層のノード間隔を計算
        # 入力層(196)は間隔1、隠れ層(10)は間隔19.6...という風に調整
        step = max_height / (layer_size + 1)
        
        # 縦方向の開始位置を調整（中央寄せにするため）
        offset = (max_height - (layer_size - 1) * step) / 2

        for i in range(layer_size):
            # y座標を - (i * step + offset) にすることで間隔を広げつつ中央に配置
            pos[node_idx] = (layer_idx, -(i * step + offset))
            G.add_node(node_idx)
            node_idx += 1
            
    current_node = 0
    for l in range(len(layers) - 1):
        for i in range(layers[l]): # 入力側ノード
            for j in range(layers[l+1]): # 出力側ノード
                u = current_node + i
                v = current_node + layers[l] + j
                
                # --- 修正箇所: 行列のインデックスを [j][i] にする ---
                # NumPy行列は (出力, 入力) の形なので weights[l][j, i] でアクセス
                weight = weights[l][j, i]
                
                # 全ての線を引くと重いので、一定以上の強さの線だけ描画する工夫
                if abs(weight) > 0.01: 
                    width = abs(weight) * 2  # 太さ調整
                    color = '#000000' if weight > 0 else 'gray'
                    G.add_edge(u, v, weight=weight, width=width, color=color)
                    
        current_node += layers[l]

    # Streamlitで表示するために plt.figure を管理
    fig = plt.figure(figsize=(10, 8))
    edges = G.edges()
    widths = [G[u][v]['width'] for u,v in edges]
    colors = [G[u][v]['color'] for u,v in edges]

    nx.draw(G, pos, with_labels=False, node_size=50, node_color='skyblue', 
            edge_color=colors, width=widths, arrowsize=5, alpha=0.6)
    
    st.pyplot(fig) # plt.show() ではなく st.pyplot() を使う
layers = [196, 10, 3]

draw_nn(st.session_state.weights, layers)
