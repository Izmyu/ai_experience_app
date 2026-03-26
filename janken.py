import streamlit as st
import pandas as pd
import random

st.title("育てる後出しジャンケンAI")

# 改良版：ユーザーの手に応じた「対応表」としての脳
if 'brain' not in st.session_state:
    st.session_state.brain = {
        "グー":   {"グー": 1.0, "チョキ": 1.0, "パー": 1.0},
        "チョキ": {"グー": 1.0, "チョキ": 1.0, "パー": 1.0},
        "パー":   {"グー": 1.0, "チョキ": 1.0, "パー": 1.0}
    }

# ユーザーの入力
user_move = st.radio("あなたの手を選んでください（AIはこれを見てから出します）", ["グー", "チョキ", "パー"])


if st.button("勝負！"):
    # ボタンが押された瞬間に、今の「ユーザーの手」に対応する重みを取得してAIの手を決定
    possible_actions = st.session_state.brain[user_move] 
    ai_move = random.choices(list(possible_actions.keys()), 
                             weights=list(possible_actions.values()))[0]
    
    # --- 勝敗判定と学習ロジック ---
   # --- 勝敗判定と学習ロジック ---
    if (ai_move == "グー" and user_move == "チョキ") or \
       (ai_move == "チョキ" and user_move == "パー") or \
       (ai_move == "パー" and user_move == "グー"):
        
        # AIの手を表示
        st.write(f"<p style='text-align: center;'>AIの手: <b>{ai_move}</b></p>", unsafe_allow_html=True)
        # 勝敗を中央に大きく表示（緑色）
        st.markdown("<h1 style='text-align: center; color: #28a745;'>AIの勝ち！</h1>", unsafe_allow_html=True)
        
        st.session_state.brain[user_move][ai_move] += 0.5 

    elif ai_move == user_move:
        st.write(f"<p style='text-align: center;'>AIの手: <b>{ai_move}</b></p>", unsafe_allow_html=True)
        # あいこを中央に大きく表示（オレンジ色）
        st.markdown("<h1 style='text-align: center; color: #ffc107;'>あいこ</h1>", unsafe_allow_html=True)
        
    else:
        st.write(f"<p style='text-align: center;'>AIの手: <b>{ai_move}</b></p>", unsafe_allow_html=True)
        # あなたの勝ちを中央に大きく表示（赤色）
        st.markdown("<h1 style='text-align: center; color: #dc3545;'>あなたの勝ち！</h1>", unsafe_allow_html=True)
        
        st.session_state.brain[user_move][ai_move] -= 0.2
        if st.session_state.brain[user_move][ai_move] < 0.1:
            st.session_state.brain[user_move][ai_move] = 0.1

# 可視化セクション
st.divider()
st.write("### AIの『対応表』の成長具合")

cols = st.columns(3)
for i, (user_hand, actions) in enumerate(st.session_state.brain.items()):
    with cols[i]:
        st.write(f"あなたが『{user_hand}』の時")
        df = pd.DataFrame(list(actions.items()), columns=["AIの手", "重み"])
        st.bar_chart(df.set_index("AIの手"))