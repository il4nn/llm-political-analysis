from analytics import *
import streamlit as st
import pandas as pd  

st.title('LLM Political Analysis')

models =  {
    'llama-3.3-70b-specdec': 'USA',
    'qwen-qwq-32b': 'CN'
}

model1 = st.selectbox(
    'Choose a first model?',
    options=models.keys(),
    index=None
)

model2 = st.selectbox(
    'Choose a second model to compare to',
    options=(m for m in models.keys() if m != model1),
    index=None
    )

if model1 and model2:
    st.write(f'{model1} {"🇨🇳" if models[model1] == "CN" else "🇺🇸"} VS    {model2} {"🇨🇳" if models[model2] == "CN" else "🇺🇸"}')
    
    df1 = parse_results(model1)
    st.dataframe(df1, hide_index=True)

    df2 = parse_results("qwen-qwq-32b")
    st.dataframe(df2, hide_index=True)

