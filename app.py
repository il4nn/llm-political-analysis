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
    st.write(f'{model1} {"🇨🇳" if models[model1] == "CN" else "🇺🇸"} \u2003 VS \u2003 {model2} {"🇨🇳" if models[model2] == "CN" else "🇺🇸"}')
    
    results_model1 = load(model1)
    df1 = parse_results(results_model1)
    st.dataframe(df1, hide_index=True)


    results_model2 = load(model2)
    df2 = parse_results(results_model2)
    st.dataframe(df2, hide_index=True)

    ans_mod1 = load(model1,result=False)
    ans_mod2 = load(model2,result=False)
    ans_mod1 = remove_null_answers(ans_mod1)
    ans_mod1 = remove_duplicate_answers(ans_mod1)
    ans_mod2 = remove_null_answers(ans_mod2)
    ans_mod2 = remove_duplicate_answers(ans_mod2)
    
    diff = compute_score_difference(ans_mod1,ans_mod2)
    
    st.write('Explore Questions with the Greatest Disagreement')
    num_questions = st.slider(
        "Number of questions",
        min_value=1,
        max_value=30,
        value=10
    )
    div = get_most_divergent_answers(ans_mod1,ans_mod2,diff,max=num_questions)
    for ans in div: 
        st.write(ans[0]['question'])




