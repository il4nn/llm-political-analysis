from analytics import *
import streamlit as st
import pandas as pd  

st.set_page_config(layout='wide')

#Decorator to cache  
@st.cache_data
def load_and_process_answers(model:str) -> list:
    ans = load(model,result=False)
    return remove_duplicate_answers(remove_invalid_answers(ans))

models =  {
    'llama-3.3-70b-specdec': 'USA',
    'qwen-qwq-32b': 'CN'
}

left, center, right = st.columns([1, 2, 1])
with center:
    st.title('LLM Political Analysis')

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
    st.subheader(f'You chose:')
    st.subheader(f' {model1} {"🇨🇳" if models[model1] == "CN" else "🇺🇸"} \u2003 VS \u2003 {model2} {"🇨🇳" if models[model2] == "CN" else "🇺🇸"}')
        
    results_model1 = load(model1)
    df1 = parse_results(results_model1)

    results_model2 = load(model2)
    df2 = parse_results(results_model2)

    col1,col2 = st.columns([1,1])
    with col1:
        st.dataframe(df1, hide_index=True,width=1000)
    with col2:
        st.dataframe(df2, hide_index=True,width=1000)


    ans_mod1 = load_and_process_answers(model1)
    ans_mod2 = load_and_process_answers(model2)
    
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.subheader(f'Explore the Distribution of Answers for Each Model')
        source = score_histogram(model1,ans_mod1,model2,ans_mod2)
        st.bar_chart(source, stack=False)
    
    left, right = st.columns([1,1])
    diff = compute_score_difference(ans_mod1,ans_mod2)
    with left: 
        st.subheader('Explore Questions with the Greatest Disagreement')

        div = get_most_divergent_answers(ans_mod1,ans_mod2,diff,max=10)
        for ans in div: 
            j0 = ans[0]['justification']
            j1 = ans[1]['justification']
            with st.expander(f"{ans[0]['question']} {'(Justification)'}"): 
                st.write(f'**Justification for {model1}**: {j0}')
                st.write(f'**Justification for {model2}**: {j1}')
                df = pd.DataFrame({
                    f'Score {model1}': ans[0]['answer'],
                    f'Score {model2}': ans[1]['answer']
                },index=[0])
                st.dataframe(df,hide_index=True)

    with right: 
        st.subheader('Explore Questions with the Greatest Consensus')

        cons = get_most_similar_answers(ans_mod1,ans_mod2,diff,max=10)
        for ans in cons: 
            j0 = ans[0]['justification']
            j1 = ans[1]['justification']
            with st.expander(f"{ans[0]['question']} {'(Justification)'}"): 
                st.write(f'**Justification for {model1}**: {j0}')
                st.write(f'**Justification for {model2}**: {j1}')
                df = pd.DataFrame({
                    f'Score {model1}': ans[0]['answer'],
                    f'Score {model2}': ans[1]['answer']
                }, index=[0])
                st.dataframe(df,hide_index=True)






