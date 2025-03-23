from analytics import *
import streamlit as st
import pandas as pd  

st.title('Hello World!')

models = st.selectbox(
    'Choose a model?',
    ('llama-3.3-70b-specdec', 'llama-3.3-70b', 'llama-3.3-70b-ftb')
)

st.write(f'You selected: {models}')

match models:
    case 'llama-3.3-70b-specdec': 
        df = parse_results(models)
        st.dataframe(df)
    case _:
        pass


