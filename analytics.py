import json
from matplotlib import pyplot as plt
import pandas as pd
from collections import Counter

def analyse_scores(model):
    answers = load_answers(model)
    scores = dict.fromkeys(['Absolutely agree', 'Somewhat agree', 'Neutral or hesitant', 'Rather disagree', 'Absolutely disagree'], 0)
    for answer in answers:
        scores[answer['answer']] +=1 
    
    df = pd.DataFrame.from_dict(scores, orient='index')
    df.plot(kind='bar')
    plt.title(f'Answer histogram for model: {model}')
    plt.xlabel('Answer')
    plt.ylabel('Frequency')
    plt.show()

def load_answers(model) -> dict:
    with open(f'answers_{model}.json', 'r') as f:
        data = json.load(f)
    return data

analyse_scores('llama-3.3-70b-specdec')

