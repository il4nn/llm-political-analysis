import json
import pandas as pd
from matplotlib import pyplot as plt

def analyse_scores(model: str):
    answers = load(model,results=False)
    scores = dict.fromkeys(['Absolutely agree', 'Somewhat agree', 'Neutral or hesitant', 'Rather disagree', 'Absolutely disagree'], 0)
    for answer in answers:
        scores[answer['answer']] +=1 
    
    df = pd.DataFrame.from_dict(scores, orient='index')
    df.plot(kind='bar')
    plt.title(f'Answer histogram for model: {model}')
    plt.xlabel('Answer')
    plt.ylabel('Frequency')
    plt.show()

def load(model: str, result: bool = True) -> dict:
    filename = f"results/{'results' if result else 'answers'}_{model}.json"
    with open(filename, 'r') as f:
       return json.load(f)

def find_diff(model1: str, model2: str):
    ans1 = load(model1,results=False)
    ans1 = load(model2,results=False) 

def parse_results(model: str) -> pd.DataFrame:
    results = load(model)
    categories = list()
    percentages = list()
    for key in results.keys():
        categories.append(key)
        l,r = map(str.strip, key.split('vs'))
        p = list(map(lambda x: ''.join(filter(str.isdigit,x)), results[key].values()))
        p.insert(1,str(100 - int(p[0]) - int(p[1])))
        percentages.append(p)

    df = pd.DataFrame(
        {
            "Categories": categories,
            "Left": [str(perc[0]) + "%" for perc in percentages],
            "Neutral": [perc[1] + "%"  for perc in percentages],
            "Right": [perc[2] + "%"  for perc in percentages]
         }
    )
    return df

parse_results('llama-3.3-70b-specdec')