import json
import pandas as pd
from matplotlib import pyplot as plt


def load(model: str, result: bool = True) -> dict:
    filename = f"results/{'results' if result else 'answers'}_{model}.json"
    with open(filename, 'r') as f:
       return json.load(f)

def score_to_int(score: str) -> int:
    match score:
        case 'Absolutely agree': return 5
        case 'Somewhat agree': return 4 
        case 'Neutral or hesitant': return 3
        case 'Rather disagree': return 2
        case 'Absolutely disagree': return 1
    
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
    
def compute_score_diff(model1: str, model2: str, max: int = 10) -> list:
    ans_mod1 = load(model1,result=False)
    ans_mod2 = load(model1,result=False) 

    assert len(ans_mod1) == len(ans_mod2)

    ans_zipped = list(map(lambda x: (x[0]['answer'],x[1]['answer']), zip(ans_mod1,ans_mod1)))
    print(ans_zipped)
    ans_zipped_int = list(map(lambda x: (score_to_int(x[0]),score_to_int(x[1])),ans_zipped))
    diff = list(map(lambda x: abs(x[0] - x[1]),ans_zipped_int))
    diff.sort()
    return diff[:max]

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
            "Left": [perc[0] + "%" for perc in percentages],
            "Neutral": [perc[1] + "%"  for perc in percentages],
            "Right": [perc[2] + "%"  for perc in percentages]
         }
    )
    return df

r = compute_score_diff('llama-3.3-70b-specdec',' n')
print(r)