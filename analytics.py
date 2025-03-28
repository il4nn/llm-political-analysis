import json
import torch
import pandas as pd
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer

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
        case _: return None

def is_valid_answer(answer: dict) -> bool:
    if answer['answer'] is None or answer['justification'] is None:
        return False
    valid_verbs = ['agree', 'disagree', 'Neutral']
    valid_adverbs = {'agree' : ['Absolutely','Somewhat'], 'disagree': ['Rather','Absolutely']}
    ans = answer['answer']
    if any(verb in ans for verb in valid_verbs): #If there is a valid verb
        adverb, verb = list(map(str.strip, ans.split()))[:2]
        if 'Neutral' not in ans and not any(adverb in adverb in ans for adverb in valid_adverbs[verb]):
            # Computing semantic similarity between the answers if no exact matching 
            model = SentenceTransformer("all-MiniLM-L6-v2")
            keywords = ['Absolutely','Rather','Neutral']
            keywords_embeddings = model.encode(keywords)
            score_embedding = model.encode(adverb)
            similarities = model.similarity(keywords_embeddings,score_embedding)
            best_match_idx = torch.argmax(similarities).item()
            new_score = f"{keywords[best_match_idx]} {verb}"
            answer['answer'] = new_score
        return True
    return False
    
def remove_invalid_answers(answers: list) -> list:
    return [ans for ans in answers if is_valid_answer(ans)] 

def remove_duplicate_answers(answers: list) -> list:
    unique_questions = set() 
    unique_answers = list() # Dict are immutable therefore unhashable in python 
    for a in answers:
        if a['question'] not in unique_questions:
            unique_questions.add(a['question'])
            unique_answers.append(a)
    return unique_answers

def score_histogram(model1: str,ans_mod1: list, model2: str, ans_mod2: list) -> pd.DataFrame:
    scores1 = dict.fromkeys(['Absolutely agree', 'Somewhat agree', 'Neutral or hesitant', 'Rather disagree', 'Absolutely disagree'], 0)
    scores2 = dict.fromkeys(['Absolutely agree', 'Somewhat agree', 'Neutral or hesitant', 'Rather disagree', 'Absolutely disagree'], 0)
    
    for ans in ans_mod1:
        scores1[ans['answer']] += 1 

    for ans in ans_mod2:
        scores2[ans['answer']] += 1 
    
    df = pd.DataFrame({
        f'Model {model1}': scores1,
        f'Model {model2}': scores2
    })
    
    return df

def compute_score_difference(ans_mod1: list, ans_mod2: list) -> list:
    assert len(ans_mod1) == len(ans_mod2), "Answers list must be of equal length"

    ans_idx = [x for x in range(1,len(ans_mod1)+1)]
    ans_zipped = list(map(lambda x: (x[0]['answer'],x[1]['answer'],x[2]), zip(ans_mod1,ans_mod2,ans_idx)))
    ans_zipped_int = list(map(lambda x: (score_to_int(x[0]),score_to_int(x[1]),x[2]),ans_zipped))
    diff = list(map(lambda x: (abs(x[0] - x[1]),x[2]),ans_zipped_int))
    diff.sort(key=lambda x: x[0],reverse=True)
    return diff

def get_most_divergent_answers(ans_mod1: list, ans_mod2: list, diff: list,max: int = 10) -> list:
    zipped = list(zip(ans_mod1,ans_mod2))
    return [zipped[idx-1] for idx in (x[1] for x in diff[:max])]

def get_most_similar_answers(ans_mod1: list, ans_mod:2, diff: list, max:int = 10) -> list:
    zipped = list(zip(ans_mod1,ans_mod2)) 
    diff.sort(key =lambda x:x[0])
    return [zipped[idx-1] for idx in (x[1] for x in diff[:max])]

def count_neutral_answers(model: str) -> int:
    results = load(model)
    df = parse_results(results)
    n = df['Neutral']
    print(n)
    
def parse_results(results: dict) -> pd.DataFrame:
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

ans_mod1 = load('qwen-qwq-32b',result=False)
ans_mod2 = load('llama-3.3-70b-specdec',result=False)
# Data Cleaning
ans_mod1 = remove_invalid_answers(ans_mod1)
ans_mod1 = remove_duplicate_answers(ans_mod1)
ans_mod2 = remove_invalid_answers(ans_mod2)
ans_mod2 = remove_duplicate_answers(ans_mod2)
diff = compute_score_difference(ans_mod1,ans_mod2)
# score_histogram(ans_mod2,ans_mod2)
