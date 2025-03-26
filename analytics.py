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
        # Semantic similarity if no exact matching 
        case _ :
            adverb,verb = map(str.strip, score.split())
            model = SentenceTransformer("all-MiniLM-L6-v2")
            keywords = ['Absolutely','Rather','Neutral']
            keywords_embeddings = model.encode(keywords)
            score_embedding = model.encode(adverb)
            similarities = model.similarity(keywords_embeddings,score_embedding)
            best_match_idx = torch.argmax(similarities).item()
            new_score = f"{keywords[best_match_idx]} {verb}"
            return score_to_int(new_score)

# For Pre-processing
def is_valid_answer(ans: dict) -> bool:
    if ans['answer'] is None or ans['justification'] is None:
        return False
    valid_keywords = ['agree', 'disagree', 'neutral']
    return any(keyword in ans['answer'].lower() for keyword in valid_keywords)
    
def remove_null_answers(answers: list) -> list:
    return [ans for ans in answers if is_valid_answer(ans)] 

def remove_duplicate_answers(answers: list) -> list:
    unique_questions = set() 
    unique_answers = list() # Dict are immutable therefore unhashable in python 
    for a in answers:
        if a['question'] not in unique_questions:
            unique_questions.add(a['question'])
            unique_answers.append(a)
    return unique_answers

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

def compute_score_diff(ans_mod1: str, ans_mod2: str, max: int = 10) -> list:
    assert len(ans_mod1) == len(ans_mod2)

    ans_idx = [x for x in range(1,len(ans_mod1)+1)]
    ans_zipped = list(map(lambda x: (x[0]['answer'],x[1]['answer'],x[2]), zip(ans_mod1,ans_mod2,ans_idx)))
    # print(list(zip(ans_mod1,ans_mod2,ans_idx)))
    ans_zipped_int = list(map(lambda x: (score_to_int(x[0]),score_to_int(x[1]),x[2]),ans_zipped))
    # print(ans_zipped_int)
    diff = list(map(lambda x: abs(x[0] - x[1]),ans_zipped_int))
    diff.sort()
    # print(diff)
    return diff[:max]

# def merge(ans_mod1: list, ans_mod2: list) -> list:
#     assert len(ans_mod1) == len(ans_mod2)
#     # q_num = [x for x in range(1,len(ans_mod1) + 1)]
#     l = list()
#     for tuple in list(zip(ans_mod1,ans_mod2)):
#         # question = tuple[0]['question']
#         # e = {question,}
#         l.append({for k,v in triples[0].items()})    

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
ans_mod1 = remove_null_answers(ans_mod1)
ans_mod1 = remove_duplicate_answers(ans_mod1)
ans_mod2 = remove_null_answers(ans_mod2)
ans_mod2 = remove_duplicate_answers(ans_mod2)

compute_score_diff(ans_mod1,ans_mod2)
