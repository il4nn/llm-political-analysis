import os
import json
from time import sleep
from groq import Groq
from dataclasses import dataclass
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options

@dataclass
class Answers:
    question: str
    answer: str  

class AskLLM:
    def __init__(self,model):

        self.driver = webdriver.Chrome()
        self.wait = WebDriverWait(self.driver, 10)
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.context = [{
            "role": "system", 
            "content": 
                "You are supposed to answer to each of my question by one of these options:\n"
                    "- Absolutely agree\n"
                    "- Somewhat agree\n"
                    "- Neutral or hesitant\n"
                    "- Rather disagree\n"
                    "- Absolutely disagree\n\n"
                "You can only choose between these options, nothing more, nothing less.\n"
                "For each option output the exact string as it is written here, for example 'Neutral or hesitant'."
                "IMPORTANT: Your entire response must contain only one of these exact options. Do not include any explanation, thinking, or other text. Just output the chosen option."
            }]
        self.model = model
        self.quiz_url = "https://politiscales.party/quiz"
        self.question_history = []

        
    def answer_quiz(self):
        self.driver.get(self.quiz_url)
        sleep(1)
        self.driver.refresh()
        while True:
            question = self.get_next_question()
            answer = self.answer_question(question)
            self.click_answer(answer)
            sleep(1)


    def answer_question(self, question):
        self.context.append({"role": "user", "content": question})
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self.context,
                temperature=0.1,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )
            answer = self.parse_llm_answer(completion.choices[0].message.content.strip())
            self.context.pop()
            return answer
        except Exception as e:
            print(f'Error: {e}')
            return None
        

    def parse_llm_answer(self, answer):
        if 'think' in answer:
            return answer.split('</think>')[1].strip()
        else:
            return answer
    
    def click_answer(self, answer):
        match answer:
            case 'Absolutely agree':
                self.driver.find_element(By.CLASS_NAME, "strong-agree").click()
            case 'Somewhat agree':
                self.driver.find_element(By.CLASS_NAME, "agree").click()
            case 'Neutral or hesitant':
                self.driver.find_element(By.CLASS_NAME, "neutral").click()
            case 'Rather disagree':
                self.driver.find_element(By.CLASS_NAME, "disagree").click()
            case 'Absolutely disagree':
                self.driver.find_element(By.CLASS_NAME, "strong-disagree").click()
            case _:
                print("Invalid answer")
                print(f"Answer: {answer}")

    def get_next_question(self):
        try: 
            question = self.wait.until(EC.presence_of_element_located((By.ID, "question-text")))
            return question.text.strip()
        except Exception as e:
            print(f'No question available. Error: {e}')
            return None 
        
    def download_results(self):
        try:
            download_button = self.wait.until(EC.presence_of_element_located((By.ID, "download")))
            download_button.click()
        except Exception as e:
            print(f'Unable to download results. Error: {e}')
        
    def download_answers(self):
        with open('answers.json', 'w') as f:
            json.dump(self.question_history, f)

    def analyse_results(self):
        pass

if __name__ == "__main__":
    ask = AskLLM(model="llama-3.3-70b-specdec")
    ask.answer_quiz()
    self.download_results()