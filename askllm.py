import os
from time import sleep
from groq import Groq
from selenium_stealth import stealth
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options


class AskLLM:
    def __init__(self):

        self.driver = webdriver.Chrome()
        self.wait = WebDriverWait(self.driver, 10)
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.context = [{
            "role": "system", 
            "content": 
                "You are supposed to answer to each of my question by one of these options: Absolutely agree, Somewhat agree, Neutral or hesitant, Rather disagree, Absolutely disagree"
                "You can only choose between these options, nothing more, nothing less."
            }]
        self.model = "llama-3.3-70b-specdec"
        self.quiz_url = "https://politiscales.party/quiz"

        stealth(self.driver,
                languages=["en"],
                vendor="Google Inc.",
                platform="Win32",
                webgl_vendor="Intel Inc.",
                renderer="Intel Iris OpenGL Engine",
                fix_hairline=True,
                )
        
    def open_website(self):
        self.driver.get(self.quiz_url)
        sleep(0.1)
        question = self.get_next_question()
        self.answer_question(question)
        input("Press Enter to continue...")

    def answer_question(self, question):
        self.context.append({"role": "user", "content": question})
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self.context,
                temperature=1,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )
            answer = completion.choices[0].message.content
            print(f"Answer: {answer}")

        except Exception as e:
            print(f'Error: {e}')


    
    def get_next_question(self):
        question = self.wait.until(EC.presence_of_element_located((By.ID, "question-text")))
        return question.text

if __name__ == "__main__":
    ask = AskLLM()
    ask.open_website()