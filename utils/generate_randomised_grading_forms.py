from random import random
import pandas as pd

FORM_COUNT = 5
OUTPUT_DIR = "./dist/"
CSV_FILE_PATH = "./assets/Team6_comparative_analysis.csv"
QUESTION_COLUMN = 0
MODEL_ROW = 1

df = pd.read_csv(CSV_FILE_PATH, header=None)

def get_questions_from_csv():
    return df.iloc[MODEL_ROW + 1 :, QUESTION_COLUMN].dropna().tolist()

def get_models_from_csv(): 
    return df.iloc[MODEL_ROW, QUESTION_COLUMN + 1 :].dropna().tolist()

def randomise_list(input_list):
    randomised_list = input_list.copy()
    randomised_list.sort(key=lambda x: random())
    return randomised_list

def save_form_to_file(form_content, form_number):
    with open(f"{OUTPUT_DIR}form_{form_number}.txt", "w") as file:
        file.write(form_content)

def generate_randomised_forms():
    models = get_models_from_csv()
    questions = get_questions_from_csv()

    for i in range(FORM_COUNT):
        randomised_questions = questions.copy()
        randomised_models = models.copy()
        randomised_questions = randomise_list(randomised_questions)    
        form_content = f"Form {i + 1}\n\n"
        
        for question in randomised_questions:
            form_content += f"Question: {question}\n"

            randomised_models = randomise_list(randomised_models)
            for model in randomised_models:
                form_content += f" - Model: {model}\n"

            form_content += "\n"
        
        save_form_to_file(form_content, i + 1)



if __name__ == "__main__":
    generate_randomised_forms()