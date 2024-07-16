import tkinter as tk
from tkinter import scrolledtext
import pandas as pd
import spacy
import re
import os
import math
import time
import random
from sympy import symbols, Eq, solve
import concurrent.futures
import Aidata as fallbackai
import csv
from textblob import TextBlob
import re
# Load spaCy model with word vectors (for semantic similarity)
nlp = spacy.load("en_core_web_sm")

# Model information
MODEL_NAME = "WATI"
MODEL_VERSION = "2.0"
fallbackai.load_text_generation_model()  # Loads the fallback model (GPT-2)

# Ensure feedback.csv file with required headers
def ensure_feedback_csv():
    feedback_file = 'feedback.csv'
    if not os.path.exists(feedback_file):
        df = pd.DataFrame(columns=['Prompt', 'Corrected Answer'])
        df.to_csv(feedback_file, index=False)
        print(f"Created {feedback_file} with required headers.")

# Function to preprocess text using spaCy NLP
def preprocess_text(text):
    doc = nlp(text.lower())
    processed_text = " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])
    return processed_text

# Function to load dataset with data review mechanism
def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        data.drop_duplicates(inplace=True)
        if data.isnull().any().any():
            raise ValueError("Dataset contains missing values.")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame(columns=['ques', 'ans'])

# Function to update feedback CSV
def update_feedback_csv(prompt, corrected_answer):
    feedback_file = 'feedback.csv'
    fieldnames = ['Prompt', 'Corrected Answer']
    try:
        with open(feedback_file, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'Prompt': prompt, 'Corrected Answer': corrected_answer})
    except IOError:
        print(f"Error writing to {feedback_file}")

# Function to update dataset entry or add new entry
def update_entry(dataset_path, question, answer):
    try:
        data = pd.read_csv(dataset_path)
        processed_question = preprocess_text(question)
        existing_entry = data['ques'].apply(preprocess_text) == processed_question
        if existing_entry.any():
            data.loc[existing_entry, 'ans'] = answer
            print(f"Updated existing entry for question: {question}")
        else:
            new_entry = pd.DataFrame({'ques': [question], 'ans': [answer]})
            data = pd.concat([data, new_entry], ignore_index=True)
            print(f"Added new entry for question: {question}")
        data.to_csv(dataset_path, index=False)
        print("Dataset updated successfully.")
    except pd.errors.EmptyDataError:
        new_data = pd.DataFrame({'ques': [question], 'ans': [answer]})
        new_data.to_csv(dataset_path, index=False)
        print("Dataset was empty. Created new dataset with the provided entry.")
    except FileNotFoundError:
        new_data = pd.DataFrame({'ques': [question], 'ans': [answer]})
        new_data.to_csv(dataset_path, index=False)
        print("Dataset file not found. Created new dataset with the provided entry.")
    except Exception as e:
        print(f"Error updating dataset entry: {e}")

# Function to solve arithmetic/math questions using NLP
def solve_math(question):
    math_pattern = r'([\d+\-*/\s\^()]+|sqrt\(\d+\)|log\(\d+\)|sin\(\d+\)|cos\(\d+\)|tan\(\d+\))'
    match = re.search(math_pattern, question)
    if match:
        try:
            result = eval(match.group(1), {"__builtins__": None}, {"sqrt": math.sqrt, "log": math.log, "sin": math.sin, "cos": math.cos, "tan": math.tan})
            return f"The result of '{match.group(1)}' is {result}."
        except Exception as e:
            print(f"Error solving math expression: {e}")
            return None
    return None

# Function to solve algebraic equations using NLP
def solve_equation(question):
    equation_pattern = r'([a-zA-Z]+\s*=\s*[\d+\-*/\s\^()]+|[\d+\-*/\s\^()]+\s*=\s*[a-zA-Z]+)'
    matches = re.findall(equation_pattern, question)
    if matches:
        try:
            eq = matches[0].replace('=', '==')
            lhs, rhs = eq.split('==')
            var = re.findall(r'[a-zA-Z]', lhs + rhs)[0]
            var = symbols(var)
            equation = Eq(eval(lhs), eval(rhs))
            solution = solve(equation, var)
            return f"The solution to the equation '{matches[0]}' is {solution}."
        except Exception as e:
            print(f"Error solving equation: {e}")
            return None
    return None

# Function to get answer from dataset with best matching question
def get_answer(question, data):
    processed_question = preprocess_text(question)
    if not data.empty:
        data['similarity'] = data['ques'].apply(lambda x: nlp(processed_question).similarity(nlp(preprocess_text(x))))
        best_match_index = data['similarity'].idxmax()
        best_similarity = data.loc[best_match_index, 'similarity']
        if best_similarity > 0.8:
            return data.loc[best_match_index, 'ans']
    return None

# Function to introduce simulated typing delay
def simulate_typing(response, chat_area):
    for char in response:
        chat_area.insert(tk.END, char)
        chat_area.update_idletasks()
        time.sleep(random.uniform(0.02, 0.08))
    chat_area.insert(tk.END, "\n")

# Function to clear the chat area and input
def clear_chat():
    chat_area.delete('1.0', tk.END)
    entry.delete(0, tk.END)
    chat_area.insert(tk.END, f"{MODEL_NAME} {MODEL_VERSION}: Welcome to WATI Assistant! How can I help you today?\n")


# Function to handle the program flow
def handle_question(question):
    if question.lower() == 'exit':
        chat_area.insert(tk.END, "Goodbye!\n")
        return
   
    if "--no dataset" in question.lower():
                # Convert text to lowercase
                question = question.lower()
                # Correct common misspellings
                question = str(TextBlob(str(question)).correct())
                # Remove extra whitespace
                question = re.sub(r'\s+', ' ', question).strip() 
                model, tokenizer = fallbackai.load_text_generation_model()
                response = fallbackai.local_text_generation(question.replace("--no dataset",""), model, tokenizer)
                answer = response
    
    else:
        # Uses Multithreading for quick answering
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            future_math = executor.submit(solve_math, question)
            future_equation = executor.submit(solve_equation, question)
            future_answer = executor.submit(get_answer, question, data)
            
            math_answer = future_math.result()
            equation_answer = future_equation.result()
            dataset_answer = future_answer.result()
            
            if dataset_answer:
                answer = "Dataset: " + dataset_answer
            elif math_answer:
                answer = math_answer
            elif equation_answer:
                answer = equation_answer
            else:
                model, tokenizer = fallbackai.load_text_generation_model()
                response = fallbackai.local_text_generation(question, model, tokenizer)
                answer = response
    
    # Print the answer with simulated typing effect
    simulated_response = f"{MODEL_NAME} {MODEL_VERSION}: {answer}\n"
    simulate_typing(simulated_response, chat_area)
    
    # Collect feedback
    user_feedback = input("Is the generated text correct? (y/n/s): ")
    if user_feedback.lower() == 'n':
        correct_answer = input("Please provide the correct answer: ")
        update_feedback_csv(question, correct_answer)
        print("Thank you for your feedback! It has been saved.")
    elif user_feedback.lower() == 'y':
        print("Thank you for confirming the generated text.")
    else:
        pass  # Handle other cases if needed

# Function for when "Enter" key is pressed
def on_enter(event):
    question = entry.get().strip()
    if question:
        chat_area.insert(tk.END, f"You: {question}\n")
        handle_question(question)
        entry.delete(0, tk.END)

# Ensure CSV and dataset are loaded correctly
dataset_path = 'token.csv'
data = load_dataset(dataset_path)
ensure_feedback_csv()

# Create main window
root = tk.Tk()
root.title("WATI Assistant")
root.configure(bg='#1e1e1e')

# Create chat display area
chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, bg='#1e1e1e', fg='white', font=('Helvetica', 12))
chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
chat_area.insert(tk.END, f"{MODEL_NAME} {MODEL_VERSION}: Welcome to WATI Assistant! How can I help you today?\n")

# Create entry widget
entry_frame = tk.Frame(root, bg='#1e1e1e')
entry_frame.pack(padx=10, pady=10, fill=tk.X)
entry = tk.Entry(entry_frame, bg='#2e2e2e', fg='white', font=('Helvetica', 12))
entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
entry.bind("<Return>", on_enter)

# Create send button
send_button = tk.Button(entry_frame, text="Send", command=lambda: on_enter(None), bg='#2e2e2e', fg='white', font=('Helvetica', 12))
send_button.pack(side=tk.RIGHT, padx=5)

# Create clear button
clear_button = tk.Button(root, text="Clear", command=clear_chat, bg='#2e2e2e', fg='white', font=('Helvetica', 12), width=10)
clear_button.pack(padx=10, pady=10)



# Start the the ui loop
root.mainloop()
