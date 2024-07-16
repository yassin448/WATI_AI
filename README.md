# WATI Assistant README

Welcome to WATI (What AI)! This is an AI assistant programmed to handle various types of questions and tasks. Below are details on how to set up and use the assistant effectively.

## Features

- **Dataset Handling**: Utilizes a dataset (`token.csv`) for answering frequently asked questions.
- **Mathematical Queries**: Solves arithmetic and algebraic equations using natural language processing (NLP).
- **Fallback to Text Generation**: Uses a pre-trained text generation model (GPT-2) when specific answers are not found in the dataset or for general conversation.
- **Feedback Mechanism**: Allows users to provide corrections to answers for continuous improvement.
- **User Interface**: Built with tkinter for a simple, interactive user interface.

## Setup Instructions

1. **Dependencies**:
   - Python 3.x
   - Required Python libraries (install via `pip install <library>`):
     - tkinter
     - pandas
     - spacy
     - sympy
     - textblob
     - concurrent.futures

2. **Dataset Setup**:
   - Ensure `token.csv` is placed in the same directory as the program.
   - The dataset should have columns named `ques` (questions) and `ans` (answers).

3. **Text Generation Model**:
   - Ensure `Aidata.py` contains the text generation model setup (`load_text_generation_model()`).

4. **Run the Program**:
   - Execute `python Ai.py` to start the WATI Assistant.

## Usage

- **Interaction**: Enter questions or commands in the entry field and press Enter or click "Send".
- **Exiting**: Type "exit" to terminate the program.
## Examples

### Example 1: Dataset Query

**User Input**:  
```
What is the capital of France?
```

**Expected Output**:  
```
WATI 2.0: Dataset: Paris
```

### Example 2: Mathematical Query

**User Input**:  
```
What is the square root of 25?
```

**Expected Output**:  
```
WATI 2.0: The result of 'sqrt(25)' is 5.0.
```

### Example 3: Algebraic Equation

**User Input**:  
```
Solve for x: 2x + 5 = 15
```

**Expected Output**:  
```
WATI 2.0: The solution to the equation '2*x + 5 == 15' is {x: 5}.
```

### Example 4: Text Generation Fallback

**User Input**:  
```
Can you tell me about artificial intelligence?
```

**Expected Output**:  
```
WATI 2.0: (Generated response using GPT-2 model)
```

### Example 5: Exit Command

**User Input**:  
```
exit
```

**Expected Output**:  
```
WATI 2.0: Goodbye!
```

## Feedback

- If the assistant provides incorrect information, you can manually correct it during interaction.
- Corrections are saved to `feedback.csv` for future improvements.

## Contributing

Contributions and improvements to the assistant's functionality are welcome. Please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the [MIT License](link-to-your-license-file).

---


