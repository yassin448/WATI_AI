import logging
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline
import Aitrainer as at
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import Aitrainer as at

def load_text_generation_model():
    try:
        
      modelname=f"output/finetunedv{at.modelver-1}"  #not used right now
      tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2") #finetuned version is not used right now due to a problem
      model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
      return model , tokenizer
    except Exception as e:
        logger.error(f"Error loading model/tokenizer: {e}")
        raise


        
def textgenmode(input,model,tokenizer,input_ids,):
    questionpattern=['what','who','how','is','are','?']
    textgenpattern=['tell','write','make','once','upon']
    codegenpattern= [
    """
    # Function to calculate the factorial of a number
    def factorial(n):
        if n == 0:
            return 1
        else:
            return n * factorial(n-1)
    """,
    """
    # Function to find the maximum number in a list
    def find_max(numbers):
        max_num = numbers[0]
        for num in numbers:
            if num > max_num:
                max_num = num
        return max_num
    """,
    """
    # Function to check if a number is prime
    def is_prime(n):
        if n <= 1:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True
    """,
    """
    # Function to sort a list using bubble sort
    def bubble_sort(arr):
        n = len(arr)
        for i in range(n):
            for j in range(0, n-i-1):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr
    """,
    """
    # Function to sum the elements of a list
    def sum_list(lst):
        total = 0
        for elem in lst:
            total += elem
        return total
    """,
    """
    # Function to reverse a string
    def reverse_string(s):
        return s[::-1]
    """,
    """
    # Function to count the occurrences of each character in a string
    def char_count(s):
        counts = {}
        for char in s:
            if char in counts:
                counts[char] += 1
            else:
                counts[char] = 1
        return counts
    """,
    """
    # Function to merge two dictionaries
    def merge_dicts(dict1, dict2):
        result = dict1.copy()
        result.update(dict2)
        return result
    """
]

    for input in questionpattern:
        output = model.generate(input_ids, max_length=512, 
                       temperature=0.7, 
                       top_p=0.9, 
                       top_k=50, 
                       repetition_penalty=1.2, 
                       num_return_sequences=1)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info(f"Generated text for question: {input}")
        logger.info(f"Generated text: {generated_text}")
        # Check if the generated text is a question
        if input in generated_text.lower():
            return generated_text
        elif input not in questionpattern:
            for input in textgenpattern:
                output = model.generate(input_ids,max_length=1024, 
                             temperature=0.9, 
                             top_p=0.95, 
                             top_k=40, 
                             repetition_penalty=1.1, 
                             num_return_sequences=1)
                generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
                logger.info(f"Generated text for question: {input}")
                logger.info(f"Generated text: {generated_text}")
        elif input not in questionpattern:
            for input in codegenpattern:
             patterns = "\n".join(codegenpattern)
             combined_prompt = patterns + "\n\n" + input_ids
             output = model.generate(combined_prompt, max_length=512, 
                             temperature=0.7, 
                             top_p=0.9, 
                             top_k=50, 
                             repetition_penalty=1.2, 
                             num_return_sequences=1)
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            logger.info(f"Generated text for question: {input}")
            logger.info(f"Generated text: {generated_text}")
                # Check if the generated text is valid answer
                
        if generated_text.strip():
            return generated_text
                
                
            

def local_text_generation( input, model, tokenizer, max_length=300):
    try:
        feedback_data = at.load_feedback_data('feedback.csv')
        prompts = feedback_data['Prompt'].tolist()

        # Encode the input prompt
        input_ids = tokenizer.encode(input, return_tensors="pt")
        
        return textgenmode(input,model,tokenizer,input_ids)
        
     
    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        raise
      
    
   
