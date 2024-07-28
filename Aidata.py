import os
from groq import Groq

# Initialize the Groq client
client = Groq(
    api_key=os.environ.get("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"),
)

# Initialize conversational memory
conversation_history = []

def get_token_length(text):
    # This is a simple approximation of token length
    # Modify this function based on the actual tokenization method if needed
    return len(text.split())

def truncate_conversation_history():
    # Ensure the total token length is within the limit
    total_length = sum(get_token_length(message["content"]) for message in conversation_history)
    
    while total_length > 1024:
        removed_message = conversation_history.pop(0)  # Remove the oldest message
        total_length -= get_token_length(removed_message["content"])

def get_Groq_response(user_input):
    try:
        # Add the user message to the conversation history
        conversation_history.append({"role": "user", "content": user_input})
        
        # Truncate conversation history if necessary
        truncate_conversation_history()

        # Get a response from the model
        chat_completion = client.chat.completions.create(
            messages=conversation_history,
            model="llama3-8b-8192",
        )

        # Extract the model's response
        response = chat_completion.choices[0].message.content

        # Add the model's response to the conversation history
        conversation_history.append({"role": "assistant", "content": response})
        
        print("Groq Response:", response)  # Debugging statement
        
        return response

    except Exception as e:
        print(f"Error getting Groq response: {e}")
        return "I'm sorry, there was an error processing your request."