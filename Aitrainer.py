import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from time import sleep
modelver=3
class TextDataset(Dataset):
    def __init__(self, encodings):
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        self.labels = encodings['input_ids'].clone()  # labels should be the same as input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

def load_feedback_data(csv_file):
    df = pd.read_csv(csv_file)
    return df

def tokenize_data(tokenizer, prompts, answers):
    inputs = [f"question: {prompt} answer: {answer}" for prompt, answer in zip(prompts, answers)]
    encodings = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)
    return encodings

def main(modelver):
    try:
     while(1):
        model_name = f"output/finetunedv{modelver-1}"
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if(modelver>=60):
            break

        # Ensure tokenizer has a padding token
        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

        # Load feedback data
        feedback_data = load_feedback_data('feedback.csv')
        prompts = feedback_data['Prompt'].tolist()
        corrected_answers = feedback_data['Corrected Answer'].tolist()

        # Tokenize data
        encodings = tokenize_data(tokenizer, prompts, corrected_answers)

        # Split data into train and validation sets
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        train_inputs, val_inputs, train_mask, val_mask = train_test_split(input_ids, attention_mask, test_size=0.1, random_state=42)

        train_dataset = TextDataset({'input_ids': train_inputs, 'attention_mask': train_mask})
        val_dataset = TextDataset({'input_ids': val_inputs, 'attention_mask': val_mask})

        # Define training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            logging_dir='/home/yassin/Documents/Aiwithypython/logs',
            output_dir=f'/home/yassin/Documents/Aiwithypython/output/finetunedv{modelver}',
            logging_steps=100,
            save_steps=500,
            eval_steps=500,
            save_total_limit=2,
            overwrite_output_dir=True,
            warmup_steps=500,
            learning_rate=2e-4,
            weight_decay=0.01,
            eval_strategy='steps',
        )

        # Fine-tune the model
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer
        )

        trainer.train()                                   
        trainer.save_model(output_dir=training_args.output_dir) #automate the training process
        sleep(60)
        modelver+=1
    except Exception as e:
        print(f"Error: {e}")
        
        
if __name__ == "__main__":
    main(modelver)
