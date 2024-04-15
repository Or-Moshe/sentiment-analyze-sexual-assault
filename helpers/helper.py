# translator.py
import time
from googletrans import Translator
import re
import pandas as pd

from transformers import MarianMTModel, MarianTokenizer

def setup_model():
    model_name = 'Helsinki-NLP/opus-mt-tc-big-he-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_he_to_en(text, tokenizer, model):
    # Tokenize the text
    encoded_text = tokenizer(text, return_tensors="pt", truncation=True)
    # Generate translation using the model
    translated_tokens = model.generate(**encoded_text)
    # Decode the translated tokens to text
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

def remove_empty_strings(text):
    if isinstance(text, str):
        # Split the text into lines and remove empty strings
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        # Join the non-empty lines back into a single string
        cleaned_text = '\n'.join(lines)
        return cleaned_text
    else:
        return text

def translate_text(text):
    if pd.notna(text):
        translator = Translator()
        translation = translator.translate(text, src='he', dest='en')
        return translation.text
    else:
        return None

def translate_to_english(row, col_name):
    text = row[col_name]
    try:
        # Split text into chunks of max 3000 characters and remove empty strings
        chunks = [chunk for chunk in text.split('\n') if chunk.strip()]
        translated_chunks = []
        for chunk in chunks:
            chunk = chunk.strip()  # Remove leading and trailing whitespaces
            chunk_chunks = [chunk[i:i+3000] for i in range(0, len(chunk), 2000)]  # Split each chunk into smaller parts
            for part in chunk_chunks:
                translator = Translator()
                translated_part = translator.translate(part, src='he', dest='en').text
                time.sleep(2)
                translated_chunks.append(translated_part)
        translated_text = ' '.join(translated_chunks)  # Concatenate translated chunks
        return translated_text
    except Exception as e:
        print(row.name)
        print(text)
        print("Translation error:", e)
        return None
def remove_special_characters(sentence):
    return re.sub(r'[^a-zA-Z0-9\s]', '', sentence)
