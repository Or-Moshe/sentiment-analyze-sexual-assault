import pandas as pd
import re
#from helpers.translate import translate

df = pd.read_csv('../data/withClassification.csv')

columns_to_copy = ['wordsConsumer', 'wordsAgent', 'transcriptAgent', 'transcriptConsumer', 'transcriptAll']
filtered_df = df[(df['wordsConsumer'] >= 100) & (df['wordsAgent'] >= 100)][columns_to_copy].copy()
filtered_df.to_csv('../data/filtered_df.csv', index=False)

def clean_text(text):
    if pd.isna(text):
        return text

    # Remove single quotes
    text = text.replace("'", "")

    # Remove all types of commas (including special Unicode commas)
    text = re.sub(r'[,\u201A\u201E\u0326\u0315]', '', text)

    # Regex to match all URLs, including 'kolmila.org.il'
    url_pattern = re.compile(r'''
                             https?://\S+|  # matches http or https URLs
                             www\.\S+|      # matches URLs starting with www
                             kolmila\.org\.il\b  # matches 'kolmila.org.il'
                             ''', re.VERBOSE | re.IGNORECASE)
    text = url_pattern.sub(r'', text)

    # Unicode ranges for emojis and some additional related characters
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # Trim whitespace
    text = text.strip()

    return text
'''
columns_to_translate = ['transcriptAgent', 'transcriptConsumer', 'transcriptAll']
for column in columns_to_translate:
    filtered_df[column] = filtered_df[column].apply(clean_text)

translated_df = translate(filtered_df, columns_to_translate)
translated_df.to_csv('../data/translated.csv', index=False)
'''


import requests
import json

# Define the URL of the local server where Ollama is running
url = 'http://localhost:11434/api/chat'

# Prepare the data payload as a dictionary
payload = {
  "model": "mistral",
  "messages": [
    {
      "role": "user",
      "content": "why does it rain?"
    }
  ],
  "temperature": 0.7,
  "top_p": 1,
  "max_tokens": 512,
  "stream": False,
  "safe_prompt": False,
  "random_seed": 1337
}

# Convert the dictionary to JSON format
headers = {'Content-Type': 'application/json'}
data = json.dumps(payload)

# Send a POST request
response = requests.post(url, data=data, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    print("Response from model:", response.json())
else:
    print("Failed to get a response from the model, status code:", response.status_code)
