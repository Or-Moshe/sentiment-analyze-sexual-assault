
import pandas as pd
from googletrans import Translator
from transformers import pipeline, BertForSequenceClassification, BertTokenizer, AdamW
import torch
from sklearn.model_selection import train_test_split

from helpers.helper import translate_to_english, remove_special_characters, remove_empty_strings, translate_text, setup_model, translate_he_to_en
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('../data/with classification.csv')
new_df = pd.DataFrame()

list(df.columns)

print('rows before preproccessing', df.size)

#preprocessing

filtered_df = df[df['wordsConsumer'] >= 20]

print('after filtering', filtered_df .size)
new_df['classification'] = filtered_df['classification']


def translate_texts_with_googletrans(texts):
    translator = Translator()
    translated_texts = []

    for text in texts:
        # Handle the translation
        try:
            print('*********text', text)
            translated = translator.translate(text, src='he', dest='en')
            translated_texts.append(translated.text)
        except Exception as e:
            # Print the exception and append None or an error message

            print(f"Error translating text: {e}")
            translated_texts.append(None)

    return translated_texts

'''
# Example list of texts
texts = [
    "היא שכחה לכתוב לו.",
    "זה היה יום נפלא.",
    "איך אתה מרגיש היום?"
]



# Translate the texts
translations = translate_texts_with_googletrans(filtered_df['transcriptConsumer'].tolist())

# Output the translations
for original, translation in zip(texts, translations):
    print(f"Original: {original} | Translated: {translation}")
'''
'''
pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-he-en")

# List of Hebrew texts to translate
texts = [
    "היא שכחה לכתוב לו.",
    "זה היה יום נפלא.",
    "איך אתה מרגיש היום?"
]

# Translate all texts in a batch
translated_texts = pipe(texts)

# Print each translated text
for translation in translated_texts:
    print(translation['translation_text'])
    '''
'''
tokenizer, model = setup_model()
hebrew_text = 'שלום, איך אתה?'
translated_text = translate_he_to_en(hebrew_text, tokenizer, model)
print(translated_text)
'''

'''
# Initialize the translator
translator = Translator()
import time
def translate_text2(text):
    if text is None or text.strip() == "":
        return None
    attempt = 0
    max_attempts = 5
    while attempt < max_attempts:
        try:
            translation = translator.translate(text, src='he', dest='en')
            return translation.text
        except Exception as e:
            wait = (2 ** attempt) * 0.5  # Exponential backoff factor
            print(f"Retry {attempt + 1}/{max_attempts}, waiting for {wait}s due to error: {e}")
            time.sleep(wait)
            attempt += 1
    print(f"Failed to translate after {max_attempts} attempts: {text}")
    return None

df['transcriptAgent_en'] = None
df['transcriptConsumer_en'] = None

# Iterate over each row in the DataFrame
new_df['transcriptAgent'] = filtered_df['transcriptAgent'].apply(remove_empty_strings)
new_df['transcriptConsumer'] = filtered_df['transcriptConsumer'].apply(remove_empty_strings)

for index, row in filtered_df.iterrows():
    if pd.notna(row['transcriptAgent']):
        try:
            translated_text = translate_text2(row['transcriptAgent'])
            new_df.at[index, 'transcriptAgent_en'] = translated_text
        except Exception as e:
            print("error in index: ", index)
            print("error in transcriptAgent: ", row['transcriptAgent'])
            print(f"Error during translation: {e}")
    else:
        new_df.at[index, 'transcriptAgent_en'] = None
    if pd.notna(row['transcriptConsumer']):
        try:
            translated_text = translate_text2(row['transcriptConsumer'])
            new_df.at[index, 'transcriptConsumer_en'] = translated_text
        except Exception as e:
            print("error in index: ", index)
            print("error in transcriptAgent: ", row['transcriptAgent'])
            print(f"Error during translation: {e}")
    else:
        new_df.at[index, 'transcriptConsumer_en'] = None
'''
#new_df['transcriptConsumerEnglish'] = filtered_df['transcriptAgent'].apply(translate_text2)
#english_translation = translate_text2(filtered_df['transcriptAgent'][10])
#print(english_translation)
# Concatenate customer and agent sentences
#df['transcriptConsumerEnglish'] = filtered_df['transcriptAgent'].apply(remove_special_characters)


#new_df['transcriptAgent_en'] = filtered_df['transcriptAgent'].apply(lambda text: translate_text(text))





#new_df['transcriptConsumer_en'] = df['transcriptConsumer'].apply(translate_text)
#new_df['transcriptAgent_en'] = df['transcriptAgent'].apply(translate_text)
#new_df['transcriptAll_en'] = df['transcriptAll'].apply(translate_text)

#new_df['transcriptAgentEnglish'] = filtered_df['transcriptAgent'].apply(translate_to_english)
#new_df['transcriptConsumer'] = filtered_df['transcriptConsumer'].apply(remove_empty_strings)
#new_df['transcriptConsumerEnglish'] = filtered_df['transcriptConsumer'][11]#.apply(translate_text)#filtered_df.apply(lambda row: translate_text(row,'transcriptConsumer'), axis=1)

new_df.to_csv('../data/huggingFace.csv', index=False)

print('finished')
'''
translated_transcriptAll = new_df['translated_transcriptAll'].tolist()  # Combined customer and agent sentences
labels = new_df['classification'].tolist()


# Load pre-trained model and tokenizer
model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Adjust num_labels as per your task

# Load text classification pipeline
classifier = pipeline("text-classification", model="bert-base-multilingual-cased", tokenizer="classification")
# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(translated_combined_text, labels, test_size=0.2, random_state=42)

# Specify training parameters
optimizer = AdamW(model.parameters(), lr=1e-5)
num_epochs = 3

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()
for epoch in range(num_epochs):
    for text, label in zip(train_texts, train_labels):
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
        labels = torch.tensor(label).unsqueeze(0).to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Put model in evaluation mode
model.eval()

# Lists to store predictions and true labels
predictions = []
true_labels = []

# Evaluation loop
with torch.no_grad():
    for text, label in zip(test_texts, test_labels):
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
        labels = torch.tensor(label).unsqueeze(0).to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        _, predicted = torch.max(logits, dim=1)
        predictions.append(predicted.item())
        true_labels.append(label)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)
print("Accuracy:", accuracy)

# Generate classification report
report = classification_report(true_labels, predictions)
print("Classification Report:")
print(report)
'''