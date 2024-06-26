# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1V3Wl426ZyTW8M4kDUIqDlfmuwX2do3Q_
"""
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
from textblob import TextBlob #Spelling Correction
from nltk.stem import PorterStemmer #Normalization
import re
import string
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

from googletrans import Translator
from sklearn.model_selection import train_test_split

translator = Translator()
porter = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

df = pd.read_csv('data/with classification.csv').head(10)
print('number of columns', df.size)
print('number of rows', len(df))

# Define a function to translate text
def translate_text(text):
    print('text:', text)
    if pd.notna(text):
        translation = translator.translate(text, src='he', dest='en')
        return translation.text
    else:
        return None

# Apply translation to the 'transcriptConsumer' column and add the results to a new column
df['transcriptConsumerEnglish'] = df['transcriptConsumer'].apply(translate_text)

"""***Tokenization***

1.   CC: Coordinating conjunction
2. CD: Cardinal number
3. DT: Determiner
4. EX: Existential there
5. FW: Foreign word
6. IN: Preposition or subordinating conjunction
7. JJ: Adjective
8. JJR: Adjective, comparative
9. JJS: Adjective, superlative
10. LS: List item marker
11. MD: Modal
12. NN: Noun, singular or mass
13. NNS: Noun, plural
14. NNP: Proper noun, singular
15. NNPS: Proper noun, plural
16. PDT: Predeterminer
17. POS: Possessive ending
18. PRP: Personal pronoun
19. PRP$: Possessive pronoun
20. RB: Adverb
21. RBR: Adverb, comparative
22. RBS: Adverb, superlative
23. RP: Particle
24. SYM: Symbol
25. TO: to
26. UH: Interjection
27. VB: Verb, base form
28. VBD: Verb, past tense
29. VBG: Verb, gerund or present participle
30. VBN: Verb, past participle
31. VBP: Verb, non-3rd person singular present
32. VBZ: Verb, 3rd person singular present
33. WDT: Wh-determiner
34. WP: Wh-pronoun
35. WP$: Possessive wh-pronoun
36. WRB: Wh-adverb
"""


def remove_special_characters(sentence):
    return re.sub(r'[^a-zA-Z0-9\s]', '', sentence)


def correct_spelling(text):
    return TextBlob(text).correct()


# Define a function to tokenize sentences, handling potential empty tokens
def tokenize_sentence(sentence):
    if pd.notna(sentence):
        # Tokenize the sentence
        tokens = word_tokenize(sentence)

        # Remove empty tokens
        tokens = [token.lower() for token in tokens if token.strip()]

        print('tokens', tokens)
        return tokens
    else:
        return []


def tagged_tokens(tokens):
    return nltk.pos_tag(tokens)


def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    stemmed_tokens = [porter.stem(token) for token in tokens]
    print('stemmed_tokens ', stemmed_tokens)
    filtered_sentences = [word for word in stemmed_tokens if word.lower() not in stop_words]
    print('filtered_sentences ', filtered_sentences)
    return filtered_sentences


def remove_specialCharacters(sentence):
    return re.sub(r'[^a-zA-Z0-9\s]', '', sentence)

###################################################Text Preprocessing:
df['transcriptConsumerEnglish'] = df['transcriptConsumerEnglish'].apply(remove_special_characters, correct_spelling)
# Apply word tokenization to the 'transcriptConsumerEnglish' column
df['tokens_english'] = df['transcriptConsumerEnglish'].apply(tokenize_sentence)

print('tokens_english: ', df['tokens_english'])
print('*********************************')

# Remove stopwords
df['filtered_tokens_english'] = df['tokens_english'].apply(remove_stopwords)

df['tagged_tokens_english'] = df['filtered_tokens_english'].apply(tagged_tokens)
print('filtered_tokens_english: ', df['filtered_tokens_english'])
print('tagged_tokens_english: ', df['tagged_tokens_english'])


# Count the appearance of each word
word_counts = Counter(df['filtered_tokens_english'][4])

# Print the word counts
for word, count in word_counts.items():
    print(word, ':', count)


# Plot the histogram of token frequencies
plt.figure(figsize=(12, 6))
plt.bar(word_counts.keys(), word_counts.values())
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Token Frequency Histogram')
plt.xticks(rotation=45)
plt.show()

df.to_csv('data/with classification.csv', index=False)