import pandas as pd
import string
from googletrans import Translator

translator = Translator()

def translate(filtered_df, columns_to_translate):
    for column in columns_to_translate:
        filtered_df[f'{column}_en'] = filtered_df.apply(lambda row: translate_text_with_googletrans(row, column),axis=1)
    # Save the DataFrame back to a new CSV file with translations
    return filtered_df


def translate_text_with_googletrans(row, column_name):
    text = row[column_name]
    if pd.isna(text):
        return None  # Return None if text is NaN
    try:
        # Perform the translation
        result = translator.translate(text, src='he', dest='en')
        return result.text
    except Exception as e:
        print(f"Error translating text: in '{column_name}'{row.name}' '{text}'. Error: {e}")
        return "Translation failed"


