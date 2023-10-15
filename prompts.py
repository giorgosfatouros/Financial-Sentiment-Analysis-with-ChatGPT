import openai
import json
import pandas as pd

from helper_functions import sentiment_to_numeric, get_sentiment

# Initialize OpenAI API
openai.api_key = '<ADD YOUR API KEY>'

# Load the prompts from the JSON file
with open('prompts.json', 'r') as file:
    prompts = json.load(file)

# Load annotated dataset
df = pd.read_csv('sentiment_annotated_with_texts.csv', parse_dates=True)
# print(df.head())

df['true_sent_numeric'] = df['true_sentiment'].apply(sentiment_to_numeric)


if __name__ == "__main__":
    # test with one row
    row = 1000
    print(f"Ticker: {df['ticker'][row]}")
    print(f"Headline: {df['title'][row]}")
    print(f"Article: {df['text'][row]}")
    print(f"True: {df['true_sentiment'][row]}")
    print(f"FinBERT: {df['finbert_sentiment'][row]}")
    print(f"FinBERT-N: {df['finbert_sent_score'][row]}")

    for prompt in prompts.keys():
        if 'A' not in prompt:
            result = get_sentiment(df.ticker[row], df.title[row], prompt)
        else:
            result = get_sentiment(df.ticker[row], df.text[row], prompt)
        print(f'{prompt}: {result}')


