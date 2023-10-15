import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
from helper_functions import cols_sent, sentiment_mae, highlight_max, highlight_min, cols_sent_day

warnings.filterwarnings("ignore")

df = pd.read_csv('sentiment_predictions_single_article.csv', parse_dates=True)
df2 = pd.read_csv('sentiment_predictions_allday_articles.csv', parse_dates=True)

s = df[cols_sent]
print("#" * 50)
print('Classification results')
print("#" * 50)

metrics = {
    'Model': ['FinBERT', 'FinBERT-A', 'GPT-P1', 'GPT-P2', 'GPT-P3', 'GPT-P4', 'GPT-P4A'],
    'Accuracy': [
        accuracy_score(s['true_sentiment'], s['finbert_sentiment']),
        accuracy_score(s['true_sentiment'], s['finbert_sentiment_a']),
        accuracy_score(s['true_sentiment'], s['gpt_sentiment_p1']),
        accuracy_score(s['true_sentiment'], s['gpt_sentiment_p2']),
        accuracy_score(s['true_sentiment'], s['gpt_sentiment_p3']),
        accuracy_score(s['true_sentiment'], s['gpt_sentiment_p4']),
        accuracy_score(s['true_sentiment'], s['gpt_sentiment_p7']),
    ],
    'Precision': [
        precision_score(s['true_sentiment'], s['finbert_sentiment'], average='weighted'),
        precision_score(s['true_sentiment'], s['finbert_sentiment_a'], average='weighted'),
        precision_score(s['true_sentiment'], s['gpt_sentiment_p1'], average='weighted'),
        precision_score(s['true_sentiment'], s['gpt_sentiment_p2'], average='weighted'),
        precision_score(s['true_sentiment'], s['gpt_sentiment_p3'], average='weighted'),
        precision_score(s['true_sentiment'], s['gpt_sentiment_p4'], average='weighted'),
        precision_score(s['true_sentiment'], s['gpt_sentiment_p7'], average='weighted'),
    ],
    'Recall': [
        recall_score(s['true_sentiment'], s['finbert_sentiment'], average='weighted'),
        recall_score(s['true_sentiment'], s['finbert_sentiment_a'], average='weighted'),
        recall_score(s['true_sentiment'], s['gpt_sentiment_p1'], average='weighted'),
        recall_score(s['true_sentiment'], s['gpt_sentiment_p2'], average='weighted'),
        recall_score(s['true_sentiment'], s['gpt_sentiment_p3'], average='weighted'),
        recall_score(s['true_sentiment'], s['gpt_sentiment_p4'], average='weighted'),
        recall_score(s['true_sentiment'], s['gpt_sentiment_p7'], average='weighted'),
    ],
    'F1-Score': [
        f1_score(s['true_sentiment'], s['finbert_sentiment'], average='weighted'),
        f1_score(s['true_sentiment'], s['finbert_sentiment_a'], average='weighted'),
        f1_score(s['true_sentiment'], s['gpt_sentiment_p1'], average='weighted'),
        f1_score(s['true_sentiment'], s['gpt_sentiment_p2'], average='weighted'),
        f1_score(s['true_sentiment'], s['gpt_sentiment_p3'], average='weighted'),
        f1_score(s['true_sentiment'], s['gpt_sentiment_p4'], average='weighted'),
        f1_score(s['true_sentiment'], s['gpt_sentiment_p7'], average='weighted'),
    ],
    'S-MAE': [
        sentiment_mae(s['true_sentiment'], s['finbert_sentiment']),
        sentiment_mae(s['true_sentiment'], s['finbert_sentiment_a']),
        sentiment_mae(s['true_sentiment'], s['gpt_sentiment_p1']),
        sentiment_mae(s['true_sentiment'], s['gpt_sentiment_p2']),
        sentiment_mae(s['true_sentiment'], s['gpt_sentiment_p3']),
        sentiment_mae(s['true_sentiment'], s['gpt_sentiment_p4']),
        sentiment_mae(s['true_sentiment'], s['gpt_sentiment_p7']),
    ]
}

print('Performance Results in Sentiment Classification')
print("-" * 50)
print(pd.DataFrame(metrics).round(3))
print("-" * 50)
# List of models
models = ['finbert_sentiment', 'finbert_sentiment_a', 'gpt_sentiment_p1', 'gpt_sentiment_p2', 'gpt_sentiment_p3',
          'gpt_sentiment_p4', 'gpt_sentiment_p7']

# Mapping of model names
model_name_mapping = {
    'finbert_sentiment': 'FinBERT',
    'finbert_sentiment_a': 'FinBERT-A',
    'gpt_sentiment_p1': 'GPT-P1',
    'gpt_sentiment_p2': 'GPT-P2',
    'gpt_sentiment_p3': 'GPT-P3',
    'gpt_sentiment_p4': 'GPT-P4',
    'gpt_sentiment_p7': 'GPT-P4A'
}

# Calculate and print classification report for each model
for model in models:
    print(f"Classification report for {model_name_mapping[model]}:")
    report = classification_report(s['true_sentiment'], s[model], output_dict=True)
    print('Positive:', {key: round(value, 3) for key, value in report['1'].items()})
    print('Negative:', {key: round(value, 3) for key, value in report['-1'].items()})
    print('Neutral:', {key: round(value, 3) for key, value in report['0'].items()})
    print("-" * 50)
print("#" * 50)
print('Classification results per ticker')
print("#" * 50)
new_names = ['FinBERT', 'FinBERT-A', 'GPT-P1', 'GPT-P2', 'GPT-P3', 'GPT-P4', 'GPT-P4A',
             'FinBERT-N', 'FinBERT-AN', 'GPT-P1N', 'GPT-P2N', 'GPT-P3N', 'GPT-P4N', 'GPT-P4AN']
old_names = ['finbert_sentiment', 'finbert_sentiment_a', 'gpt_sentiment_p1', 'gpt_sentiment_p2', 'gpt_sentiment_p3',
             'gpt_sentiment_p4', 'gpt_sentiment_p7',
             'finbert_sentiment_n', 'finbert_sentiment_a_n', 'gpt_sentiment_p1n', 'gpt_sentiment_p2n',
             'gpt_sentiment_p3n', 'gpt_sentiment_p4n', 'gpt_sentiment_p7n']
s.rename(columns=dict(zip(old_names, new_names)), inplace=True)

performance_data = []
# Loop through each ticker and calculate the metrics for each model
for ticker in s['ticker'].unique():
    ticker_data = s[s['ticker'] == ticker]
    true_sentiments = ticker_data['true_sentiment']
    models = ['FinBERT', 'FinBERT-A', 'GPT-P1', 'GPT-P2', 'GPT-P3', 'GPT-P4', 'GPT-P4A']

    for model in models:
        model_sentiments = ticker_data[model]
        accuracy = accuracy_score(true_sentiments, model_sentiments)
        precision = precision_score(true_sentiments, model_sentiments, average='weighted', zero_division=0)
        recall = recall_score(true_sentiments, model_sentiments, average='weighted', zero_division=0)
        f1 = f1_score(true_sentiments, model_sentiments, average='weighted', zero_division=0)
        mae = sentiment_mae(true_sentiments, model_sentiments)
        performance_data.append({
            'ticker': ticker,
            'model': model,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'S-MAE': mae
        })

# Convert the performance data list to a pandas DataFrame
performance_df = pd.DataFrame(performance_data).round(3)
performance_df.set_index(['ticker', 'model'], inplace=True)

# Reset index to make all current indices as columns
long_df = performance_df.reset_index()

# Melt DataFrame to long format
long_df = pd.melt(long_df, id_vars=['ticker', 'model'], var_name='metric', value_name='score')

best_models_df = performance_df.groupby('ticker').idxmax()
best_models_df['S-MAE'] = performance_df.groupby('ticker').idxmin()['S-MAE']
print('Best Performing Model in Sentiment Classification per forex pair')
print("-" * 50)
print(best_models_df)
print("#" * 50)

print('Daily Sentiment Analysis')
print("#" * 50)
s1 = df[cols_sent]
s2 = df2[cols_sent_day]

s1['published_at'] = pd.to_datetime(s1['published_at']).dt.date
s2['published_at'] = pd.to_datetime(s2['published_at']).dt.date

s1_grouped = s1.groupby(['published_at', 'ticker']).sum().reset_index()
s = pd.merge(s1, s2, how='outer', on=['published_at', 'ticker'])
# calculate volume of news per day
s_grouped = s.groupby(['published_at', 'ticker']).agg(
    true_sentiment=('true_sentiment', 'sum'),
    finbert_sentiment=('finbert_sentiment', 'sum'),
    finbert_sentiment_a=('finbert_sentiment_a', 'sum'),
    gpt_sentiment_p1=('gpt_sentiment_p1', 'sum'),
    gpt_sentiment_p2=('gpt_sentiment_p2', 'sum'),
    gpt_sentiment_p3=('gpt_sentiment_p3', 'sum'),
    gpt_sentiment_p4=('gpt_sentiment_p4', 'sum'),
    gpt_sentiment_p5=('gpt_sentiment_p5', 'sum'),
    gpt_sentiment_p6=('gpt_sentiment_p6', 'sum'),
    gpt_sentiment_p7=('gpt_sentiment_p7', 'sum'),
    finbert_sentiment_n=('finbert_sentiment_n', 'sum'),
    finbert_sentiment_a_n=('finbert_sentiment_a_n', 'sum'),
    gpt_sentiment_p1n=('gpt_sentiment_p1n', 'sum'),
    gpt_sentiment_p2n=('gpt_sentiment_p2n', 'sum'),
    gpt_sentiment_p3n=('gpt_sentiment_p3n', 'sum'),
    gpt_sentiment_p4n=('gpt_sentiment_p4n', 'sum'),
    gpt_sentiment_p5n=('gpt_sentiment_p5n', 'sum'),
    gpt_sentiment_p6n=('gpt_sentiment_p6n', 'sum'),
    gpt_sentiment_p7n=('gpt_sentiment_p7n', 'sum'),
    news_vol=('gpt_sentiment_p1n', 'count')
).reset_index()
s_grouped['published_at'] = pd.to_datetime(s1_grouped['published_at']).dt.date

# calculate naive sentiment
s_grouped['naive_sentiment'] = 0.0

for t in s1.ticker.unique():
    value_counts = s1[s1.ticker == t]['true_sentiment'].value_counts()
    most_frequent_value = value_counts.idxmax()
    s_grouped['naive_sentiment'].loc[s_grouped.ticker == t] = most_frequent_value

# read market returns
price_data = pd.read_csv('forex_price_data.csv')
price_data['datetime'] = pd.to_datetime(price_data['datetime']).dt.date

# Merge the daily sentiment scores with daily returns
daily_sentiment_and_returns = s_grouped.merge(price_data, left_on=['ticker', 'published_at'],
                                              right_on=['ticker', 'datetime'])
daily_sentiment_and_returns = daily_sentiment_and_returns.set_index('datetime')

cols_to_keep = ['ticker', 'true_sentiment', 'naive_sentiment', 'finbert_sentiment', 'finbert_sentiment_a',
                'gpt_sentiment_p1', 'gpt_sentiment_p2', 'gpt_sentiment_p3',
                'gpt_sentiment_p4', 'gpt_sentiment_p5', 'gpt_sentiment_p6', 'gpt_sentiment_p7',
                'finbert_sentiment_n', 'finbert_sentiment_a_n',
                'gpt_sentiment_p1n', 'gpt_sentiment_p2n', 'gpt_sentiment_p3n',
                'gpt_sentiment_p4n', 'gpt_sentiment_p5n', 'gpt_sentiment_p6n', 'gpt_sentiment_p7n',
                'news_vol', 'daily_returns']

daily_sentiment_and_returns = daily_sentiment_and_returns[cols_to_keep]

new_names = ['True Sent', 'Naive Sent', 'FinBERT', 'FinBERT-A', 'GPT-P1', 'GPT-P2', 'GPT-P3', 'GPT-P4', 'GPT-P5',
             'GPT-P6', 'GPT-P4A',
             'FinBERT-N', 'FinBERT-AN', 'GPT-P1N', 'GPT-P2N', 'GPT-P3N', 'GPT-P4N', 'GPT-P5N', 'GPT-P6N', 'GPT-P4AN',
             'Returns']
old_names = ['true_sentiment', 'naive_sentiment', 'finbert_sentiment', 'finbert_sentiment_a', 'gpt_sentiment_p1',
             'gpt_sentiment_p2', 'gpt_sentiment_p3', 'gpt_sentiment_p4',
             'gpt_sentiment_p5', 'gpt_sentiment_p6', 'gpt_sentiment_p7', 'finbert_sentiment_n', 'finbert_sentiment_a_n',
             'gpt_sentiment_p1n', 'gpt_sentiment_p2n', 'gpt_sentiment_p3n',
             'gpt_sentiment_p4n', 'gpt_sentiment_p5n', 'gpt_sentiment_p6n', 'gpt_sentiment_p7n', 'daily_returns']
daily_sentiment_and_returns.rename(columns=dict(zip(old_names, new_names)), inplace=True)

# select only days with news
daily_sentiment_and_returns = daily_sentiment_and_returns[daily_sentiment_and_returns['news_vol'] > 0]

# Calculate the correlation matrix for sentiment columns and daily returns
sentiment_columns = ['FinBERT', 'FinBERT-A', 'GPT-P1', 'GPT-P2', 'GPT-P3', 'GPT-P4', 'GPT-P5', 'GPT-P6', 'GPT-P4A',
                     'FinBERT-N', 'FinBERT-AN', 'GPT-P1N', 'GPT-P2N', 'GPT-P3N', 'GPT-P4N', 'GPT-P5N', 'GPT-P6N',
                     'GPT-P4AN',
                     'True Sent', 'Naive Sent', 'Returns']
corr_matrix = daily_sentiment_and_returns[sentiment_columns].corr()

# Display the correlation matrix as a heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.xticks(rotation=45)  # add this line to rotate x-axis labels

plt.savefig('fig_res_corr.png', dpi=500)

plt.title("Correlation Matrix: Sentiment vs. Daily Returns")

# Initialize an empty list to store the correlation data
sentiment_columns = ['FinBERT', 'FinBERT-A', 'GPT-P1', 'GPT-P2', 'GPT-P3', 'GPT-P4', 'GPT-P5', 'GPT-P6', 'GPT-P4A',
                     'FinBERT-N', 'FinBERT-AN',
                     'GPT-P1N', 'GPT-P2N', 'GPT-P3N', 'GPT-P4N', 'GPT-P5N', 'GPT-P6N', 'GPT-P4AN',
                     'True Sent', 'Naive Sent']
correlation_data = []

# Loop through each ticker and calculate the correlation between daily sentiment and returns
for ticker in daily_sentiment_and_returns['ticker'].unique():
    ticker_data = daily_sentiment_and_returns[daily_sentiment_and_returns['ticker'] == ticker]

    for col in sentiment_columns:
        corr_matrix = ticker_data[[col, 'Returns']].corr()
        correlation_data.append({'ticker': ticker, 'sentiment_model': col, 'correlation': corr_matrix.iloc[0, 1]})

# Convert the correlation data list to a pandas DataFrame
correlation_df = pd.DataFrame(correlation_data)

# Reshape the correlation DataFrame using pivot()
compact_correlation_df = correlation_df.pivot(index='sentiment_model', columns='ticker', values='correlation')

# Reset the index to make the DataFrame more readable
# compact_correlation_df.reset_index(inplace=True)
print("-" * 50)
print('Correlation of Predicted Sentiment and Forex pair returns')
print("-" * 50)
print(compact_correlation_df)
print("-" * 50)


