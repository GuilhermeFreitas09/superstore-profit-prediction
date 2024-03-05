import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp

def get_bad_rate(df, feat, target, num_buckets=10):
    data = df.copy()
    
    data['feature_bucket'] = pd.qcut(data[feat], num_buckets, precision=2, duplicates='drop')
    
    bad_rate = data.groupby('feature_bucket')[target].mean().reset_index()
    
    plt.figure(figsize=(12, 6)) 
    sns.barplot(data=bad_rate, x="feature_bucket", y=target, estimator=np.mean, ci=None)
    plt.xticks(rotation=30)
    plt.title(f'Bad rate for {feat}')
    plt.show()


def calculate_ks_corr(df, features, target, order_by = 'correlation'):
    result = pd.DataFrame(columns=['Feature', 'ks', 'correlation'])

    for feature in features:
        data = df.copy()
        data = data.dropna(subset=[feature])
        ks_statistic, _ = ks_2samp(data.loc[data[target] == 0, feature], data.loc[df[target] == 1, feature])
        correlation = np.corrcoef(data[feature], data[target])[0, 1]

        result = pd.concat([result, pd.DataFrame([[feature, ks_statistic, correlation]], columns=['Feature', 'ks', 'correlation'])], ignore_index=True)

    if order_by:
        result = result.sort_values(by=order_by, ascending=False)

    return result
