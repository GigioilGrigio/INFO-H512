import pandas as pd
import numpy as np
from sklearn.feature_selection import f_regression

df = pd.read_pickle("../../data/interim/engineeredfeatures_df.pkl")


def select_top_features(df, target_cols=["y1_target", "y2_target"], k=10):
    """
    Select top k features based on correlation with multiple targets.

    Parameters:
        df (pd.DataFrame): dataframe with features + targets
        target_cols (list): target column names
        k (int): number of features to select

    Returns:
        df_selected (pd.DataFrame): dataframe with only top features + targets
        selected_features (list): list of selected feature names
    """

    # Separate features and targets
    X = df.drop(columns=target_cols)
    y1 = df[target_cols[0]]
    y2 = df[target_cols[1]]

    # Compute F-scores for each target
    scores_y1, _ = f_regression(X, y1)
    scores_y2, _ = f_regression(X, y2)

    # Combine scores (you can also use np.max instead of mean)
    combined_scores = (scores_y1 + scores_y2) / 2

    # Rank features
    feature_scores = pd.Series(combined_scores, index=X.columns)
    top_features = feature_scores.nlargest(k).index.tolist()

    # Return reduced dataframe
    df_selected = df[top_features + target_cols]

    return df_selected, top_features


df_selected, top_features = select_top_features(df, k=10)

print("Top features:")
print(top_features)

df_selected.to_pickle("../../data/processed/best_features_df.pkl")
