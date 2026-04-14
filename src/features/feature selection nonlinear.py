from sklearn.ensemble import RandomForestRegressor
import pandas as pd

df = pd.read_pickle("../../data/interim/engineeredfeatures_df.pkl")


def select_top_features_tree(df, target_cols=["y1_target", "y2_target"], k=10):
    """
    Select top k features using Random Forest importance (nonlinear).
    """

    X = df.drop(columns=target_cols)
    y = df[target_cols]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    # Average importance across outputs
    importances = model.feature_importances_

    feature_scores = pd.Series(importances, index=X.columns)
    top_features = feature_scores.nlargest(k).index.tolist()

    df_selected = df[top_features + target_cols]

    return df_selected, top_features


df_selected, top_features = select_top_features_tree(df, k=20)

print("Top features:")
print(top_features)

df_selected.to_pickle("../../data/processed/best_features_df.pkl")
