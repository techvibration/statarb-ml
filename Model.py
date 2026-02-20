import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, classification_report
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_parquet("data/processed/training_set.parquet")

# 2. Define Features and Target
features = ['Z_Score', 'RSI', 'ADX', 'BBW', 'VOL_24H']
X = df[features]
y = df['Meta']

# 3. Time-Series Split
# We train on the first 80% of time, and test on the unseen future 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"Training on {len(X_train)} trades, Testing on {len(X_test)} unseen trades...")

# 4. Train the Model
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    class_weight='balanced', # Helps because you have more 0s than 1s
    random_state=42
)
model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)[:, 1]

# Let's act like a strict bouncer. We only take trades the AI is 55% sure about.
CONFIDENCE_THRESHOLD = 0.55 
y_pred_strict = (y_pred_proba > CONFIDENCE_THRESHOLD).astype(int)

# Calculate metrics based on the strict threshold
base_win_rate = y_test.mean()
trades_approved = sum(y_pred_strict)

# Prevent division by zero if the AI rejects everything
if trades_approved > 0:
    # Precision: Out of the trades we took, how many actually won?
    from sklearn.metrics import precision_score
    ai_win_rate = precision_score(y_test, y_pred_strict)
else:
    ai_win_rate = 0.0

results_df = pd.DataFrame({
    'Datetime': df.loc[X_test.index, 'Datetime'],
    'Ticker': df.loc[X_test.index, 'Ticker'],
    'Actual_Target': y_test.values,
    'AI_Probability': y_pred_proba
})


results_df = results_df.sort_values('Datetime').reset_index(drop=True)


results_df.to_csv("data/processed/predictions.csv", index=False)

print("-" * 40)
print("--- ALPHAFINDER V2 PERFORMANCE (STRICT) ---")
print(f"Base Strategy Win Rate:  {base_win_rate * 100:.2f}%")
print(f"AI-Filtered Win Rate:    {ai_win_rate * 100:.2f}%")
print(f"Trades Taken by AI:      {trades_approved} out of {len(y_test)}")
print("-" * 40)
# 6. Feature Importance Visualization
importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
importance = importance.sort_values(by='Importance', ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(importance['Feature'], importance['Importance'], color='coral')
plt.title('Which Indicators Drive the AI Decisions?')
plt.xlabel('LightGBM Feature Importance')
plt.show()

