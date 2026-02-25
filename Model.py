import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

# 1. CRITICAL: Sort by time to prevent future data from leaking into the past
df = pd.read_parquet("data/processed/training_set.parquet")
df = df.sort_values('Datetime').reset_index(drop=True)

features = ['Z_Score', 'ADX', 'BBW', 'VOL_24H']

# Get a sorted list of all unique timestamps
unique_times = df['Datetime'].sort_values().unique()

# --- Walk-Forward Parameters ---
TRAIN_WINDOW = 90  # Train on 90 periods
TEST_WINDOW = 30   # Test on 30 periods
PURGE_WINDOW = 8   # Skip 8 periods between train and test

all_results = []
models = []

# 2. Walk-Forward Loop
for i in range(0, len(unique_times) - TRAIN_WINDOW - PURGE_WINDOW - TEST_WINDOW + 1, TEST_WINDOW):
    
    # Define time boundaries for this fold
    train_start = unique_times[i]
    train_end = unique_times[i + TRAIN_WINDOW - 1]
    
    test_start = unique_times[i + TRAIN_WINDOW + PURGE_WINDOW]
    test_end = unique_times[i + TRAIN_WINDOW + PURGE_WINDOW + TEST_WINDOW - 1]
    
    # Create boolean masks to slice the dataframe
    train_mask = (df['Datetime'] >= train_start) & (df['Datetime'] <= train_end)
    test_mask = (df['Datetime'] >= test_start) & (df['Datetime'] <= test_end)
    
    X_train, y_train = df.loc[train_mask, features], df.loc[train_mask, 'Meta']
    X_test, y_test = df.loc[test_mask, features], df.loc[test_mask, 'Meta']
    
    if X_train.empty or X_test.empty:
        continue
        
    # 3. Train the model for this specific time regime
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        class_weight='balanced', 
        random_state=42
    )
    model.fit(X_train, y_train)
    models.append(model)
    
    # 4. Predict on the strict out-of-sample test window
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Store results for this fold
    fold_results = pd.DataFrame({
        'Datetime': df.loc[test_mask, 'Datetime'],
        'Ticker': df.loc[test_mask, 'Ticker'],
        'Actual_Target': y_test.values,
        'AI_Probability': y_pred_proba,
        'Fold': i // TEST_WINDOW
    })
    all_results.append(fold_results)

# 5. Aggregate all out-of-sample predictions
final_results_df = pd.concat(all_results, ignore_index=True)

# 6. Evaluate aggregate performance
CONFIDENCE_THRESHOLD = 0.55 
final_results_df['y_pred_strict'] = (final_results_df['AI_Probability'] > CONFIDENCE_THRESHOLD).astype(int)

trades_approved = final_results_df['y_pred_strict'].sum()
base_win_rate = final_results_df['Actual_Target'].mean()
total_samples = len(final_results_df)

if trades_approved > 0:
    overall_win_rate = precision_score(final_results_df['Actual_Target'], final_results_df['y_pred_strict'])
else:
    overall_win_rate = 0.0

# 7. Save and Print Results
final_results_df.to_csv("data/processed/predictions.csv", index=False)

print(f"Base Strategy Win Rate:  {base_win_rate * 100:.2f}%")
print(f"Model Enhanced Win Rate: {overall_win_rate * 100:.2f}%")
print(f"Trades Taken by Model:   {trades_approved} out of {total_samples}")

# 8. Feature Importance Visualization (Averaged across all walk-forward folds)
# Calculate the mean importance for each feature across all stored models
avg_importances = np.mean([m.feature_importances_ for m in models], axis=0)
importance = pd.DataFrame({'Feature': features, 'Importance': avg_importances})
importance = importance.sort_values(by='Importance', ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(importance['Feature'], importance['Importance'], color='coral')
plt.title('Average Feature Importance Across All Walk-Forward Folds')
plt.xlabel('LightGBM Feature Importance')
plt.show()