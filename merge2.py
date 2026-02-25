import pandas as pd 

raw = pd.read_parquet("data/processed/top100af.parquet")
raw['Datetime'] = pd.to_datetime(raw['Datetime'])

#loading the model's predictions
pred = pd.read_csv("data/processed/predictions.csv")
pred['Datetime'] = pd.to_datetime(pred['Datetime'])
p = pred[['Datetime','Ticker','AI_Probability']]
main = pd.merge(raw,p,on=['Datetime','Ticker'],how ='left')
main['AI_Probability'] = main['AI_Probability'].fillna(0.0)
path = "data/processed/main.parquet"
main.to_parquet(path,index=False)
print(main[main['AI_Probability'] > 0.65][['Datetime', 'Ticker', 'Close', 'VOL_24H', 'AI_Probability']].head())