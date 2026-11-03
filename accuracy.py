import pandas as pd
import numpy as np

df = pd.read_csv('outputs/predictions.csv')
test = df[df['Split'] == 'test']

for crop in ['Wheat', 'Tomato', 'Onion']:
    c = test[test['Commodity'] == crop]
    mape = np.mean(np.abs((c['Actual'] - c['Hybrid_Pred']) / c['Actual'])) * 100
    accuracy = 100 - mape
    print(f"{crop}: {accuracy:.2f}% accurate")