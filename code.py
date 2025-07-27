import pandas as pd
import ta
from ta.volatility import BollingerBands
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Step 1: Extract data from CSV
df = pd.read_csv('original_stock_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Step 2: Transform data by adding technical indicators

df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)

# MACD
macd = ta.trend.MACD(close=df['Close'])
df['MACD'] = macd.macd()
df['MACD_signal'] = macd.macd_signal()
df['MACD_hist'] = macd.macd_diff()

# Bollinger Bands
bbands = BollingerBands(close=df['Close'], window=20)
df['BB_upper'] = bbands.bollinger_hband()
df['BB_middle'] = bbands.bollinger_mavg()
df['BB_lower'] = bbands.bollinger_lband()


# Define features (technical indicators)
features = ['SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist', 'BB_upper', 'BB_middle', 'BB_lower']

# Step 3: Create target variable (direction of next day's movement)
df['direction'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 
                           np.where(df['Close'].shift(-1) < df['Close'], -1, 0))

# Step 4: Drop rows with NaN values in features or target
df_trainable = df.dropna(subset=features + ['direction'])

# Step 5: Split data into training and testing sets
split_idx = int(0.8 * len(df_trainable))
X_train = df_trainable.iloc[:split_idx][features]
y_train = df_trainable.iloc[:split_idx]['direction']
X_test = df_trainable.iloc[split_idx:][features]
y_test = df_trainable.iloc[split_idx:]['direction']

# Step 6: Encode target variable for classification
le = LabelEncoder()
y_train_le = le.fit_transform(y_train)
y_test_le = le.transform(y_test)

# Step 7: Train a Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train_le)

# Step 8: Evaluate the model on the test set
y_pred_le = model.predict(X_test)
print("Classification Report:")
target_names = [str(cls) for cls in le.classes_]
print(classification_report(y_test_le, y_pred_le, target_names=target_names, zero_division=0))


# Step 9: Predict for the next day (using the last row of the original data)
X_pred = df.iloc[-1][features].values.reshape(1, -1)
y_pred_next_le = model.predict(X_pred)
y_pred_next = le.inverse_transform(y_pred_next_le)[0]

# Step 10: Interpret the prediction
if y_pred_next == 1:
    print("Next day prediction: Positive")
elif y_pred_next == -1:
    print("Next day prediction: Negative")
else:
    print("Next day prediction: Neutral")