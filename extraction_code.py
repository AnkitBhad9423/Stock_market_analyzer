import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.model_selection import train_test_split
def label_data(df):
    df = df.copy()
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df

def adding_indicators(df):
    df['daily_return'] = df['Close'].pct_change()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['price_change'] = df['Close'] - df['Open']
    df['high_low_range'] = df['High'] - df['Low']
    return df

def moving_averages(df):
    for window in [5, 10, 20, 50]:
        df[f'sma_{window}'] = df['Close'].rolling(window).mean()
        df[f'ema_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    return df     

def momentum_indicator(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    return df

def volatility_measures(df):
    df['volatility_20'] = df['Close'].rolling(window=20).std()
    df['bollinger_upper'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
    df['bollinger_lower'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
    return df

def volume_based_feature(df):
    df['volume_change'] = df['Volume'].pct_change()
    df['vol_sma_20'] = df['Volume'].rolling(window=20).mean()
    return df

def data_based_features(df):
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
    return df


def using_ml_analysis(df):
    df = df.dropna().reset_index(drop=True)
    features = df.drop(columns=['Date', 'Open', 'High', 'Low', 'Adj Close', 'target'])  # drop target after extracting
    X = features
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)







def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df['Close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    return df


def fetch_index_data(ticker: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
    index = yf.Ticker(ticker)
    hist = index.history(period=period, interval=interval)
    hist.reset_index(inplace=True)
    return hist

# Example: NIFTY 50 via ^NSEI, S&P 500 via ^GSPC
df = fetch_index_data('^NSEI', period='2y')
print(df.tail(30))
print('='*100)
print()

df.to_csv('original_stock_data.csv', index=False)


label_df= label_data(df)
indicator_df=add_indicators(label_df)
print(indicator_df.tail(22))
print("="*200)
print()


indicator_df=adding_indicators(indicator_df)
moving_averages_df=moving_averages(indicator_df)
momentum_indicator_added_df=momentum_indicator(moving_averages_df)
volatility_measures_df= volatility_measures(momentum_indicator_added_df)
volume_based_feature_df= volume_based_feature(volatility_measures_df)
data_based_features_df=data_based_features(volume_based_feature_df)
print(data_based_features_df.tail(20))
print("="*200)
print()

# using_ml_analysis()

data_based_features_df.to_csv('stock_intermediate_for_ml_data.csv', index=False)
names=data_based_features_df.columns.tolist()
print(names)