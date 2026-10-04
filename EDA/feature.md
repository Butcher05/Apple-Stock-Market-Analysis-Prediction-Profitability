# Create a copy
data = df.copy()

# Basic returns
data["Daily_Return"] = data["Close"].pct_change()

data["Return_5D"] = data["Close"].pct_change(5)

data["Return_10D"] = data["Close"].pct_change(10)

data["Return_20D"] = data["Close"].pct_change(20)

# Moving averages
data["SMA_5"] = data["Close"].rolling(window=5).mean()

data["SMA_10"] = data["Close"].rolling(window=10).mean()

data["SMA_20"] = data["Close"].rolling(window=20).mean()

data["SMA_50"] = data["Close"].rolling(window=50).mean()

# Exponential moving averages
data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()

data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()

# RSI
rsi_indicator = RSIIndicator(
    close=data["Close"],
    window=14
)

data["RSI"] = rsi_indicator.rsi()

# MACD
macd_indicator = MACD(
    close=data["Close"],
    window_slow=26,
    window_fast=12,
    window_sign=9
)

data["MACD"] = macd_indicator.macd()

data["MACD_Signal"] = macd_indicator.macd_signal()

data["MACD_Diff"] = macd_indicator.macd_diff()

# Volatility
data["Volatility_10D"] = data["Daily_Return"].rolling(10).std()

data["Volatility_20D"] = data["Daily_Return"].rolling(20).std()

# High-Low range
data["High_Low_Range"] = (
    data["High"] - data["Low"]
) / data["Close"]

# Open-Close range
data["Open_Close_Range"] = (
    data["Close"] - data["Open"]
) / data["Open"]

# Volume change
data["Volume_Change"] = data["Volume"].pct_change()

# Volume moving average
data["Volume_SMA_20"] = data["Volume"].rolling(20).mean()

print("Technical indicators created!")

display(data.tail())
# Historical closing price lags
for lag in [1, 2, 3, 5, 10, 20]:
    data[f"Close_Lag_{lag}"] = data["Close"].shift(lag)

# Historical return lags
for lag in [1, 2, 3, 5, 10]:
    data[f"Return_Lag_{lag}"] = data["Daily_Return"].shift(lag)

# Volume lags
for lag in [1, 2, 5]:
    data[f"Volume_Lag_{lag}"] = data["Volume"].shift(lag)

print("Lag features created!")

display(data.tail())
# Next trading day's return
data["Target_Return"] = (
    data["Close"].shift(-1) / data["Close"] - 1
)

# Optional: next day's closing price
data["Target_Close"] = data["Close"].shift(-1)

display(
    data[
        ["Close", "Target_Return", "Target_Close"]
    ].head(10)
)feature_columns = [
    "Daily_Return",
    "Return_5D",
    "Return_10D",
    "Return_20D",

    "SMA_5",
    "SMA_10",
    "SMA_20",
    "SMA_50",

    "EMA_12",
    "EMA_26",

    "RSI",

    "MACD",
    "MACD_Signal",
    "MACD_Diff",

    "Volatility_10D",
    "Volatility_20D",

    "High_Low_Range",
    "Open_Close_Range",

    "Volume_Change",
    "Volume_SMA_20",

    "Close_Lag_1",
    "Close_Lag_2",
    "Close_Lag_3",
    "Close_Lag_5",
    "Close_Lag_10",
    "Close_Lag_20",

    "Return_Lag_1",
    "Return_Lag_2",
    "Return_Lag_3",
    "Return_Lag_5",
    "Return_Lag_10",

    "Volume_Lag_1",
    "Volume_Lag_2",
    "Volume_Lag_5"
]

# Normalize price-related lag features by current close
for lag in [1, 2, 3, 5, 10, 20]:
    data[f"Close_Lag_{lag}"] = (
        data[f"Close_Lag_{lag}"] / data["Close"] - 1
    )

# Normalize volume SMA and lags by current volume
data["Volume_SMA_20"] = (
    data["Volume_SMA_20"] / data["Volume"] - 1
)

for lag in [1, 2, 5]:
    data[f"Volume_Lag_{lag}"] = (
        data[f"Volume_Lag_{lag}"] / data["Volume"] - 1
    )

# Remove rows with NaN values
data = data.dropna().copy()

print("Final dataset shape:", data.shape)

X = data[feature_columns]
y = data["Target_Return"]

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)