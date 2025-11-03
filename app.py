# ==========================================
# 📊 Trading Prediction App (Streamlit)
# ==========================================
# Predict next-day stock price movement (UP/DOWN)
# using RandomForestClassifier and Yahoo Finance data.

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ----------------------------
# 🧱 Sidebar Inputs
# ----------------------------
st.sidebar.title("📈 Stock Predictor")
asset = st.sidebar.selectbox("Select Asset", ["AAPL", "GOOG", "MSFT", "TSLA", "BTC-USD", "ETH-USD"])
period = st.sidebar.selectbox("Select Time Frame", ["1y", "2y", "5y"])

# ----------------------------
# 📥 Fetch Data
# ----------------------------
st.write(f"### Fetching data for **{asset}** ({period})...")
data = yf.download(asset, period=period, interval="1d")

# Feature engineering
data["Return"] = data["Close"].pct_change()
data["MA5"] = data["Close"].rolling(5).mean()
data["MA10"] = data["Close"].rolling(10).mean()
data["MA20"] = data["Close"].rolling(20).mean()
data["Target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)
data.dropna(inplace=True)

# ----------------------------
# 🧠 Train Model
# ----------------------------
features = ["Return", "MA5", "MA10", "MA20"]
X = data[features]
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ----------------------------
# 📊 Results
# ----------------------------
accuracy = accuracy_score(y_test, y_pred)
st.write(f"### ✅ Model Accuracy: **{accuracy*100:.2f}%**")

# ----------------------------
# 🧩 Plot Predictions (fixed)
# ----------------------------
data_test = data.iloc[len(X_train):].copy()
data_test = data_test.iloc[:len(y_pred)]  # ensure same length
data_test["Predicted"] = y_pred

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data_test.index, data_test["Close"], label="Actual Price")

# Only plot predicted buy signals
buy_signals = data_test[data_test["Predicted"] == 1]
ax.scatter(
    buy_signals.index,
    buy_signals["Close"],
    label="Predicted Buy Signal",
    color="green",
    marker="^",
    alpha=0.8
)

ax.set_title(f"{asset} — Predicted Trading Signals")
ax.legend()
st.pyplot(fig)

# ----------------------------
# 🔮 Predict Next Day
# ----------------------------
latest = data[features].iloc[-1:].values
next_day = model.predict(latest)[0]

if next_day == 1:
    st.success("📈 The model predicts: Price will go **UP** tomorrow.")
else:
    st.error("📉 The model predicts: Price will go **DOWN** tomorrow.")
