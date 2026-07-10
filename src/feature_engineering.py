import pandas as pd
import numpy as np

from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands


def create_features(price):

    # -------------------------
    # Convert Date
    # -------------------------
    price["DateTime"] = pd.to_datetime(price["DateTime"])

    # -------------------------
    # Moving Averages
    # -------------------------
    price["MA_5"] = price["Close"].rolling(5).mean()
    price["MA_20"] = price["Close"].rolling(20).mean()

    # -------------------------
    # Price Features
    # -------------------------
    price["Price_Change"] = price["Close"] - price["Open"]

    price["High_Low_Range"] = (
        price["High"] - price["Low"]
    )

    price["Return"] = (
        price["Close"].pct_change()
    )

    # -------------------------
    # RSI
    # -------------------------
    rsi = RSIIndicator(
        close=price["Close"],
        window=14
    )

    price["RSI"] = rsi.rsi()

    # -------------------------
    # MACD
    # -------------------------
    macd = MACD(
        close=price["Close"]
    )

    price["MACD"] = macd.macd()

    price["MACD_Signal"] = (
        macd.macd_signal()
    )

    price["MACD_Histogram"] = (
        macd.macd_diff()
    )

    # -------------------------
    # Bollinger Bands
    # -------------------------
    bb = BollingerBands(
        close=price["Close"],
        window=20,
        window_dev=2
    )

    price["BB_Upper"] = bb.bollinger_hband()

    price["BB_Middle"] = bb.bollinger_mavg()

    price["BB_Lower"] = bb.bollinger_lband()

    # -------------------------
    # Target
    # -------------------------
    price["Target"] = np.where(
        price["Close"].shift(-1) > price["Close"],
        1,
        0
    )

    # -------------------------
    # Remove Missing Values
    # -------------------------

    # -------------------------
    # RL State Features
    # -------------------------
    
    # RSI State
    # 0 = Oversold
    # 1 = Neutral
    # 2 = Overbought
    
    price["RSI_State"] = np.where(
        price["RSI"] < 30,
        0,
        np.where(price["RSI"] > 70, 2, 1)
    )
    
    # MACD State
    # 1 = Bullish
    # 0 = Bearish
    
    price["MACD_State"] = np.where(
        price["MACD"] > price["MACD_Signal"],
        1,
        0
    )
    
    # Trend State
    # 1 = Uptrend
    # 0 = Downtrend
    
    price["Trend_State"] = np.where(
        price["MA_5"] > price["MA_20"],
        1,
        0
    )

    price = price.dropna().copy()

    return price