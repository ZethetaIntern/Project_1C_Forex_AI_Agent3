import pandas as pd


def load_data():
    price = pd.read_csv("../data/forex_price_data.csv")
    mc = pd.read_csv("../data/mc_scenarios.csv")
    trade = pd.read_csv("../data/trade_log.csv")

    price["DateTime"] = pd.to_datetime(price["DateTime"])

    return price, mc, trade


if __name__ == "__main__":
    price, mc, trade = load_data()

    print(price.head())