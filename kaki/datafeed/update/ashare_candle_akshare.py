import akshare as ak
from tqdm import tqdm
import pandas as pd
from pymongo import MongoClient
from datetime import datetime

# MongoDB connection
client = MongoClient("mongodb://192.168.31.120:27017/")  # Update with your MongoDB connection string
db = client["ashare_candle"]  # Database name
collection = db["daily_data"]  # Collection name

# Function to fetch or update stock data from MongoDB
def get_stock_data_from_db(stock_code, start_date, end_date):
    # Check if data exists in DB and is up-to-date
    latest_record = collection.find_one({"symbol": stock_code}, sort=[("date", -1)])
    if latest_record and latest_record['date'] >= end_date:
        # Data is up-to-date, return from DB
        return pd.DataFrame(list(collection.find({"symbol": stock_code})))
    else:
        # Fetch data from API and update DB
        stock_data_df = ak.stock_zh_a_hist(symbol=stock_code, start_date=start_date, end_date=end_date, adjust="hfq")
        if not stock_data_df.empty:
            # Ensure all dates are in datetime.datetime format
            stock_data_df['日期'] = pd.to_datetime(stock_data_df['日期'])
            stock_data_df['date'] = stock_data_df['日期'].apply(lambda x: datetime.combine(x, datetime.min.time()))

            # Additional check to confirm the data type
            if not all(isinstance(d, datetime) for d in stock_data_df['date']):
                print(f"Data type issue in stock {stock_code}")

            # MongoDB insertion
            stock_data_df['symbol'] = stock_code
            collection.delete_many({"symbol": stock_code})  # Remove old data
            collection.insert_many(stock_data_df.to_dict('records'))  # Insert new data

        return stock_data_df
    
# Function to calculate MACD
def calculate_macd(df, short_period=12, long_period=26, signal_period=9):
    df['EMA12'] = df['收盘'].ewm(span=short_period, adjust=False).mean()
    df['EMA26'] = df['收盘'].ewm(span=long_period, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    return df

# Function to check for gold crossover
def has_gold_crossover(df):
    if len(df) < 2:
        return False
    # Checking if the latest MACD crossed above the latest Signal Line
    # and the previous MACD was below the previous Signal Line
    return df.iloc[-1]['MACD'] > df.iloc[-1]['Signal_Line'] and df.iloc[-2]['MACD'] < df.iloc[-2]['Signal_Line']

# Get stock list
stock_list_df = ak.stock_zh_a_spot_em()
stock_list = stock_list_df["代码"].tolist()

# Filtering out ST stocks
stock_zh_a_st_em_list = ak.stock_zh_a_st_em()["代码"].tolist()
stock_list_safe = [stock for stock in stock_list if stock not in stock_zh_a_st_em_list]

print(f"Total stocks: {len(stock_list_safe)}")

# List to store stocks with gold crossover
stocks_with_gold_crossover = []

for stock_code in tqdm(stock_list_safe):
    try:
        # Get or update stock daily data from MongoDB
        stock_data_df = get_stock_data_from_db(stock_code, "20200101", "20240118")
        if stock_data_df.empty:
            continue

        # Calculate MACD
        stock_data_df_with_macd = calculate_macd(stock_data_df)

        # Check for gold crossover
        if has_gold_crossover(stock_data_df_with_macd):
            stocks_with_gold_crossover.append(stock_code)
    except Exception as e:
        print(f"Error processing {stock_code}: {e}")

print(stocks_with_gold_crossover)