import pandas as pd
import pymongo
import logging
import time
import concurrent.futures
import requests
import okx.PublicData as PublicData
import okx.MarketData as MarketData
from pymongo import MongoClient
import random
import numpy as np

class CryptoDataUpdater:
    def __init__(self):
        self.client = MongoClient('localhost', 27017)
        self.db = self.client.crypto
        self.collection = self.db.crypto_kline
        self.base_url = "https://www.okx.com/api/v5/market/history-candles"
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", 
                            handlers=[logging.FileHandler("crypto_AIO.log"), logging.StreamHandler()])
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }

    def find_bounds(self, inst_id:str, bar:str) -> tuple:
        latest_record = self.collection.find_one({'instId': inst_id, 'bar': bar}, sort=[('timestamp', pymongo.DESCENDING)])
        earlist_record = self.collection.find_one({'instId': inst_id, 'bar': bar}, sort=[('timestamp', pymongo.ASCENDING)])
        if latest_record and earlist_record:
            # a, b
            return latest_record['timestamp'], earlist_record['timestamp']
        return None, None
    
    def post_data_clean(self):
        pass

    def insert_data_to_mongodb(self, data):
        if not data.empty:
            data_dict = data.to_dict('records')
            self.collection.insert_many(data_dict)
            logging.info(f"Inserted {len(data_dict)} new records.")

    def get_all_coin_pairs(self):
        publicDataAPI = PublicData.PublicAPI(flag="0")
        spot_result = publicDataAPI.get_instruments(instType="SPOT")
        swap_result = publicDataAPI.get_instruments(instType="SWAP")
        return [i["instId"] for i in spot_result["data"]] + [i["instId"] for i in swap_result["data"]]
    
    def process_kline(self, data, inst_id, bar):
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm'])
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms')
        numeric_fields = ['open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote', 'confirm']
        for field in numeric_fields:
            df[field] = pd.to_numeric(df[field], errors='coerce')
        df['instId'] = inst_id
        df['bar'] = bar
        return df

    def newest_data_ts(self, inst_id, bar):
        result = self.get_history_candlesticks(inst_id, bar, limit=1).json()["data"][0][0]
        # logging.info(f"Newest data for {inst_id}-{bar} is at {self.ts_to_date(result)}.")
        return result

    def fetch_kline_data(self, inst_id: str, bar: str, initial_delay=1):
        # Check DB for existing data
        db_b, db_a = self.find_bounds(inst_id, bar)
        # Initially, 'before' is None to fetch the latest data
        a = None
        b = None
        is_first_time = True

        # Initialize dynamic sleep time
        sleep_time = initial_delay
        min_sleep_time = 0.1  # Minimum sleep time
        max_sleep_time = 5.0  # Maximum sleep time
        sleep_adjustment_factor = 0.5  # Factor to adjust sleep time

        while True:
            try:
                # logging.info(f"Fetching data for {inst_id} {bar} from {self.ts_to_date(a)} to {self.ts_to_date(b)}.")
                params = {
                    'instId': inst_id,
                    'before': str(b) if b else "",
                    'after': str(a) if a else "",
                    'bar': bar
                }
                
                response = requests.get(self.base_url, params = params, headers = self.headers, timeout=3)
                
                # If http status is not too many request, we can assume that we need to break the loop
                if response.status_code == 200:
                    result = response.json(parse_int=np.int64)
                    logging.info(f"Received {len(result['data'])} records for {inst_id}-{bar}.")
                    # Check if result is empty or contains data
                    if not result['data']:
                        logging.info(f"No more data to fetch or empty data returned for {inst_id}-{bar}.")
                        break
                    else:

                        # Process the data
                        df = self.process_kline(result['data'], inst_id=inst_id, bar=bar)
                        # print(df)
                        # Insert data to MongoDB if applicable
                        if not df.empty:
                            self.insert_data_to_mongodb(df)
                            logging.info(f"Successfully inserted data for {inst_id} {bar}.")

                        # Update 'before' and 'after' for the next request
                        a = int(result['data'][-1][0])
                        if is_first_time:
                            time_interval = int(result['data'][0][0]) - a
                            is_first_time = False
                        b = a - time_interval - 4 + random.randint(1, 10)*2
                        continue
                    
                elif response.status_code == 429:
                    # Sleep for a random time between 1 and 5 seconds
                    logging.info(f"Too many requests. Sleeping for {sleep_time:.2f} seconds.")
                    time.sleep(sleep_time)
                
                else:
                    logging.error(f"Failed to fetch data: {result.text}")
                    break

                # Reduce sleep time after a successful request, but not below the minimum
                sleep_time = max(min_sleep_time, sleep_time - sleep_adjustment_factor)

                time.sleep(sleep_time)

            except Exception as e:
                logging.error(f"Error occurred: {e}")
                # Increase sleep time after an error, but not above the maximum
                sleep_time = min(max_sleep_time, sleep_time + sleep_adjustment_factor)
                time.sleep(sleep_time)

    def update_data(self):
        pair_list = self.get_all_coin_pairs()
        # pair_list = ["BTC-USDT-SWAP"]
        bar_sizes = ['1m', '3m', '5m', '15m', '30m', '1H', '4H', '1D', '1W']
        # bar_sizes = ['1m']
        # self.fetch_kline_data(inst_id=pair_list[0], bar=bar_sizes[0])
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for inst_id in pair_list:
                for bar in bar_sizes:
                    futures.append(executor.submit(self.fetch_kline_data, inst_id, bar))
            concurrent.futures.wait(futures)

if __name__ == "__main__":
    updater = CryptoDataUpdater()
    updater.update_data()
