import numpy as np
import pandas as pd
import requests
from binance.client import Client
import datetime

def download_klines(symbol, interval, start_time=None, end_time=None, limit=1000):
    """
    Downloads historical klines data from Binance.
    
    Args:
    symbol (str): Trading pair symbol e.g. "BTCUSDT"
    interval (str): Kline interval, e.g. "1h" for hourly, "1d" for daily
    start_time (datetime): Optional start time for data (default is None, which downloads all data)
    end_time (datetime): Optional end time for data (default is None, which downloads up to the current time)
    limit (int): Number of klines to download per request (default is 1000, max is 1000)
    
    Returns:
    pandas.DataFrame: DataFrame with kline data
    """
    
    # Create Binance API client
    api_key = 'your_api_key'
    api_secret = 'your_api_secret'
    client = Client(api_key, api_secret)

    # Convert interval to Binance API format
    interval_map = {
        "1m": Client.KLINE_INTERVAL_1MINUTE,
        "3m": Client.KLINE_INTERVAL_3MINUTE,
        "5m": Client.KLINE_INTERVAL_5MINUTE,
        "15m": Client.KLINE_INTERVAL_15MINUTE,
        "30m": Client.KLINE_INTERVAL_30MINUTE,
        "1h": Client.KLINE_INTERVAL_1HOUR,
        "2h": Client.KLINE_INTERVAL_2HOUR,
        "4h": Client.KLINE_INTERVAL_4HOUR,
        "6h": Client.KLINE_INTERVAL_6HOUR,
        "8h": Client.KLINE_INTERVAL_8HOUR,
        "12h": Client.KLINE_INTERVAL_12HOUR,
        "1d": Client.KLINE_INTERVAL_1DAY,
        "3d": Client.KLINE_INTERVAL_3DAY,
        "1w": Client.KLINE_INTERVAL_1WEEK,
        "1M": Client.KLINE_INTERVAL_1MONTH
    }
    kline_interval = interval_map[interval]

    # Set start and end times if specified
    if start_time is not None:
        start_time = int(start_time.timestamp() * 1000)
    if end_time is not None:
        end_time = int(end_time.timestamp() * 1000)

    # Download klines data in multiple requests if necessary
    klines = []
    while True:
        data = client.futures_klines(symbol=symbol, interval=kline_interval, startTime=start_time, endTime=end_time, limit=limit)
        if len(data) == 0:
            break
        klines += data
        if len(data) < limit:
            break

def download_klines(symbol, interval, start_time=None, end_time=None, limit=1000):
    """
    Downloads historical klines data from Binance.
    
    Args:
    symbol (str): Trading pair symbol e.g. "BTCUSDT"
    interval (str): Kline interval, e.g. "1h" for hourly, "1d" for daily
    start_time (datetime): Optional start time for data (default is None, which downloads all data)
    end_time (datetime): Optional end time for data (default is None, which downloads up to the current time)
    limit (int): Number of klines to download per request (default is 1000, max is 1000)
    
    Returns:
    pandas.DataFrame: DataFrame with kline data
    """
    
    # Create Binance API client
    api_key = 'your_api_key'
    api_secret = 'your_api_secret'
    client = Client(api_key, api_secret)

    # Convert interval to Binance API format
    interval_map = {
        "1m": Client.KLINE_INTERVAL_1MINUTE,
        "3m": Client.KLINE_INTERVAL_3MINUTE,
        "5m": Client.KLINE_INTERVAL_5MINUTE,
        "15m": Client.KLINE_INTERVAL_15MINUTE,
        "30m": Client.KLINE_INTERVAL_30MINUTE,
        "1h": Client.KLINE_INTERVAL_1HOUR,
        "2h": Client.KLINE_INTERVAL_2HOUR,
        "4h": Client.KLINE_INTERVAL_4HOUR,
        "6h": Client.KLINE_INTERVAL_6HOUR,
        "8h": Client.KLINE_INTERVAL_8HOUR,
        "12h": Client.KLINE_INTERVAL_12HOUR,
        "1d": Client.KLINE_INTERVAL_1DAY,
        "3d": Client.KLINE_INTERVAL_3DAY,
        "1w": Client.KLINE_INTERVAL_1WEEK,
        "1M": Client.KLINE_INTERVAL_1MONTH
    }
    kline_interval = interval_map[interval]

    # Set start and end times if specified
    if start_time is not None:
        start_time = int(start_time.timestamp() * 1000)
    if end_time is not None:
        end_time = int(end_time.timestamp() * 1000)

    # Download klines data in multiple requests if necessary
    klines = []
    while True:
        data = client.futures_klines(symbol=symbol, interval=kline_interval, startTime=start_time, endTime=end_time, limit=limit)
        if len(data) == 0:
            break
        klines += data
        if len(data) < limit:
            break
        start_time = int(data[-1][0]) + 1

    # Convert data to DataFrame
    df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "num_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignored"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")

    return df

def get_binance_data(symbol, interval, start_time=None, end_time=None):
    # Binance API endpoint for klines
    endpoint = 'https://api.binance.com/api/v3/klines'

    # Define query parameters
    params = {
        'symbol': symbol,
        'interval': interval
    }

    if start_time is not None:
        params['startTime'] = int(pd.Timestamp(start_time).timestamp() * 1000)

    if end_time is not None:
        params['endTime'] = int(pd.Timestamp(end_time).timestamp() * 1000)

    # Send request to Binance API
    response = requests.get(endpoint, params=params)

    # Convert response to pandas DataFrame
    data = pd.DataFrame(response.json(), columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                                  'close_time', 'quote_asset_volume', 'number_of_trades',
                                                  'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                                                  'ignore'])

    # Convert timestamp to datetime format
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

    # Convert columns to float data type
    data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].astype(float)

    # Set timestamp as index
    data.set_index('timestamp', inplace=True)

    return data

#You can call this function by passing in the trading pair symbol (e.g. "BTCUSDT"), the desired interval (e.g. "1h"), and optional start and 
#end times in the format "YYYY-MM-DD HH:MM:SS":


def get_binance_data(symbol, interval, start_time=None, end_time=None):
    # Binance API endpoint for klines
    endpoint = 'https://api.binance.com/api/v3/klines'

    # Define query parameters
    params = {
        'symbol': symbol,
        'interval': interval
    }

    if start_time is not None:
        params['startTime'] = int(pd.Timestamp(start_time).timestamp() * 1000)

    if end_time is not None:
        params['endTime'] = int(pd.Timestamp(end_time).timestamp() * 1000)

    # Send request to Binance API
    response = requests.get(endpoint, params=params)

    # Convert response to pandas DataFrame
    data = pd.DataFrame(response.json(), columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                                  'close_time', 'quote_asset_volume', 'number_of_trades',
                                                  'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                                                  'ignore'])

    # Convert timestamp to datetime format
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

    # Convert columns to float data type
    data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].astype(float)

    # Set timestamp as index
    data.set_index('timestamp', inplace=True)

    return data
data = get_binance_data('BTCUSDT', '1h', '2022-04-01 00:00:00', '2022-04-30 23:59:59')
# Construct the Binance API URL
url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}'

# Add optional start and end times to the URL if provided
if start_time is not None:
    url += f'&startTime={start_time}'
if end_time is not None:
    url += f'&endTime={end_time}'

# Send the HTTP request to the Binance API
response = requests.get(url)

# Check if the request was successful
if response.status_code != 200:
    raise Exception('Failed to get Binance data')

# Load the JSON response into a Pandas DataFrame
df = pd.read_json(response.text)

# Rename the columns of the DataFrame to match our naming convention
df.columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
              'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
              'Taker buy quote asset volume', 'Ignore']

# Convert the timestamp columns from Unix timestamp to Python datetime format
df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')

# Set the index of the DataFrame to be the Open time column
df.set_index('Open time', inplace=True)

# Convert the DataFrame columns from strings to floats
df = df.astype(float)

    return df
def getCoinBalance(client, currency):
    balance = float(client.get_asset_balance(asset=currency)['free'])
    return balance

#Market buy
def executeBuy(client, market, qtyBuy):
    
    order = client.order_market_buy(symbol=market,quantity=qtyBuy)

#Market sell
def executeSell(client, market, qtySell):

    order = client.order_market_sell(symbol=market, quantity=qtySell)

#format the data correctly for later use
def CreateOpenHighLowCloseVolumeData(indata):
    
    out = pd.DataFrame()
    
    d = []
    o = []
    h = []
    l = []
    c = []
    v = []
    for i in indata:
        #print(i)
        d.append(float(i[0]))
        o.append(float(i[1]))
        h.append(float(i[2]))
        c.append(float(i[3]))
        l.append(float(i[4]))
        v.append(float(i[5]))

    out['date'] = d
    out['open'] = o
    out['high'] = h
    out['low'] = l
    out['close'] = c
    out['volume'] = v
    
    #print(out)
    
    return out

#This is the main function for feature creation and manipulation, modify this by adding your own functions and feature creation
#prehaps try using technical analysis libraries for RSI or
#Sentiment data from bitfinex or Fear and greed data
def FeatureCreation(candles):
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'closetime', 'quoteassetvolume', 'numberoftrades', 'takerbuybaseassetvolume', 'takerbuyquoteassetvolume', 'ignore'])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    
    df['SMA5'] = df['close'].rolling(window=5).mean()
    df['SMA10'] = df['close'].rolling(window=10).mean()
    df['SMA15'] = df['close'].rolling(window=15).mean()
    df['SMA20'] = df['close'].rolling(window=20).mean()
    df['SMA25'] = df['close'].rolling(window=25).mean()
    df['SMA30'] = df['close'].rolling(window=30).mean()
    
    df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA15'] = df['close'].ewm(span=15, adjust=False).mean()
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA25'] = df['close'].ewm(span=25, adjust=False).mean()
    df['EMA30'] = df['close'].ewm(span=30, adjust=False).mean()
    
    df['RSI'] = talib.RSI(df['close'].values, timeperiod=14)
    df['ADX'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
    
    return df

    convertedData = CreateOpenHighLowCloseVolumeData(indata)
    FeatureData = pd.DataFrame()
    FeatureData['o'] = convertedData['open']
    FeatureData['h'] = convertedData['high']
    FeatureData['l'] = convertedData['low']
    FeatureData['c'] = convertedData['close']
    FeatureData['v'] = convertedData['volume']
    candleRatios(FeatureData)
    StepData(FeatureData['c'],FeatureData)
    GetChangeData(FeatureData)
    
    return FeatureData
    
#Create targets for our machine learning model. This is done by predicting if the closing price of the next candle will 
#be higher or lower than the current one.
def CreateTargets(data, offset):
    
    y = []
    
    
    for i in range(0, len(data)-offset):
        current = float(data[i][3])
        comparison = float(data[i+offset][3])
        
        if current<comparison:
            y.append(1)

        elif current>=comparison:
            y.append(0)
            
    return y

#FEATURE EXAMPLES
#Calculate the change in the values of a column
def GetChangeData(x):

    cols = x.columns
    
    for i in cols:
        j = "c_" + i
        
        try:
            dif = x[i].diff()
            x[j] = dif
        except Exception as e:
            print(e)
            
#FEATURE EXAMPLES  
#Calculate the percentage change between this bar and the previoud x bars
def ChangeTime(x, step):
    
    out = []
    
    for i in range(len(x)):
        try:
            a = x[i]
            b = x[i-step]
            
            change = (1 - b/a) 
            out.append(change)
        except Exception as e:
            out.append(0)
    
    return out

#FEATURE EXAMPLES
#Automate the creation of percentage changes for 48 candles.  
def StepData(x, data):
    
    for i in range(1,48):
        
        data[str(i)+"StepDifference"] = ChangeTime(x, i)


#FEATURE EXAMPLES
#Features that take into acount the relations between the candle values  
def candleRatios(data):
    data['v_c'] = data['v'] / data['c']
    data['h_c'] = data['h'] / data['c']
    data['o_c'] = data['o'] / data['c']
    data['l_c'] = data['l'] / data['c']
    
    data['h_l'] = data['h'] / data['l']
    data['v_l'] = data['v'] / data['l']
    data['o_l'] = data['o'] / data['l']
    
    data['o_h'] = data['o'] / data['h']
    data['v_h'] = data['v'] / data['h']
    
    data['v_o'] = data['v'] / data['o']

# RSI
def rsi(data, window=14):
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ema_up = up.ewm(com=window-1, adjust=True).mean()
    ema_down = down.ewm(com=window-1, adjust=True).mean()
    rs = ema_up/ema_down
    return pd.Series(100 - (100/(1 + rs)), name="RSI")

# Bollinger Bands
def bollinger_bands(data, window=20):
    rolling_mean = data['close'].rolling(window).mean()
    rolling_std = data['close'].rolling(window).std()
    upper_band = rolling_mean + (rolling_std*2)
    lower_band = rolling_mean - (rolling_std*2)
    return upper_band, lower_band

# Bitfinex Fear and Greed
def get_bitfinex_fear_and_greed():
    url = 'https://api.alternative.me/fng/'
    response = requests.get(url).json()
    value = response['data'][0]['value']
    return value

# Fibonacci Retracement
def fibonacci_retracement(data, high_col='high', low_col='low', levels=[0.236, 0.382, 0.5, 0.618, 0.786]):
    data['price_range'] = data[high_col] - data[low_col]
    data['fib_levels'] = data[high_col] - data['price_range']*np.array(levels)
    return data[['fib_levels']]

# Fibonacci Extension
def fibonacci_extension(data, high_col='high', low_col='low', levels=[0.382, 0.5, 0.618, 1.0, 1.382, 1.618]):
    data['price_range'] = data[high_col] - data[low_col]
    data['fib_levels'] = data[high_col] + data['price_range']*np.array(levels)
    return data[['fib_levels']]

# Fibonacci Average Price
def fibonacci_avg_price(data, high_col='high', low_col='low'):
    data['price_range'] = data[high_col] - data[low_col]
    data['fib_avg_price'] = (data[high_col] + data[low_col] + data['price_range']*0.5)
    return data[['fib_avg_price']]

# MACD
def MACD(data, fast=12, slow=26, signal=9):
    exp1 = data['close'].ewm(span = fast, adjust=False).mean()
    exp2 = data['close'].ewm(span = slow, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span = signal, adjust=False).mean()
    return macd,signal

# ADX
def ADX(data, window=14):
    delta_high = data['high'].diff()
    delta_low = data['low'].diff()
    tr = pd.concat([data['high']-data['low'], (data['high']-delta_high)-(data['low']+delta_low)], axis=1)
    atr = tr.max(axis=1)
    atr_avg = atr.rolling(window).mean()
    up = data['high'].diff().fillna(0)
    down = -1*data['low'].diff().fillna(0)
    zero = 0.0001
    pos_directional_index = up/(atr_avg+zero)
    neg_direction

def atr(data, window=14):
    high = data['high']
    low = data['low']
    close = data['close']
    tr1 = abs(high - low)
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=False)
    atr = tr.rolling(window).mean()
    return atr

def talon_sniper(data, window=14):
    high = data['high']
    low = data['low']
    close = data['close']
    talon = pd.Series(0.0, index=data.index)
    for i in range(window, len(data)):
        max_high = max(high[i - window:i + 1])
        min_low = min(low[i - window:i + 1])
        talon[i] = close[i] - ((max_high - min_low) / 2)
    return talon

def time_weighted_average_price(data, window=14):
    high = data['high']
    low = data['low']
    close = data['close']
    twap = pd.Series(0.0, index=data.index)
    total_volume = pd.Series(0.0, index=data.index)
    for i in range(window, len(data)):
        twap[i] = ((close[i] + high[i] + low[i]) / 3) * data['volume'][i]
        total_volume[i] = data['volume'][i]
    twap = twap.rolling(window).sum() / total_volume.rolling(window).sum()
    return twap

def triple_ema(data, period1=8, period2=15, period3=30):
    ema1 = data['close'].ewm(span=period1, adjust=False).mean()
    ema2 = data['close'].ewm(span=period2, adjust=False).mean()
    ema3 = data['close'].ewm(span=period3, adjust=False).mean()
    return pd.concat([ema1, ema2, ema3], axis=1)

def stochastic_rsi(data, window=14, smoothk=3, smoothd=3):
    close = data['close']
    rsi = ((close - close.rolling(window).mean()) /
           close.rolling(window).std())
    rsi_k = rsi.rolling(smoothk).mean()
    rsi_d = rsi_k.rolling(smoothd).mean()
    return pd.concat([rsi_k, rsi_d], axis=1)

def ma_cross(data, short_window=10, long_window=30):
    short_ma = data['close'].rolling(short_window).mean()
    long_ma = data['close'].rolling(long_window).mean()
    signal = pd.Series(np.where(short_ma > long_ma, 1.0, 0.0),
                       index=data.index)
    return signal

def average_day_range(data, window=14):
    high = data['high']
    low = data['low']
    adr = pd.Series(0.0, index=data.index)
    for i in range(window, len(data)):
        adr[i] = (high[i] - low[i]) / window
    return adr

def volume(data, window=24):
    return data['volume'].rolling(window=window).sum().rename('volume_{}h'.format(window))

def talon_sniper(data, fast_ma=5, slow_ma=10):
    fast = data['close'].rolling(window=fast_ma).mean()
    slow = data['close'].rolling(window=slow_ma).mean()
    buy_signal = (fast > slow).astype(int)
    sell_signal = (fast < slow).astype(int)
    return pd.DataFrame({'talon_sniper_buy': buy_signal, 'talon_sniper_sell': sell_signal})

def time_weighted_average_price(data, window=24):
    return ((data['close']*data['volume']).rolling(window=window).sum() / data['volume'].rolling(window=window).sum()).rename('twap_{}h'.format(window))

def triple_ema(data, fast_ema=12, medium_ema=26, slow_ema=50):
    fast = data['close'].ewm(span=fast_ema, adjust=False).mean()
    medium = data['close'].ewm(span=medium_ema, adjust=False).mean()
    slow = data['close'].ewm(span=slow_ema, adjust=False).mean()
    return pd.DataFrame({'tema_fast': fast, 'tema_medium': medium, 'tema_slow': slow})

def stochastic(data, fastk_period=14, slowk_period=3, slowd_period=3):
    low_min  = data['low'].rolling(window=fastk_period).min()
    high_max = data['high'].rolling(window=fastk_period).max()
    fastk = 100*((data['close'] - low_min)/(high_max - low_min))
    slowk = fastk.rolling(window=slowk_period).mean()
    slowd = slowk.rolling(window=slowd_period).mean()
    return pd.DataFrame({'stochastic_fastk': fastk, 'stochastic_slowk': slowk, 'stochastic_slowd': slowd})

def stochastic_rsi(data, window=14, fastk_period=14, slowk_period=3, slowd_period=3):
    rsi_values = rsi(data['close'], window)
    low_min  = rsi_values.rolling(window=fastk_period).min()
    high_max = rsi_values.rolling(window=fastk_period).max()
    fastk = 100*((rsi_values - low_min)/(high_max - low_min))
    slowk = fastk.rolling(window=slowk_period).mean()
    slowd = slowk.rolling(window=slowd_period).mean()
    return pd.DataFrame({'stoch_rsi_fastk': fastk, 'stoch_rsi_slowk': slowk, 'stoch_rsi_slowd': slowd})

def ma_cross(data, fast_ma=5, slow_ma=10):
    fast = data['close'].rolling(window=fast_ma).mean()
    slow = data['close'].rolling(window=slow_ma).mean()
    buy_signal = (fast > slow).astype(int)
    sell_signal = (fast < slow).astype(int)
    return pd.DataFrame({'ma_cross_buy': buy_signal, 'ma_cross_sell': sell_signal})

def average_day_range(data, window=24):
    return (data['high'] - data['low']).rolling(window=window).mean().rename('adr_{}h'.format(window))

def volume(data, window=24):
    return data['volume'].rolling(window=window).sum().rename('volume_{}h'.format(window))
