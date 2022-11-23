import pyupbit

def save_ohlcv():
    df = pyupbit.get_ohlcv("KRW-BTC", count=600)
    df.to_csv("./data/KRW_BTC.csv", sep=",")

def preprocessing_ohlcv():
    df = pyupbit.get_ohlcv("KRW-BTC", count=1000)

    '''    
    ohlcv 데이터는 비율 데이터로 변환하여 사용
    open_ratio      = 오늘 시가 / 어제 시가
    high_open_ratio = 고가 / 시가
    low_open_ratio  = 저가 / 시가
    close_open_ratio= 종가 / 시가
    volume_ratio    = 오늘 거래량 / 어제 거래량
    ma_7            = 7일 이평가 
    ma_15           = 15일 이평가
    ma_30           = 30일 이평가
    ma_15_7_ratio   = 15 이평 / 7 이평
    ma_30_7_ratio   = 30 이평 / 7 이평
    ma_30_15_ratio  = 30 이평 / 15 이평    
    '''

    df["open_ratio"] = df.open / df.open.shift(1)
    df["high_open_ratio"] = df.high / df.open
    df["low_open_ratio"] = df.low / df.open
    df["close_open_ratio"] = df.close / df.open
    df["volume_ratio"] = df.volume / df.volume.shift(1)
    df["ma_7"] = df.close.rolling(window=7).mean()
    df["ma_15"] = df.close.rolling(window=15).mean()
    df["ma_30"] = df.close.rolling(window=30).mean()
    df["ma_15_7_ratio"] = df.ma_15 / df.ma_7
    df["ma_30_7_ratio"] = df.ma_30 / df.ma_7
    df["ma_30_15_ratio"] = df.ma_30 / df.ma_15

    df["open_ratio"] -= 1
    df["high_open_ratio"] -= 1
    df["low_open_ratio"] -= 1
    df["close_open_ratio"] -= 1
    df["volume_ratio"] -= 1
    df["ma_15_7_ratio"] -= 1
    df["ma_30_7_ratio"] -= 1
    df["ma_30_15_ratio"] -= 1

    df = df.dropna(axis=0) # 결측치 행 제거
    df = df.drop(['open', 'high', 'low', 'close', 'volume', 'value', 'ma_7', 'ma_15', 'ma_30'], axis=1)

    df.to_csv("./data/KRW_BTC_traing.csv", sep=",")


if __name__ == "__main__":
    preprocessing_ohlcv()