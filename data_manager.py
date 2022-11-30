import pyupbit

def save_ohlcv(data_cnt):
    df = pyupbit.get_ohlcv("KRW-BTC", count=data_cnt)
    df.to_csv("./data/KRW_BTC.csv", sep=",")
    return df

def preprocessing_ohlcv():
    df = save_ohlcv(365)
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

    df["open_r"] = df.open / df.open.shift(1)
    df["high_r"] = df.high / df.high.shift(1)
    df["low_r"] = df.low / df.low.shift(1)
    df["close_r"] = df.close / df.close.shift(1)
    df["volume_r"] = df.volume / df.volume.shift(1)
    df["ma_7"] = df.close.rolling(window=7).mean()
    df["ma_15"] = df.close.rolling(window=15).mean()
    df["ma_30"] = df.close.rolling(window=30).mean()

    df["ma_7_r"] = df.ma_7 / df.close
    df["ma_15_r"] = df.ma_15 / df.close
    df["ma_30_r"] = df.ma_30 / df.close

    # 데이터 범위를 -1 ~ 1로 조정
    # df["open_r"] *= 10
    # df["high_r"] *= 10
    # df["low_r"] *= 10
    # df["close_r"] *= 10
    # df["volume_r"] *= 10
    # df["ma_7_r"] *= 10
    # df["ma_15_r"] *= 10
    # df["ma_30_r"] *= 10


    df = df.dropna(axis=0) # 결측치 행 제거



    #df_nnInput = df.drop(['value'], axis=1)
    df_nnInput = df.drop(['open', 'high', 'low', 'close', 'volume', 'value', 'ma_7', 'ma_15', 'ma_30'], axis=1)

    #df_nnInput = (df_nnInput - df_nnInput.mean()) / df_nnInput.std()

    df_ohlcv = df[['open', 'high', 'low', 'close', 'volume']]

    df_nnInput.to_csv("./data/KRW_BTC_traing.csv", sep=",")
    df_ohlcv.to_csv("./data/KRW_BTC.csv", sep=",")


if __name__ == "__main__":
    preprocessing_ohlcv()