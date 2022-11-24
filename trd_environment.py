import pandas as pd
from typing import Final

class Trading_Env:

    BUY: Final = 0
    SELL: Final = 1
    HOLD: Final = 2
    ACTION_SIZE: Final = 3
    trading_fee:Final = 0.0005 # 거래 수수료

    def __init__(self, ohlcvfile_path, nninputfile_path, balance=1000000):
        '''
        상태 데이터 정의
         - 트레이딩 기본자료 (KRW_BTC_traing.csv) 데이터
         - 잔액 변화량
         - 코인 변화량
         - 포트폴리오 가치 변화량
        '''

        self.df_ohlcv = pd.read_csv(ohlcvfile_path) # ohlcv 데이터
        self.df_nninput = pd.read_csv(nninputfile_path) # 딥러닝 입력 데이터

        # 잔액
        self.balance = balance
        # 코인수량
        self.coin_quantity = 0
        # 평단가
        self.avg_price = 0
        # 현재 데이터 인덱스
        self.cur_data_idx = 0
        # 거래 기록
        self.trading_record = [] # 날짜, 데이터 인덱스, 행동, 매도/매수 물량, 코인가격, 거래수수료

    def get_state(self, action):

        sr_ohlcv = self.df_ohlcv.iloc[self.cur_data_idx]

        if action == self.BUY:
            pass
        elif action == self.SELL:
            pass
        else: # HOLD
            pass

    def trading(self, action, confidence): # confidence: 행동에 대한 확신 값

        sr_ohlcv = self.df_ohlcv.iloc[self.cur_data_idx]
        price = sr_ohlcv.close
        volume = 0

        if action == self.BUY:
            maxVolume = self.balance / (price*(1+self.trading_fee)) # 거래 수수료 계산
            buyVolume = maxVolume * confidence

            self.coin_quantity += buyVolume

        elif action == self.SELL:
            pass
        else: # HOLD
            pass

        pass

