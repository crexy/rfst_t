import pandas as pd
from typing import Final

class Trading_Env:

    BUY: Final = 0
    SELL: Final = 1
    HOLD: Final = 2
    ACTION_SIZE: Final = 3

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

    def get_state(self, action):

        sr_ohlcv = self.df_ohlcv.iloc[self.cur_data_idx]

        if action == self.BUY:
            pass
        elif action == self.SELL:
            pass
        else:
            pass


    def trading(self, action):
        pass


