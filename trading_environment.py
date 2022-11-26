import pandas as pd

class Trading_Env:

    def __init__(self, ohlcvfile_path='', nninputfile_path=''):

        self.df_ohlcv = pd.read_csv(ohlcvfile_path) # ohlcv 데이터
        self.df_nninput = pd.read_csv(nninputfile_path) # 딥러닝 입력 데이터

        # 누락된 날짜 컬럼 이름 지정하기
        self.df_ohlcv.rename( columns={'Unnamed: 0':'date'}, inplace=True)
        self.df_nninput.rename(columns={'Unnamed: 0': 'date'}, inplace=True)

        self.data_idx = 0

    def reset(self, balance=1000000):
        # 현재 데이터 인덱스
        self.data_idx = 0

    def get_price(self):
        return self.df_ohlcv.loc[self.data_idx].close

    def get_date(self):
        return self.df_ohlcv.loc[self.data_idx].date

    def get_data_idx(self):
        return self.data_idx

    def step_forward(self):
        self.data_idx += 1

    def get_state(self):
        return self.df_nninput.loc[self.data_idx].drop('date').to_list()

    def get_data_size(self):
        return self.df_nninput.shape[0]



if __name__ == "__main__":
    a = 0

    env = Trading_Env(ohlcvfile_path="./data/KRW_BTC.csv", nninputfile_path="./data/KRW_BTC_traing.csv")
    state = env.get_price()
    print(state)