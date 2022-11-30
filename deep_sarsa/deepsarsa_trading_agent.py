from dataclasses import dataclass

import pandas as pd

from trading_environment import Trading_Env
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import matplotlib.pyplot as plt

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

#날짜, 데이터 인덱스, 행동, 매도/매수 물량, 코인가격, 거래수수료

@dataclass
class Agent_log:
    date:str = ''
    balance: float=0.
    coin_quantity: float=0.
    avg_price: float=0.
    profitloss: float=0.
    portfolio_val: float = 0.
    action: int = -1
    trading_volume: float = 0.
    price: int = 0
    trading_price: float=0.
    trading_fee: float = 0.



class DNN_Sarsa(nn.Module):
    def __init__(self, state_size, action_size):
        super(DNN_Sarsa, self).__init__()

        self.fc1 = nn.Linear(state_size, 50)
        self.fc2 = nn.Linear(50, 50)
        self.out = nn.Linear(50, action_size)
        #self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        # out = F.relu(self.out(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        #x = F.relu(self.fc3(x))
        out = self.out(x)
        #out = F.softmax(x, dim=1)

        return out


class Trading_Agent:
    BUY: Final = 0
    SELL: Final = 1
    HOLD: Final = 2
    ACTION_SIZE: Final = 3
    trading_fee:Final = 0.0005 # 거래 수수료
    STATE_SIZE = 10

    '''
    상태 데이터 정의
     - 트레이딩 기본자료 (KRW_BTC_traing.csv) 데이터
     - 잔액 변화량
     - 코인 변화량
     - 포트폴리오 가치 변화량
    '''

    def __init__(self,balance=1000000):
        # 거래 환경 객체
        self.env = Trading_Env("../data/KRW_BTC.csv", "../data/KRW_BTC_traing.csv")
        # 잔액
        self.init_balance = balance
        self.balance = balance
        # 코인수량
        self.coin_quantity = 0.
        # 평단가
        self.avg_price = 0.
        # 손익률
        self.profitloss = 0.


        self.hold_ratio = 0.
        self.portfolio_val = 0.

        self.model = DNN_Sarsa(self.STATE_SIZE, self.ACTION_SIZE).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.epoch_cnt = 300
        self.discount_factor = 0.9
        self.epsilon = 0.5
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.05

        # 로그
        self.list_agent_log = []  #
        self.agent_log_add = self.list_agent_log.append



    # 평단가 갱신
    def update_avgprice(self, buy_vol, price):
        self.avg_price = ((self.avg_price * self.coin_quantity) + (price * buy_vol))/(self.coin_quantity+buy_vol)

    def save_agent_log(self, date, balance, coin_quantity, avg_price, profitloss, portfolio_val, action,
                       trdVolume, price):
        rec = Agent_log(date=date, balance=balance, coin_quantity=coin_quantity,
                        avg_price=avg_price, profitloss=profitloss, portfolio_val=portfolio_val, action=action,
                        trading_volume=trdVolume, price=price, trading_price=trdVolume*(self.trading_fee+1)*price,
                        trading_fee=trdVolume*(self.trading_fee)*price)
        self.agent_log_add(rec)

    def reset(self, balance=1000000):
        self.list_agent_log.clear()
        self.balance = balance
        self.coin_quantity = 0
        self.avg_price = 0
        self.profitloss = 0.
        self.portfolio_val = 0.
        self.env.reset()

    def trading(self, action, confidence):  # confidence: 행동에 대한 확신 값

        date = self.env.get_date()
        price = self.env.get_price()
        trdVolume = 0

        trading_condi = True
        if action == self.BUY:
            if self.balance > 10000:
                maxVolume = self.balance / (price * (1 + self.trading_fee))  # 거래 수수료 계산
                trdVolume = maxVolume * confidence

                if trdVolume * price < 10000: # 주문금엑이 최소주문 금액보다 적다면
                    trdVolume = 10000 / price
                    if trdVolume * price * (1 + self.trading_fee) > self.balance:
                        trading_condi = False
            else:
                trading_condi = False

            if trading_condi == True:
                self.update_avgprice(trdVolume, price)  # 평단가 갱신
                self.coin_quantity += trdVolume
                self.balance -= (trdVolume * price * (1 + self.trading_fee))
            else:
                action = self.HOLD
                trdVolume = 0

        elif action == self.SELL:
            if self.coin_quantity > 0.0000001:
                trdVolume = self.coin_quantity * confidence

                if trdVolume * price < 10000:  # 판매 금액이 최소 주문금액 보다 적다면
                    trading_condi = False
            else: # 보유코인이 없다면
                trading_condi = False

            if trading_condi:
                self.coin_quantity -= trdVolume
                self.balance += (trdVolume * price * (1 - self.trading_fee))
            else:
                action = self.HOLD
                trdVolume = 0

        else:  # HOLD
            pass

        self.portfolio_val = self.coin_quantity * price + self.balance

        self.profitloss = self.portfolio_val / self.init_balance -1

        self.save_agent_log(date, self.balance, self.coin_quantity, self.avg_price,
                            self.profitloss, self.portfolio_val, action, trdVolume, price)

        self.env.step_forward()
        #print(f"data_index: {self.env.get_data_idx()}")

        # 보상 계산
        next_price = self.env.get_price()
        next_portfolio_val = self.coin_quantity * next_price + self.balance
        reward = (next_portfolio_val / self.portfolio_val) - 1  # 데이터 범위(-1 ~ 1)

        self.hold_ratio = (self.coin_quantity*self.avg_price)/self.portfolio_val

        #print(f"수익률: {self.profitloss*100:.3f}")

        return reward, self.get_state()

    def learn(self, state, action, reward, next_state):

        self.model.train()
        self.optimizer.zero_grad()
        state = torch.Tensor(state).unsqueeze(0).to(device)
        next_state = torch.Tensor(next_state).unsqueeze(0).to(device)

        predict = self.model(state)[0]
        predict_val = predict[action]
        next_q = self.model(next_state)
        next_q_val = torch.max(next_q)
        target = reward + self.discount_factor * next_q_val

        # print("predict")
        # print(predict)
        # print("next_q")
        # print(next_q)

        #loss = self.criterion(target, predict)
        loss = torch.mean(torch.square(target - predict_val))
        loss.backward()
        self.optimizer.step()


    def softmax(self, x):
        max = np.max(x)
        x = x-max
        sum = np.sum(np.exp(x))
        y = np.exp(x)/sum
        return y

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            confidence = np.min([np.random.rand(), 0.5]) # 탐험의 경우 확신도는 0.5 이하의 램덤수
            return np.random.choice(self.ACTION_SIZE), confidence # or return np.random.randrange(self.ACTION_SIZE)
        state = torch.Tensor(state).unsqueeze(0).to(device)
        q_values = self.model(state)[0].detach().numpy()
        confidence = self.softmax(q_values)
        return np.argmax(q_values), max(confidence)

    def get_state(self):
        state =self.env.get_state()
        state.extend([self.hold_ratio, self.profitloss])
        return state

    def run(self, training=True):
        list_profitloss = []
        max_portfolio_val = 0
        for ep in range(self.epoch_cnt):
            print(f"\n================= Epoch: {ep+1} =================")
            self.reset()
            state = self.get_state()
            action, confidence = self.get_action(state)



            for idx in range(self.env.get_data_size()-1):

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                reward, next_state = self.trading(action, confidence)
                next_action, confidence = self.get_action(next_state)

                self.learn(state, action, reward, next_state)

                state = next_state
                action = next_action

            print(f"포트폴리오 가치: {self.portfolio_val:.0f}")
            list_profitloss.append(self.profitloss)

            if self.portfolio_val > max_portfolio_val:
                self.save_record_to_csv()
                torch.save(self.model.state_dict(), f"../model/deep_sarsa.pth")
                print(f"모델 저장!")
                max_portfolio_val = self.portfolio_val

        plt.plot(list_profitloss)
        plt.show()


    def save_record_to_csv(self):
        df = pd.DataFrame(self.list_agent_log)
        df.to_csv(f"../log/deep_sarsa.csv")

if __name__ == "__main__":
    # x = torch.FloatTensor(range(15))  # 입력 값
    # x0 = torch.FloatTensor(range(15)).unsqueeze(0)  # 입력 값
    # x1 = torch.FloatTensor(range(15)).unsqueeze(1)  # 입력 값

    agent = Trading_Agent()
    agent.run()