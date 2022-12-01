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



class SimpleDNN(nn.Module):
    def __init__(self, state_size, action_size):
        super(SimpleDNN, self).__init__()

        self.fc1 = nn.Linear(state_size, 50)
        self.fc2 = nn.Linear(50, 100)
        self.out = nn.Linear(100, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        out = F.softmax(x, dim=1)
        return out


class Trading_Agent:
    BUY: Final = 0
    SELL: Final = 1
    HOLD: Final = 2
    ACTION_SIZE: Final = 3
    trading_fee:Final = 0.0005 # 거래 수수료
    STATE_SIZE = 11
    EPOCH_CNT = 300

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

        self.model = SimpleDNN(self.STATE_SIZE, self.ACTION_SIZE).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.discount_factor = 0.9
        # self.epsilon = 0.5
        # self.epsilon_decay = 0.95
        # self.epsilon_min = 0.05

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

        self.env.step_forward() # 입력 데이터 스텝 +1
        #print(f"data_index: {self.env.get_data_idx()}")

        # 보상 계산
        next_price = self.env.get_price()
        next_portfolio_val = self.coin_quantity * next_price + self.balance
        reward = (next_portfolio_val / self.portfolio_val) - 1  # 데이터 범위(-1 ~ 1)

        self.hold_ratio = (self.coin_quantity*self.avg_price)/self.portfolio_val

        #print(f"수익률: {self.profitloss*100:.3f}")

        return reward

    def discounted_rewards(self, list_reward):
        list_d_reward = [ 0. for x in list_reward]
        Gt = 0
        list_size = len(list_reward)
        for i, reward in enumerate(reversed(list_reward)):
            Gt = reward + self.discount_factor*Gt
            list_d_reward[list_size-1-i] = Gt
        return np.array(list_d_reward)


    def learn(self, list_action, list_reward, list_state):

        discounted_rewards = self.discounted_rewards(list_reward)
        # 정규화
        discounted_rewards -= discounted_rewards.mean()
        discounted_rewards /= discounted_rewards.std()

        self.model.train()
        self.optimizer.zero_grad()
        state = torch.Tensor(list_state).unsqueeze(0).to(device)

        policies = self.model(state)[0]
        one_hot_action = F.one_hot(list_action)

        action_prob = torch.sum(policies*one_hot_action, dim=1)
        cross_entropy_loss = -torch.sum(torch.log(action_prob+1e-5)*discounted_rewards)

        cross_entropy_loss.backward()

        self.optimizer.step()


    def softmax(self, x):
        max = np.max(x)
        x = x-max
        sum = np.sum(np.exp(x))
        y = np.exp(x)/sum
        return y

    def get_action(self, state):
        state = torch.Tensor(state).unsqueeze(0).to(device)
        q_values = self.model(state)[0].detach().numpy()
        action = np.random.choice(self.ACTION_SIZE, p=q_values)
        return action, q_values[action]

    def get_state(self):
        state =self.env.get_state()
        state.extend([self.hold_ratio, self.balance/self.init_balance, self.profitloss])
        return state

    def run(self):
        max_portfolio_val = 0
        list_profitloss = [] # 매 에포크의 손익률 저장 => 그래프 전시

        for ep in range(self.EPOCH_CNT):
            print(f"\n================= Epoch: {ep+1} =================")
            self.reset()

            list_action = []
            list_reward = []
            list_state = []

            for idx in range(self.env.get_data_size()-1):

                state = self.get_state()
                action, confidence = self.get_action(state)
                reward = self.trading(action, confidence)

                list_action.append(action)
                list_reward.append(reward)
                list_state.append(state)

            self.learn(list_action, list_reward, list_state)

            print(f"{self.env.get_date()}) 포트폴리오 가치: {self.portfolio_val:.0f}")
            list_profitloss.append(self.profitloss)

            if self.portfolio_val > max_portfolio_val:
                self.save_record_to_csv()
                torch.save(self.model.state_dict(), f"../model/deep_reinforce.pth")
                print(f"모델 저장!")
                max_portfolio_val = self.portfolio_val

        plt.plot(list_profitloss)
        plt.show()


    def save_record_to_csv(self):
        df = pd.DataFrame(self.list_agent_log)
        df.to_csv(f"../log/deep_reinforce.csv")

if __name__ == "__main__":
    # x = torch.FloatTensor(range(15))  # 입력 값
    # x0 = torch.FloatTensor(range(15)).unsqueeze(0)  # 입력 값
    # x1 = torch.FloatTensor(range(15)).unsqueeze(1)  # 입력 값

    agent = Trading_Agent()
    agent.run()