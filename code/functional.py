import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

from stable_baselines3 import A2C,PPO,DQN
from stable_baselines3.common.torch_layers import FlattenExtractor

import gymnasium as gym
from trading_environment import trading_env
import tensorflow as tf

from torch import nn
plt.rc('figure',titleweight='bold',titlesize='large',figsize=(15,6))
plt.rc('axes',labelweight='bold',labelsize='large',titleweight='bold',titlesize='large',grid=True)


def max_drawdown(portforlio_history):
    portforlio_history=pd.Series(portforlio_history)
    running_max=portforlio_history.cummax()
    drawdown=(running_max-portforlio_history)/running_max
    drawdown=drawdown.max()
    return drawdown*100

def PnL(portforlio_history):
    portforlio_history=pd.Series(portforlio_history)
    start_portforlio=portforlio_history.iloc[0]
    end_portforlio=portforlio_history.iloc[-1]
    return end_portforlio-start_portforlio

def ROI(portforlio_history):
    portforlio_history=pd.Series(portforlio_history)
    start_portforlio=portforlio_history.iloc[0]
    end_portforlio=portforlio_history.iloc[-1]
    return (end_portforlio-start_portforlio)*100/start_portforlio

class agent():
    def __init__(self,model='DQN',policy='MlpPolicy',total_timesteps=10000,**kwargs):
        self.model_choice={'DQN':DQN,'A2C':A2C,'PPO':PPO}
        self.model_name=model
        self.total_timesteps=total_timesteps
        
        with open(os.path.join(Path(__file__).parent,'parameters.json'),'r+') as f:
                parameter=json.loads(f.read())
                self.start_date=parameter['start_date']
                self.test_date=parameter['test_date']
                self.end_date=parameter['end_date']
                self.window_size=parameter['window_size']
                self.symbol=parameter['symbol']
        self.df=pd.read_csv(f'data/{self.symbol}/{self.symbol}_1d.csv',index_col=0)
        self.df_train=self.df.loc[self.start_date:self.test_date]
        self.df_test=self.df.loc[self.test_date:self.end_date]

        gym.register(id='trading/trading_env',entry_point=trading_env)
        self.env_train=gym.make('trading/trading_env',df=self.df_train,window_size=self.window_size)
        self.env_test=gym.make('trading/trading_env',df=self.df_test,window_size=self.window_size)
        
        self.policy=policy
        if type(self.policy)!=str:
            print('Using custom/other policy !!')
            self.model=self.model_choice[model](policy,self.env_train,policy_kwargs=kwargs,verbose=1)
        else:
            print('Using normal policy !!')
            self.model=self.model_choice[model](policy,self.env_train,verbose=1)
    
    def learn(self):
        self.model.learn(total_timesteps=self.total_timesteps)
    
    def evaluate(self):

        self.portforlio_history=[]
        self.action_list=[]

        current_state,info=self.env_test.reset()
        while True:
            action,state=self.model.predict(current_state)
            action=int(action)
            current_state,reward,terminated,truncated,info=self.env_test.step(action)

            self.portforlio_history.append(self.env_test.total)
            self.action_list.append(action)

            if terminated:
                break
        drawdown=max_drawdown(self.portforlio_history)
        pnl=PnL(self.portforlio_history)
        roi=ROI(self.portforlio_history)
        return pd.Series([drawdown,pnl,roi],index=['Max drawdown','PnL','ROI'])
    
    def visualize(self):
    
        fg=plt.figure()
        ax=fg.add_subplot()
        self.df_test['Close'].plot(ax=ax)
        for i in range(len(self.action_list)):
            if self.action_list[i]==0:
                ax.text(i,self.df_test.iloc[i,0],'B',color='C2')
            elif self.action_list[i]==2:
                ax.text(i,self.df_test.iloc[i,0],'S',color='C3')
        ax.set_title(f'{self.model_name}-{self.symbol}')

if __name__=='__main__':
    m=[10,11,12,13,14,15,14]
    print(ROI(m))