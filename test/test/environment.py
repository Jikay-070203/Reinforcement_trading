import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
import gymnasium as gym
plt.rc('figure',titleweight='bold',titlesize='large',figsize=(15,6))
plt.rc('axes',labelweight='bold',labelsize='large',titleweight='bold',titlesize='large',grid=True)

class trading_env(gym.Env):
    def __init__(self,df,window_size=1,initial_balance=10**2,percent=0.05):
        super().__init__()

        self.window_size=window_size
        self.df=df
        self.df['Diff_pct']=self.df['Close'].pct_change(1).fillna(0)*100
        self.processed_df=self.df[['Close','Diff_pct']]

        self.index=self.window_size
        
        self.observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size,self.processed_df.shape[1]),dtype=float)
        self.action_space=gym.spaces.Discrete(3)

        self.observation_max=self.processed_df.max().values
        self.observation_min=self.processed_df.min().values
        
        self.initial_balance=initial_balance

        self.usd=self.initial_balance
        self.coin=0

        self.total=self.initial_balance
        self.prev=0

        self.buy_price=self.df.iloc[self.index,0]
        self.sell_price=self.df.iloc[self.index,0]

    def step(self,action):
        self.total=self.processed_df.iloc[self.index,0]*self.coin+self.usd
        reward=0
        new_state=self.processed_df.iloc[self.index-self.window_size+1:self.index+1].values
        terminate,truncate=False,False

        if self.index==self.df.shape[0]-1:
            terminate,truncate=True,True
        if action==2:
            if self.coin>0:
                reward=self.df.iloc[self.index,0]-self.buy_price
                self.sell_price=self.df.iloc[self.index,0]

                self.usd=self.coin*self.df.iloc[self.index,0]
                self.coin=0
        elif action==1:
            if self.coin>0:
                reward=(self.df.iloc[self.index,0]-self.buy_price)*0.1
            else:
                reward=(self.sell_price-self.df.iloc[self.index,0])*0.1
        elif action==0:
            if self.usd>0:
                reward=self.sell_price-self.df.iloc[self.index,0]
                self.buy_price=self.df.iloc[self.index,0]

                self.coin=self.usd/self.df.iloc[self.index,0]
                self.usd=0

        self.prev=self.total
        self.index+=1
        return new_state,reward,terminate,truncate,{}

    def reset(self,seed=None,option=None):
        self.index=self.window_size
        self.total=self.initial_balance
        self.usd=self.initial_balance
        self.coin=0
        self.prev=0
        self.buy_price=self.df.iloc[self.index,0]
        self.sell_price=self.df.iloc[self.index,0]

        return self.processed_df.iloc[self.index-self.window_size:self.index].values,{}

if __name__=='__main__':
    df=pd.read_csv(os.path.join(Path(__file__).parent,'data/SPX/SPX_1d.csv'),index_col=0).iloc[-1500:]
    env=trading_env(df=df,window_size=5)
    print(env.observation_space.shape)
