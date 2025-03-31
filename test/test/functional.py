import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C,PPO

import gymnasium as gym
from environment import trading_env
import tensorflow as tf
import random
plt.rc('figure',titleweight='bold',titlesize='large',figsize=(15,6))
plt.rc('axes',labelweight='bold',labelsize='large',titleweight='bold',titlesize='large',grid=True)


def visualize(df,action_list,title):
    fg=plt.figure()
    ax=fg.add_subplot()
    df['Close'].plot(ax=ax)
    for i in range(len(action_list)):
        if action_list[i]==0:
            ax.text(i,df.iloc[i,0],'B',color='C2')
        elif action_list[i]==2:
            ax.text(i,df.iloc[i,0],'S',color='C3')
    ax.set_title(title)

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

def evaluate(portforlio_history):
    drawdown=max_drawdown(portforlio_history)
    pnl=PnL(portforlio_history)
    roi=ROI(portforlio_history)
    return pd.Series([drawdown,pnl,roi],index=['Max drawdown','PnL','ROI'])

if __name__=='__main__':
    m=[10,11,12,13,14,15,14]
    print(ROI(m))