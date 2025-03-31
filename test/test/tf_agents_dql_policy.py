# from functional import *
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec,ArraySpec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy
from tf_agents.networks import sequential

import tensorflow as tf
import numpy as np
import pandas as pd


class trading_env(py_environment.PyEnvironment):

    def __init__(self,df,window_size=1,initial_balance=10**2,percent=0.05):
        
        self.window_size=window_size
        self.df=df
        self.df['Diff_pct']=self.df['Close'].pct_change(1).fillna(0)*100
        self.processed_df=self.df[['Close','Diff_pct']]

        self.index=self.window_size

        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        # self._observation_spec = array_spec.BoundedArraySpec(shape=(self.window_size,self.processed_df.shape[1]), dtype=np.float32, minimum=-10000.0,maximum=10000.0, name='observation')
        # self._observation_spec = array_spec.BoundedArraySpec(shape=(2,), dtype=np.float32, minimum=-100000.0,maximum=100000.0, name='observation')
        self._observation_spec = ArraySpec(shape=(1,2), dtype=np.float32, name='observation')

        self.observation_max=self.processed_df.max().values
        self.observation_min=self.processed_df.min().values
        
        self.initial_balance=initial_balance

        self.usd=self.initial_balance
        self.coin=0

        self.total=self.initial_balance
        self.prev=0

        self.buy_price=self.df.iloc[self.index,0]
        self.sell_price=self.df.iloc[self.index,0]

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.index=self.window_size
        self.total=self.initial_balance
        self.usd=self.initial_balance
        self.coin=0
        self.prev=0
        self.buy_price=self.df.iloc[self.index,0]
        self.sell_price=self.df.iloc[self.index,0]

        return ts.restart(observation=self.processed_df.iloc[self.index-self.window_size:self.index].values)

    def _step(self, action):

        self.total=self.processed_df.iloc[self.index,0]*self.coin+self.usd
        reward=0
        new_state=self.processed_df.iloc[self.index-self.window_size+1:self.index+1].values
        terminate,truncate=False,False

        if self.index==self.df.shape[0]-1:
            return ts.termination(observation=new_state,reward=reward)
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
        return ts.transition(observation=new_state,reward=reward)

df=pd.read_csv('/home/golderalex2/Downloads/statistic/python/reinforce_trading/test/SPX_1d.csv',index_col=0)
train_env=tf_py_environment.TFPyEnvironment(trading_env(df,window_size=1))
eval_env=tf_py_environment.TFPyEnvironment(trading_env(df,window_size=1))

num_iterations = 20000

initial_collect_steps =100
collect_steps_per_iteration=1
replay_buffer_max_length=100000
batch_size=64 
learning_rate=1e-3
log_interval=200
num_eval_episodes=10
eval_interval=1000


fc_layer_params = (100, 50)
num_actions = train_env.action_spec().maximum - train_env.action_spec().minimum + 1
def dense_layer(num_units):
    return tf.keras.layers.Dense(num_units,activation=tf.keras.activations.relu,kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal'))
dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
lstm_layers=tf.keras.layers.LSTM(3)
q_values_layer = tf.keras.layers.Dense(num_actions,activation=None,kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),bias_initializer=tf.keras.initializers.Constant(-0.2))
# flatten=tf.keras.layers.Flatten()
q_net = sequential.Sequential([
    lstm_layers,
    q_values_layer
    ])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
train_step_counter = tf.Variable(0)
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)
agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),train_env.action_spec())

def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):
        print(_)

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            # print(episode_return)
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

print(compute_avg_return(eval_env, random_policy, num_eval_episodes))