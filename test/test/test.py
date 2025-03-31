import pandas as pd
import numpy as np
from time import sleep

learning_rate=0.1
discount_value=0.9
epochs=2000
show_each=200

def step(action,current_state):
    if current_state[0]==3 and action==2:
        return None,1,False,True
    elif current_state[0]==0 and action==0:
        return None,-1,True,False
    else:
        if action==2:
            return (current_state[0]+1,),-1,False,False
        elif action==1:
            return (current_state[0],),-1,False,False
        else:
            return (current_state[0]-1,),-1,False,False

q_table=np.array([[2,1,0] for _ in range(4)],dtype=float)
for epoch in range(epochs):
    print(f'Epoch :{epoch}')
    current_state=(2,)
    while True:
        action=np.argmax(q_table[current_state])
        new_state,reward,end,win=step(action,current_state)

        current_q_value=q_table[current_state+(action,)]
        if new_state is None:
            new_q_value=(1-learning_rate)*current_q_value+learning_rate*(reward)
        else:
            new_q_value=(1-learning_rate)*current_q_value+learning_rate*(reward+discount_value*np.max(q_table[new_state]))
        q_table[current_state + (action,)] = new_q_value
        if win:
            print('Win!!!')
            raise
        if end:
            print('End epoch')
            break

        current_state=new_state

        print(q_table)
        sleep(0.1)