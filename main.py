from IMPORTANCE import IMPORTANCE
from RCMP import RCMP
from RANDOM import RANDOM
from NOADV import NOADV
from random import randint
import gym
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = {'Epoch':[],
        'Reward':[],
        'Algorithm':[]}

data1 = {'Epoch':[],
        'Advice':[],
        'Algorithm':[]}
data2 = {'Epoch':[],
        'Uncertainty':[],
        'Algorithm':[]}
data3 = {'Epoch':[],
        'eps':[],
        'Algorithm':[]}

epoh = 800
mr = 0
nhead = 2

env = gym.make('FrozenLake8x8-v0', is_slippery=False)
env.reset()

# for i in [2,5,10]:
#     a = RCMP(env,nhead= i,epoch = epoh,vt = 1,missing_rate = mr, limit = 800)
#     a.train()

#     data2['Epoch'].extend(list(np.asarray(a.var_history).T[0]))
#     data2['Uncertainty'].extend(list(np.asarray(a.var_history).T[1]))
#     data2['Algorithm'].extend( [f'head: {i}'] * len(a.var_history) )

for _ in range(10):
    rcmp = RCMP(env,nhead= nhead,epoch = epoh,vt = 2,missing_rate = mr, limit = 800)
    rand = RANDOM(env,nhead= nhead,epoch = epoh,vt = 2,missing_rate = mr, limit = 800)
    noadv = NOADV(env,nhead= nhead,epoch = epoh,vt = 2,missing_rate = mr, limit = 800)
    impt = IMPORTANCE(env,nhead= nhead,epoch = epoh,vt = 0.5,missing_rate = mr, limit = 800)
    algo = [rcmp,rand,noadv,impt]
    for a in algo:
        a.train()
        data['Epoch'].extend(list(np.asarray(a.histroy).T[0]))
        data['Reward'].extend(list(np.asarray(a.histroy).T[1]))
        data['Algorithm'].extend( [str(type(a).__name__)] * len(a.histroy) )

        data1['Epoch'].extend(list(np.asarray(a.advices).T[0]))
        data1['Advice'].extend(list(np.asarray(a.advices).T[1]))
        data1['Algorithm'].extend( [str(type(a).__name__)] * len(a.advices) )
# for _ in range(10):
#     for i in [0.1,0.3,0.5,1,1.5,2]:
#         a = RCMP(env,nhead= nhead,epoch = epoh,vt = i,missing_rate = mr, limit = 800)
#         a.train()

#         data['Epoch'].extend(list(np.asarray(a.histroy).T[0]))
#         data['Reward'].extend(list(np.asarray(a.histroy).T[1]))
#         data['Algorithm'].extend( [str(type(a).__name__) + ' eps ' + str(i)] * len(a.histroy) )

#         data1['Epoch'].extend(list(np.asarray(a.advices).T[0]))
#         data1['Advice'].extend(list(np.asarray(a.advices).T[1]))
#         data1['Algorithm'].extend( [str(type(a).__name__) + ' eps ' + str(i)] * len(a.advices) )



    


  
# Create DataFrame
df = pd.DataFrame(data)
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
sns.lineplot(data = df,x = 'Epoch', y = 'Reward', hue = 'Algorithm')
plt.show()
plt.clf() 
sns.lineplot(data = df1,x = 'Epoch', y = 'Advice', hue = 'Algorithm')
plt.show()
# plt.clf() 
# sns.lineplot(data = df2,x = 'Epoch', y = 'Uncertainty', hue = 'Algorithm')
# plt.show()


