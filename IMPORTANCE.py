from random import randint
import gym
import numpy as np 
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
from baselines.deepq.replay_buffer import ReplayBuffer
from collections import Counter
from network import CoreNet
from PI import PI
import random
random.seed(101)
np.random.seed(101)



class IMPORTANCE:
    def __init__(self,env,nhead = 5,epoch = 1000,vt = 0.11,missing_rate = 0, limit = 800):
        self.env = env
        self.limit = limit
        self.advice_count = 0
        self.missing = missing_rate
        self.vt = vt
        self.epoch = epoch
        self.nhead = nhead
        self.agents = [CoreNet(observation_space_size = env.observation_space.n) for i in range(nhead)]
        self.optimizer = [optim.Adam(params=self.agents[i].parameters()) for i in range(nhead)]
        self.success = []
        self.replay_buffer = ReplayBuffer(200)
        self.replay_buffer1 = []
        self.random_state = np.random.RandomState(101)
        pi = PI(env)
        self.reference = pi.policy_iteration()
        self.minQPI,self.maxQPI = pi.get_QPI()
        self.var_history = []
        self.tmp_var_histroy = []
        self.histroy = []
        self.advices = []
        self.nadvice = 0
        
    def learn(self):
        for s,a,r,s1,d in self.replay_buffer1:
            # mask = self.random_state.binomial(1, 0.9, self.nhead)
            for k in range(self.nhead):
                target_q = r + 0.99 * torch.max(self.agents[k](s1).detach()) # detach from the computing flow
                loss = F.smooth_l1_loss(self.agents[k](s)[a], target_q)
                self.optimizer[k].zero_grad()
                loss.backward()
                self.optimizer[k].step()

    def train(self):
        t = 0
        for i in range(self.epoch):
            s = self.env.reset()
            for _ in range(50):
                
                # perform chosen action
                a = self.choose_action(s,head = None)
                s1, r, d, _ = self.env.step(a)
                
                if d == True and r < 1:
                    r = -1
                
                self.replay_buffer1.append((s,a,r,s1,d))
               
                self.learn()
                self.replay_buffer1.clear()
                
                s = s1
              
                t += 1
                
                if d == True: break
            
            if d == True and r > 0:
                self.success.append(1)
            else:
                self.success.append(0)

            if i % 20 == 0:
                self.histroy.append( (i,self.eval()) )
                self.advices.append((i,self.nadvice))

        print("last 100 epoches success rate: " + str(sum(self.success[-100:])) + "%")
    
    def var(self,s):
            return self.maxQPI[s] - self.minQPI[s] > self.vt
      
                

    def choose_action(self, s, head = None, eval = False):
        
        if self.var(s)  and (np.random.rand(1) > self.missing) and self.advice_count < self.limit:
            self.advice_count += 1
            self.nadvice += 1
            return self.reference[s]

        if (np.random.rand(1) < 0.1) and not eval: 
            return self.env.action_space.sample()
        else:
            if head is not None:
                with torch.no_grad():
                    vals = self.agents[head](s)
                    action = torch.argmax(vals).item()
                    # print(vals)
                    return  action
            else:
                # vote
                with torch.no_grad():
                    vs = [0,0,0,0]
                    for h in range(self.nhead):
                        v = self.agents[h](s)
                        for i in range(4):
                            vs[i] += float(v[i])
                    return np.argmax(vs)
                    acts = [torch.argmax(self.agents[h](s)).item() for h in range(self.nhead)]
                    data = Counter(acts)
                    action = data.most_common(1)[0][0]
                    return action

    def eval(self):
        eval_rewards = []
        for _ in range(10):
            step_number = 0
            state = self.env.reset()
            terminal = False
            r = 0
            while not terminal and step_number < 200:
                action = self.choose_action(state, head = None, eval = True)
                next_state, reward, terminal, _ = self.env.step(action)
                state = next_state
                step_number += 1
                reward = -1
                if terminal == True:
                    if reward < 1:
                        reward = -20
                    else:
                        reward = 20
                r += reward
                

            # env.render()
            eval_rewards.append(r)

        return np.mean(eval_rewards)
