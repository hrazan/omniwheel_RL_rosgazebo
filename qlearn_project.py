import random
import os.path
from os import path
import csv

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        if path.exists('/home/katolab/experiment_data/qtable.csv'):
            with open('/home/katolab/experiment_data/qtable.csv',mode='r') as qtable:
                #self.q = csv.DictReader(qtable)
                reader = csv.reader(qtable, quoting=csv.QUOTE_NONNUMERIC)
                #self.q = {rows[0]:rows[1] for rows in reader}
                for rows in reader:
                #    self.q[rows[0]] = rows[1]
                    self.q[(str(rows[0]),int(rows[1]))] = rows[2]
            print('Qtabel loaded!')
            print(self.q)
        else: 
            #self.q = {}
            print('New qtabel is made!')
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))            
        '''
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q)

        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))]
            maxQ = max(q)

        count = q.count(maxQ)
        # In case there're several state-action max values 
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ]
            i = random.choice(best)
        else:
            i = q.index(maxQ)

        action = self.actions[i]        
        if return_q: # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)
