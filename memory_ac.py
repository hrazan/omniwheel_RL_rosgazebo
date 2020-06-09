import random
import numpy as np
import pandas as pd

class Memory:
    """
    This class provides an abstraction to store the [s, a, r, a'] elements of each iteration.
    Instead of using tuples (as other implementations do), the information is stored in lists 
    that get returned as another list of dictionaries with each key corresponding to either 
    'cur_state', 'action', 'reward', 'next_state', 'done' or "isFinal".
    """
    def __init__(self, max_size, load=False, memories=None):
        self.max_size = max_size
        #self.exp = pd.DataFrame(columns = ['cur_state', 'action', 'reward', 'next_state', 'done'], dtype = np.float32)
        if load == False:
            self.exp = pd.DataFrame(columns = ['cur_state', 'action', 'reward', 'next_state', 'done'], dtype = np.float32)
        else:
            self.exp = memories
            self.exp = self.exp.reset_index()
            self.exp = self.exp.drop(columns=['index'])
            
            # String to array of float
            for x in range(len(self.exp.index)):
                cur_state_raw = self.exp.at[x,'cur_state'][2:-1].split(' ')
                cur_state = []
                for y in range(len(cur_state_raw)):
                    if cur_state_raw[y] != '':
                        try:
                            cur_state.append(float(cur_state_raw[y]))
                        except:
                            cur_state.append((0.1**int(cur_state_raw[y][-1]))*float(cur_state_raw[y][0:-5]))
                self.exp.at[x,'cur_state'] = np.asarray(tuple(cur_state))
                
                next_state_raw = self.exp.at[x,'next_state'][2:-1].split(' ')
                next_state = []
                for y in range(len(next_state_raw)):
                    if next_state_raw[y] != '':
                        try:
                            next_state.append(float(next_state_raw[y]))
                        except:
                            next_state.append((0.1**int(next_state_raw[y][-1]))*float(next_state_raw[y][0:-5]))
                self.exp.at[x,'next_state'] = np.asarray(tuple(next_state))
                
                action_raw = self.exp.at[x,'action'][1:-2].split(' ')
                action = []
                for y in range(len(action_raw)):
                    if action_raw[y] != '':
                        try:
                            action.append(float(action_raw[y]))
                        except:
                            action.append((0.1**int(action_raw[y][-1]))*float(action_raw[y][0:-5]))
                self.exp.at[x,'action'] = np.array(action, dtype=np.float32)
                
            #print self.exp.at[0,'cur_state'], self.exp.at[0,'next_state'], self.exp.at[0,'action']

    def getMiniBatch(self, size, mode) :
        if mode=='positive':
            positive_memories = self.exp.nlargest(1000, 'reward', keep='last')
            memories_1 = positive_memories.sample(n=int(size/2))
            memories_2 = self.exp.sample(n=(size - int(size/2)))
            memories = pd.concat([memories_1,memories_2])
        elif mode=='pos_neg':
            positive_memories = self.exp.nlargest(1000, 'reward', keep='last')
            negative_memories = self.exp.nsmallest(1000, 'reward', keep='last')
            memories_1 = positive_memories.sample(n=int(size/3))
            memories_2 = negative_memories.sample(n=int(size/3))
            memories_3 = pd.concat([memories_1,memories_2])
            memories_4 = self.exp.sample(n=(size - (2*int(size/3))))
            memories = pd.concat([memories_3,memories_4])
        elif mode == 'random':
            memories = self.exp.sample(n=size)
        else:
            print "Error! Enter mode: 'positive' or 'random'"
        memories = memories.values.tolist()
        #print memories
        return memories

    def addMemory(self, cur_state, action, reward, next_state, done) :
        # Delete a memory from DataFrame if the size is equal or bigger than the max size
        if len(self.exp.index) >= self.max_size:
            #index = random.randrange(self.max_size)
            self.exp = self.exp.drop([0])
            self.exp = self.exp.reset_index()
            self.exp = self.exp.drop(columns=['index'])
        # Add a new experience to memory DataFrame
        new_memory = pd.DataFrame([[cur_state, action, reward, next_state, done]], columns = ['cur_state', 'action', 'reward', 'next_state', 'done'])
        self.exp = self.exp.append(new_memory, ignore_index=True)   
