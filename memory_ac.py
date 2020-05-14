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
    def __init__(self, max_size):
        self.max_size = max_size
        self.exp = pd.DataFrame(columns = ['cur_state', 'action', 'reward', 'next_state', 'done'], dtype = np.float32)

    def getMiniBatch(self, size, mode) :
        if mode=='positive':
            memories = self.exp.nlargest(size, 'reward') #,keep='last')
        elif mode == 'random':
            memories = self.exp.sample(n=size)
        else:
            print "Error! Enter mode: 'positive' or 'random'"
        memories = memories.values.tolist()
        return 

    def addMemory(self, cur_state, action, reward, next_state, done) :
        # Delete a memory from DataFrame if the size is equal or bigger than the max size
        if len(self.exp.index) >= self.max_size:
            index = random.randrange(self.max_size)
            self.exp.drop([index])
        # Add a new experience to memory DataFrame
        new_memory = pd.DataFrame([[cur_state, action, reward, next_state, done]], columns = ['cur_state', 'action', 'reward', 'next_state', 'done'])
        self.exp.append(new_memory, ignore_index=True)   
