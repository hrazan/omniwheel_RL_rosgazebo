import random

import numpy as np
from keras import Sequential, optimizers
from keras.layers import Dense, Activation, Dropout
from keras.models import load_model
from keras.optimizers import Adam

import memory


class DeepQ:
    """
    DQN abstraction.

    As a quick reminder:
        traditional Q-learning:
            Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
        DQN:
            target = reward(s,a) + gamma * max(Q(s')

    """
    def __init__(self, inputs, outputs, memorySize, discountFactor, learningRate, learnStart):
        """
        Parameters:
            - inputs: input size
            - outputs: output size
            - memorySize: size of the memory that will store each state
            - discountFactor: the discount factor (gamma)
            - learningRate: learning rate
            - learnStart: steps to happen before for learning. Set to 128
        """
        self.input_size = inputs
        self.output_size = outputs
        self.memory = memory.Memory(memorySize)
        self.discountFactor = discountFactor
        self.learnStart = learnStart
        self.learningRate = learningRate

    def initNetworks(self, hiddenLayers):
        actor_model, critic_model = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)
        self.actor_model = actor_model
        self.critic_model = critic_model

        target_actor_model, target_critic_model = self.createModel(self.input_size, self.output_size, hiddenLayers, "relu", self.learningRate)
        self.target_actor_model = target_actor_model
        self.target_critic_model = target_critic_model

	# ========================================================================= #
	#                              Model Definitions                            #
	# ========================================================================= #

    def createModel(self, inputs, outputs, hiddenLayers, activationType, learningRate):
		adam  = Adam(lr=learningRate)        
        
        # Actor Network Model
        state_input = Input(shape=(inputs, ))
		h1 = Dense(128, activation=activationType)(state_input)
		h2 = Dense(128, activation=activationType)(h1)
		h3 = Dense(128, activation=activationType)(h2)
		output = Dense(outputs, activation=activationType)(h3)
		
		actor_model = Model(inputs=state_input, outputs=output)
		actor_model.compile(loss="mse", optimizer=adam)
		actor_model.summary()
		
		# Critic Network Model
		state_input = Input(shape=self.env.observation_space.shape)
		state_h1 = Dense(128, activation=activationType)(state_input)
		
		action_input = Input(shape=self.env.action_space.shape)
		merged = Concatenate()([state_h1, action_input])
		merged_h1 = Dense(128, activation=activationType)(merged)
		merged_h2 = Dense(128)(merged_h1)
		output = Dense(1, activation=activationType)(merged_h1)
		critic_model  = Model(inputs=[state_input,action_input], outputs=output)
		critic_model.compile(loss="mse", optimizer=adam)
		critic_model.summary()
        
        return actor_model, critic_model

    def printNetwork(self):
        i = 0
        for layer in self.model.layers:
            weights = layer.get_weights()
            print("layer ",i,": ",weights)
            i += 1

    def backupNetwork(self, model, backup):
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1

    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)

    # predict Q values for all the actions
    def getQValues(self, state):
        predicted = self.model.predict(state.reshape(1,len(state)))
        #print(state)
        #print("predicted: ",predicted)
        return predicted[0]

    def getTargetQValues(self, state):
        #predicted = self.targetModel.predict(state.reshape(1,len(state)))
        predicted = self.targetModel.predict(state.reshape(1,len(state)))

        return predicted[0]

    def getMaxQ(self, qValues):
        return np.max(qValues)

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # calculate the target function
    def calculateTarget(self, qValuesNewState, reward, isFinal):
        """
        target = reward(s,a) + gamma * max(Q(s')
        """
        if isFinal:
            return reward
        else :
            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

    # select the action with the highest Q value
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate :
            action = np.random.randint(0, self.output_size)
            #print('random')
        else :
            action = self.getMaxIndex(qValues)
            #print(qValues, action)
        return action

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

    def learnOnLastState(self):
        if self.memory.getCurrentSize() >= 1:
            return self.memory.getMemory(self.memory.getCurrentSize() - 1)

    def learnOnMiniBatch(self, miniBatchSize, useTargetNetwork=True):
        # Do not learn until we've got self.learnStart samples
        if self.memory.getCurrentSize() > self.learnStart:
            # learn in batches of 128
            miniBatch = self.memory.getMiniBatch(miniBatchSize)
            X_batch = np.empty((0,self.input_size), dtype = np.float64)
            Y_batch = np.empty((0,self.output_size), dtype = np.float64)
            for sample in miniBatch:
                isFinal = sample['isFinal']
                state = sample['state']
                action = sample['action']
                reward = sample['reward']
                newState = sample['newState']

                qValues = self.getQValues(state)
                if useTargetNetwork:
                    qValuesNewState = self.getTargetQValues(newState)
                else :
                    qValuesNewState = self.getQValues(newState)
                targetValue = self.calculateTarget(qValuesNewState, reward, isFinal)

                X_batch = np.append(X_batch, np.array([state.copy()]), axis=0)
                Y_sample = qValues.copy()
                Y_sample[action] = targetValue
                Y_batch = np.append(Y_batch, np.array([Y_sample]), axis=0)
                if isFinal:
                    X_batch = np.append(X_batch, np.array([newState.copy()]), axis=0)
                    Y_batch = np.append(Y_batch, np.array([[reward]*self.output_size]), axis=0)
            self.model.fit(X_batch, Y_batch, batch_size = len(miniBatch), epochs=1, verbose = 0)

    def saveModel(self, path):
        self.model.save(path)

    def loadWeights(self, path):
        self.model.set_weights(load_model(path).get_weights())
