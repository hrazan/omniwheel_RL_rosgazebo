import random

import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Add, Input, Concatenate
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow as tf
from collections import deque
import ac_memory as memory

#for xrange
from past.builtins import xrange

class Actor:
    def __init__(self, sess, action_dim, observation_dim, learningRate, hiddenLayer):
        # setting our created session as default session
        K.set_session(sess)
        self.sess = sess
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.learningRate = learningRate
        
        self.state_input, self.model = self.create_model(hiddenLayer)
        _, self.target_model = self.create_model(hiddenLayer)
        
        # Placeholder for critic gradients with respect to action_input.
        self.actor_critic_grads = tf.placeholder(tf.float32, [None, action_dim]) # where we will feed de/dC (from critic)
        
        model_weights = self.model.trainable_weights
        # Calulate and update the weights of the model to optimize the actor
        self.actor_grads = tf.gradients(self.model.output, model_weights, -self.actor_critic_grads)
        grads = zip(self.actor_grads, model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learningRate).apply_gradients(grads)

    def create_model(self, hiddenLayer):
        state_h = hiddenLayer[:]
        
        if len(state_h)<1: error
        state_input = Input(shape=self.observation_dim)
        try:
            state_h[0] = Dense(state_h[0], activation='relu')(state_input)
        except ValueError:
            print ("Error: insert at least one hidden layer of Actor's model")
        if len(state_h)>1:
            for layer in xrange(1, len(state_h), 1):
                state_h[layer] = Dense(state_h[layer], activation='relu')(state_h[layer-1])
        output = Dense(3, activation='tanh')(state_h[-1])
        
        model = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=self.learningRate)
        model.compile(loss="mse", optimizer=adam)
        model.summary()
        return state_input, model

    def train(self, samples, critic, env):
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            
            cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
            action = action.reshape((1, env.action_space.shape[0]))
            new_state = new_state.reshape((1, env.observation_space.shape[0]))
            
            predicted_action = self.model.predict(cur_state)
            grads = self.sess.run(critic.critic_grads, feed_dict={
				critic.state_input:  cur_state,
				critic.action_input: predicted_action
			})[0]

            self.sess.run(self.optimize, feed_dict={
                self.state_input: cur_state,
                self.actor_critic_grads: grads
            })

class Critic:
    def __init__(self, sess, action_dim, observation_dim, learningRate, hiddenLayer):
        # setting our created session as default session
        K.set_session(sess)
        self.sess = sess
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.learningRate = learningRate
        
        self.state_input, self.action_input, self.model = self.create_model(hiddenLayer)
        _, _, self.target_model = self.create_model(hiddenLayer)
        self.critic_grads = tf.gradients(self.model.output, self.action_input) # where we calcaulte de/dC for feeding above
        
        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    def create_model(self, hiddenLayer):
        state_h = hiddenLayer[0][:]
        action_h = hiddenLayer[1][:]
        merge_h = hiddenLayer[2][:]
        
        # Before merging with action
        # State's Hidden Layers
        state_input = Input(shape=self.observation_dim)
        try:
            state_h[0] = Dense(state_h[0], activation='relu')(state_input)
            if len(state_h)>1:
                for layer in xrange(1, len(state_h), 1):
                    state_h[layer] = Dense(state_h[layer], activation='relu')(state_h[layer-1])
            input_merge = Dense(merge_h[0], activation='relu')(state_h[-1])
        except:
            input_merge = Dense(merge_h[0], activation='relu')(state_input)
        
        # Action's Hidden Layers
        action_input = Input(shape=(self.action_dim,))
        try:
            action_h[0] = Dense(merge_h[0], activation='relu')(action_input)
            if len(merge_h)>1:
                for layer in xrange(1, len(action_h), 1):
                    action_h[layer] = Dense(action_h[layer], activation='relu')(action_h[layer-1])
            action_merge = Dense(merge_h[0], activation='relu')(action_h[-1])
        except:
            action_merge = Dense(merge_h[0], activation='relu')(action_input)
        
        # After merging with action        
        # merge_layer = Add()([input_merge, action_merge])
        try:
            merge_h[0] = Add()([input_merge, action_merge])
            if len(merge_h)>1:
                for layer in xrange(1, len(merge_h), 1):
                    merge_h[layer] = Dense(merge_h[layer], activation='relu')(merge_h[layer-1])
        except ValueError:
            print ("Error: insert at least one hidden layer of Critic's model")
        output = Dense(1, activation='linear')(merge_h[-1])
                
        model = Model(inputs=[state_input, action_input], outputs=output)
        model.compile(loss="mse", optimizer=Adam(lr=self.learningRate))
        model.summary()
        return state_input, action_input, model

    def train(self, samples, actor, gamma, batch_size):
        X_states = []
        X_actions = []
        y = []
        for sample in samples:
            #print sample
            cur_state, action, reward, new_state, done = sample
            """
            print "cur_state: ", cur_state
            print "action: ", action
            print "reward: ", reward
            print "new_state: ", new_state
            print "done :", done
            """
            target_action = actor.target_model.predict(np.expand_dims(new_state, axis=0))
            future_reward = self.target_model.predict([np.expand_dims(new_state, axis=0), target_action])[0][0]
            reward += gamma * future_reward
			
            X_states.append(cur_state)
            X_actions.append(action)
            y.append(reward)

        X_states = np.array(X_states)
        X_actions = np.array(X_actions)
        X = [X_states, X_actions]
        y = np.array(y)
        y = np.expand_dims(y, axis=1)
        # Train critic model
        self.model.fit(X, y, batch_size=batch_size, verbose = 0)

class ActorCritic:
    def __init__(self, env, actor, critic, DISCOUNT_FACTOR, MINIBATCH_SIZE, REPLAY_MEMORY_SIZE, TARGET_DISCOUNT, continue_execution, MEMORIES):
        # Environment details
        self.env = env
        self.actor = actor
        self.critic = critic
        self.MINIBATCH_SIZE = MINIBATCH_SIZE
        self.DISCOUNT = DISCOUNT_FACTOR
        self.TARGET_DISCOUNT = TARGET_DISCOUNT
        self.bg_noise = None
        self.action_dim = self.env.action_space.shape[0]
        
        # Replay memory to store experiences of the model with the environment
        # self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        if continue_execution:
            self.replay_memory = memory.Memory(REPLAY_MEMORY_SIZE, load=continue_execution, memories=MEMORIES)
        else:
            self.replay_memory = memory.Memory(REPLAY_MEMORY_SIZE)
        
    def train(self, mode):
        minibatch = self.replay_memory.getMiniBatch(self.MINIBATCH_SIZE, mode)
        self.critic.train(minibatch, self.actor, self.DISCOUNT, self.MINIBATCH_SIZE)
        self.actor.train(minibatch, self.critic, self.env)     
    
    def ou_noise(self, x, rho=0.15, mu=0, dt=0.1, sigma=0.2, dim=1):
        return x + rho * (mu-x) * dt + sigma * np.sqrt(dt) * np.random.normal(size=dim)
    
    def act(self, cur_state, new_episode, explorationRate):
        action = [] # array of action for learning [-1,1] or [0,1]
        action_step = [] # array of action for environment [min, max]
        action = self.actor.model.predict(np.expand_dims(cur_state, axis=0))[0]
        
        # Apply noise for exploration with greedy
        rand = random.random()
        if rand <= explorationRate :
            if new_episode: self.bg_noise = np.zeros(self.action_dim)
            noise = self.ou_noise(self.bg_noise, dim=self.action_dim) 
        else :
            noise = np.zeros(self.action_dim)
        action = np.clip(action + noise, -1, 1)
        
        for a in range(len(action)):
            action_step += [action[a]*self.env.action_space.high[a]]
            
        self.bg_noise = noise
        
        return action, action_step
    
    def saveModel(self, actor_path, critic_path, actor_target_path, critic_target_path):
        self.actor.model.save(actor_path)
        self.critic.model.save(critic_path)
        self.actor.target_model.save(actor_target_path)
        self.critic.target_model.save(critic_target_path)
        
    def loadModels(self, actor_path, critic_path, actor_target_path, critic_target_path):
        self.actor.model = load_model(actor_path)
        self.actor.target_model = load_model(actor_target_path)
        self.critic.model = load_model(critic_path)
        self.critic.target_model = load_model(critic_target_path)
                
    def updateTarget(self):
        # Update Actor model
        actor_model_weights  = self.actor.model.get_weights()
        actor_target_weights = self.actor.target_model.get_weights()
        
        """
        target_weights = TARGET_DISCOUNT*model_weights + (1-TARGET_DISCOUNT)*target_weights
        """
        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = self.TARGET_DISCOUNT*actor_model_weights[i] + (1-self.TARGET_DISCOUNT)*actor_target_weights[i]
        self.actor.target_model.set_weights(actor_target_weights)
		
		# Update Critic model
        critic_model_weights  = self.critic.model.get_weights()
        critic_target_weights = self.critic.target_model.get_weights()
		
        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = self.TARGET_DISCOUNT*critic_model_weights[i] + (1-self.TARGET_DISCOUNT)*critic_target_weights[i]
        self.critic.target_model.set_weights(critic_target_weights)

        #print "Network Upadated!"
