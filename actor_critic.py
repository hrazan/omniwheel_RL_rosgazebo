import random

import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Add, Input, Concatenate
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow as tf
from collections import deque
import memory 

class Actor:
    def __init__(self, sess, action_dim, observation_dim, learningRate, hiddenLayer):
        # setting our created session as default session
        self.sess = sess
        K.set_session(sess)
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.learningRate = learningRate
        self.state_input, self.output, self.model = self.create_model(hiddenLayer)
        _, _, self.target_model = self.create_model(hiddenLayer)
        model_weights = self.model.trainable_weights
        # Placeholder for critic gradients with respect to action_input.
        self.actor_critic_grads = tf.placeholder(tf.float32, [None, action_dim])
        # Adding small number inside log to avoid log(0) = -infinity
        log_prob = tf.math.log(self.output + 10e-10)
        # Multiply log by -1 to convert the optimization problem as minimization problem.
        # This step is essential because apply_gradients always do minimization.
        neg_log_prob = tf.multiply(log_prob, -1)
        # Calulate and update the weights of the model to optimize the actor
        self.actor_grads = tf.gradients(neg_log_prob, model_weights, self.actor_critic_grads)
        grads = zip(self.actor_grads, model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learningRate).apply_gradients(grads)

    def create_model(self, hiddenLayer):
        state_h = hiddenLayer[:]
        
        if len(state_h)<1: error
        state_input = Input(shape=self.observation_dim)
        try:
            state_h[0] = Dense(state_h[0], activation='relu')(state_input)
        except ValueError:
            print "Error: insert at least one hidden layer of Actor's model"
        if len(state_h)>1:
            for layer in xrange(1, len(state_h), 1):
                state_h[layer] = Dense(state_h[layer], activation='relu')(state_h[layer-1])
        output = Dense(self.action_dim, activation='softmax')(state_h[-1])
        
        model = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=self.learningRate)
        model.compile(loss='categorical_crossentropy', optimizer=adam)
        model.summary()
        return state_input, output, model

    def train(self, critic_gradients_val, X_states):
        #print self.action_dim, self.observation_dim
        self.sess.run(self.optimize, feed_dict={self.state_input:X_states, self.actor_critic_grads:critic_gradients_val})

class Critic:
    def __init__(self, sess, action_dim, observation_dim, learningRate, hiddenLayer):
        # setting our created session as default session
        K.set_session(sess)
        self.sess = sess
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.learningRate = learningRate
        self.state_input, self.action_input, self.output, self.model = self.create_model(hiddenLayer)
        _, _, self.target_model = self.create_model(hiddenLayer)
        self.critic_gradients = tf.gradients(self.output, self.action_input)

    def create_model(self, hiddenLayer):
        state_h = hiddenLayer[0][:]
        merge_h = hiddenLayer[1][:]
        
        # Before merging with action
        state_input = Input(shape=self.observation_dim)
        try:
            state_h[0] = Dense(state_h[0], activation='relu')(state_input)
        except ValueError:
            print "Error: insert at least one hidden layer of Critic's model"
        if len(state_h)>1:
            for layer in xrange(1, len(state_h), 1):
                state_h[layer] = Dense(state_h[layer], activation='relu')(state_h[layer-1])
        
        # After merging with action
        action_input = Input(shape=(self.action_dim,))
        merge_layer = Concatenate()([state_h[-1], action_input])
        try:
            merge_h[0] = Dense(merge_h[0], activation='relu')(merge_layer)
        except ValueError:
            print "Error: insert at least one hidden layer of Critic's model"
        if len(merge_h)>1:
            for layer in xrange(1, len(merge_h), 1):
                merge_h[layer] = Dense(merge_h[layer], activation='relu')(merge_h[layer-1])
        output = Dense(1, activation='linear')(merge_h[-1])
        
        model = Model(inputs=[state_input, action_input], outputs=output)
        model.compile(loss="mse", optimizer=Adam(lr=self.learningRate))
        model.summary()
        return state_input, action_input, output, model

    def get_critic_gradients(self, X_states, X_actions):
        # critic gradients with respect to action_input to feed in the weight updation of actor
        critic_gradients_val = self.sess.run(self.critic_gradients, feed_dict={self.state_input:X_states, self.action_input:X_actions})
        return critic_gradients_val[0]

class ActorCritic:
    def __init__(self, env, actor, critic, DISCOUNT_FACTOR, MINIBATCH_SIZE, REPLAY_MEMORY_SIZE):
        # Environment details
        self.env = env
        self.actor = actor
        self.critic = critic
        self.MINIBATCH_SIZE = MINIBATCH_SIZE
        self.DISCOUNT = DISCOUNT_FACTOR
        
        # Replay memory to store experiences of the model with the environment
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        
    def train(self):
        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)

        X_states = []
        X_actions = []
        y = []
        for sample in minibatch:
            cur_state, cur_action, reward, next_state, done = sample
            next_actions = self.actor.model.predict(np.expand_dims(next_state, axis=0))
            
            # Q(st, at) = reward + DISCOUNT * Q(s(t+1), a(t+1))
            next_reward = self.critic.model.predict([np.expand_dims(next_state, axis=0), next_actions])[0][0]
            reward = reward + self.DISCOUNT * next_reward

            X_states.append(cur_state)
            X_actions.append(cur_action)
            y.append(reward)

        X_states = np.array(X_states)
        X_actions = np.array(X_actions)
        X = [X_states, X_actions]
        y = np.array(y)
        y = np.expand_dims(y, axis=1)
        # Train critic model
        self.critic.model.fit(X, y, batch_size=self.MINIBATCH_SIZE, verbose = 0)

        # Get the actions for the cur_states from the minibatch.
        # We are doing this because now actor may have learnt more optimal actions for given states
        # as Actor is constantly learning and we are picking the states from the previous experiences.
        X_actions_new = []
        for sample in minibatch:
            X_actions_new.append(self.actor.model.predict(np.expand_dims(sample[0], axis=0))[0])
        X_actions_new = np.array(X_actions_new)

        # grad(J(actor_weights)) = sum[ grad(log(pi(at | st, actor_weights)) * grad(critic_output, action_input), actor_weights) ]
        critic_gradients_val = self.critic.get_critic_gradients(X_states, X_actions)
        self.actor.train(critic_gradients_val, X_states)
    
    def act(self, cur_state, epsilon):
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
            action = np.array(action, dtype=np.float32)
            return action
        return self.actor.model.predict(np.expand_dims(cur_state, axis=0))[0]
    
    def saveModel(self, actor_path, critic_path):
        self.actor.model.save(actor_path)
        self.critic.model.save(critic_path)
        
    def loadWeights(self, actor_weights_path, critic_weights_path):
        self.actor.model.set_weights(load_model(actor_weights_path).get_weights())
        self.critic.model.set_weights(load_model(critic_weights_path).get_weights())
                
    def updateTarget(self):
        # Update Actor model
        actor_model_weights  = self.actor.model.get_weights()
		actor_target_weights = self.actor.target_model.get_weights()
		
		for i in range(len(actor_target_weights)):
			actor_target_weights[i] = actor_model_weights[i]
		self.actor.target_model.set_weights(actor_target_weights)
		
		# Update Critic model
        critic_model_weights  = self.critic.model.get_weights()
		critic_target_weights = self.critic.target_model.get_weights()
		
		for i in range(len(critic_target_weights)):
			critic_target_weights[i] = critic_model_weights[i]
		self.critic.target_model.set_weights(critic_target_weights)

