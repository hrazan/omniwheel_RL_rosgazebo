import random

import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Add, Input, Concatenate
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from collections import deque
import memory 

# Hyperparameters
REPLAY_MEMORY_SIZE = 100000
MINIBATCH_SIZE = 64
DISCOUNT = 0.99

class Actor(object):
    def __init__(self, sess, action_dim, observation_dim):
        # setting our created session as default session
        self.sess = sess
        K.set_session(sess)
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.state_input, self.output, self.model = self.create_model()
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
        self.optimize = tf.train.AdamOptimizer(0.001).apply_gradients(grads)

    def create_model(self):
        state_input = Input(shape=self.observation_dim)
        state_h1 = Dense(128, activation='relu')(state_input)
        state_h2 = Dense(128, activation='relu')(state_h1)
        state_h3 = Dense(128, activation='relu')(state_h2)
        output = Dense(self.action_dim, activation='softmax')(state_h2)
        model = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=adam)
        return state_input, output, model

    def train(self, critic_gradients_val, X_states):
        self.sess.run(self.optimize, feed_dict={self.state_input:X_states, self.actor_critic_grads:critic_gradients_val})

class Critic(object):
    def __init__(self, sess, action_dim, observation_dim):
        # setting our created session as default session
        K.set_session(sess)
        self.sess = sess
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.state_input, self.action_input, self.output, self.model = self.create_model()
        self.critic_gradients = tf.gradients(self.output, self.action_input)

    def create_model(self):
        state_input = Input(shape=self.observation_dim)
        state_h1 = Dense(128, activation='relu')(state_input)
        
        action_input = Input(shape=(self.action_dim,))
        state_action = Concatenate()([state_h1, action_input])
        state_action_h1 = Dense(128, activation='relu')(state_action)
        state_action_h2 = Dense(128, activation='relu')(state_action_h1)        
        output = Dense(1, activation='linear')(state_action_h1)

        model = Model(inputs=[state_input, action_input], outputs=output)
        model.compile(loss="mse", optimizer=Adam(lr=0.005))
        return state_input, action_input, output, model

    def get_critic_gradients(self, X_states, X_actions):
        # critic gradients with respect to action_input to feed in the weight updation of actor
        critic_gradients_val = self.sess.run(self.critic_gradients, feed_dict={self.state_input:X_states, self.action_input:X_actions})
        return critic_gradients_val[0]

class ActorCritic:
    def __init__(self, env, sess):
        # Environment details
        self.env = env
        self.action_dim = env.action_space.shape[0]
        self.observation_dim = env.observation_space.shape[0]
        
        # Actor model to take actions 
        # state -> action
        self.actor = Actor(sess, self.action_dim, self.observation_dim)
        # Critic model to evaluate the action taken by the actor
        # state + action -> Expected reward to be achieved by taking action in the state.
        self.critic = Critic(sess, self.action_dim, self.observation_dim)
        
        # Replay memory to store experiences of the model with the environment
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        
    def train(self):
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        X_states = []
        X_actions = []
        y = []
        for sample in minibatch:
            cur_state, cur_action, reward, next_state, done = sample
            next_actions = self.actor.model.predict(np.expand_dims(next_state, axis=0))
            if done:
                # If episode ends means we have lost the game so we give -ve reward
                # Q(st, at) = -reward
                reward = -reward
            else:
                # Q(st, at) = reward + DISCOUNT * Q(s(t+1), a(t+1))
                next_reward = self.critic.model.predict([np.expand_dims(next_state, axis=0), next_actions])[0][0]
                reward = reward + DISCOUNT * next_reward

            X_states.append(cur_state)
            X_actions.append(cur_action)
            y.append(reward)

        X_states = np.array(X_states)
        X_actions = np.array(X_actions)
        X = [X_states, X_actions]
        y = np.array(y)
        y = np.expand_dims(y, axis=1)
        # Train critic model
        self.critic.model.fit(X, y, batch_size=MINIBATCH_SIZE, verbose = 0)

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
            return self.env.action_space.sample()
        return self.actor.predict(cur_state)


