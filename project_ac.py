#!/usr/bin/env python

import gym
from gym import wrappers
import gym_gazebo
import time
from distutils.dir_util import copy_tree
import os
import json
import liveplot
import actor_critic as ac
import csv
import numpy as np
import tensorflow.compat.v1 as tf
import random

"""
# setting seed for reproducibility of results. This is not super important.
random.seed(2212)
np.random.seed(2212)
tf.set_random_seed(2212)
"""

def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith('openaigym')]

def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return
    for file in files:
        print(file)
        os.unlink(file)

if __name__ == '__main__':
    sess = tf.Session()
    tf.keras.backend.set_session(sess)
    
	#REMEMBER!: project_setup.bash must be executed.
    env = gym.make('GazeboProjectTurtlebotAc-v0')
    outdir = '/home/katolab/experiment_data/AC_data/gazebo_gym_experiments/'
    path = '/home/katolab/experiment_data/AC_data/project_dqn_ep'
    plotter = liveplot.LivePlot(outdir)
    
    action_dim = env.action_space.shape[0]
    observation_dim = env.observation_space.shape
    
    continue_execution = False
    #fill this if continue_execution=True
    resume_epoch = '100' # change to epoch to continue from
    resume_path = path + resume_epoch
    actor_weights_path =  resume_path + '_actor.h5'
    critic_weights_path = resume_path + '_critic.h5'
    actor_monitor_path = resume_path + '_actor'
    critic_monitor_path = resume_path + '_critic'
    params_json  = resume_path + '.json'
    
    if not continue_execution:
        #Each time we take a sample and update our weights it is called a mini-batch.
        #Each time we run through the entire dataset, it's called an epoch.
        #PARAMETER LIST
        EPISODES = 10
        STEPS = 100
        UPDATE_NETWORK = 1000
        EPSILON = 1
        EPSILON_DECAY = 0.999
        MIN_EPSILON = 0.001
        MINIBATCH_SIZE = 64
        MINIMUM_REPLAY_MEMORY = 100
        A_LEARNING_RATE = 0.001
        C_LEARNING_RATE = 0.005
        DISCOUNT_FACTOR = 0.99
        MEMORY_SIZE = 1000
        A_HIDDEN_LAYER = [128,128,128]
        C_HIDDEN_LAYER = [[128],[128,128]] # [[befor merging],[after merging]]
        CURRENT_EPISODE = 0

    else:
        #Load weights, monitor info and parameter info.
        with open(params_json) as outfile:
            d = json.load(outfile)
            EPISODES = 2000 #d.get('EPISODES')
            STEPS = d.get('STEPS')
            UPDATE_NETWORK = d.get('UPDATE_NETWORK')
            EPSILON = d.get('EPSILON')
            EPSILON_DECAY = d.get('EPSILON_DECAY')
            MIN_EPSILON = d.get('MIN_EPSILON')
            MINIBATCH_SIZE = d.get('MINIBATCH_SIZE')
            MINIMUM_REPLAY_MEMORY = d.get('MINIMUM_REPLAY_MEMORY')
            A_LEARNING_RATE = d.get('A_LEARNING_RATE')
            C_LEARNING_RATE = d.get('C_LEARNING_RATE')
            DISCOUNT_FACTOR = d.get('DISCOUNT_FACTOR')
            MEMORY_SIZE = d.get('MEMORY_SIZE')
            A_HIDDEN_LAYER = d.get('A_HIDDEN_LAYER')
            C_HIDDEN_LAYER = d.get('C_HIDDEN_LAYER')
            CURRENT_EPISODE = d.get('CURRENT_EPISODE')
            
        clear_monitor_files(outdir)
        copy_tree(actor_monitor_path,outdir)
        copy_tree(critic_monitor_path,outdir)
    
    # Actor model to take actions 
    # state -> action
    actor = ac.Actor(sess, action_dim, observation_dim, A_LEARNING_RATE, A_HIDDEN_LAYER)
    # Critic model to evaluate the action taken by the actor
    # state + action -> Expected reward to be achieved by taking action in the state.
    critic = ac.Critic(sess, action_dim, observation_dim, C_LEARNING_RATE, C_HIDDEN_LAYER)

    sess.run(tf.initialize_all_variables())
    actor_critic = ac.ActorCritic(env, actor, critic, DISCOUNT_FACTOR, MINIBATCH_SIZE, MEMORY_SIZE)
    
    if continue_execution : actor_critic.loadWeights(actor_weights_path, critic_weights_path)
    
    env._max_episode_steps = STEPS # env returns done after _max_episode_steps
    env = gym.wrappers.Monitor(env, outdir,force=not continue_execution, resume=continue_execution)

    stepCounter = 0
    min_distance = 20
    max_reward = 0
    
    start_time = time.time()

    #start iterating from 'current epoch'.
    for episode in xrange(CURRENT_EPISODE+1, EPISODES+1, 1):
        done = False
        cur_state = env.reset()
        episode_reward = 0
        episode_step = 0
        
        while not done:
            action = actor_critic.act(cur_state, EPSILON)
            next_state, reward, done, _ = env.step(action)

            episode_reward += reward

            # Add experience to replay memory
            actor_critic.replay_memory.append((cur_state, action, reward, next_state, done))

            cur_state = next_state
            
            episode_step += 1

            if len(actor_critic.replay_memory) < MINIMUM_REPLAY_MEMORY:
                continue

            actor_critic.train()

            if EPSILON > MIN_EPSILON and len(actor_critic.replay_memory) >= MINIMUM_REPLAY_MEMORY:
                EPSILON *= EPSILON_DECAY
                EPSILON = max(EPSILON, MIN_EPSILON)

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        
        max_reward = max(max_reward, episode_reward)
        
        print("EP:" + str(episode) + " - " + str(episode_step) + "/" + str(STEPS) + " steps |" + " Reward: " + str(episode_reward) + " | Max Reward:" + str(max_reward) + " | epsilon: " + str(EPSILON) + "| Time: %d:%02d:%02d" % (h, m, s))
        
        if (episode)%100==0:
            #save model weights and monitoring data every 100 epochs.
            actor_critic.saveModel(path+str(episode)+'_actor.h5', path+str(episode)+'_critic.h5')
            env._flush()
            copy_tree(outdir,path+str(episode)+'_actor')
            copy_tree(outdir,path+str(episode)+'_critic')
            
            #save simulation parameters.
            parameter_keys = ['EPISODES', 'STEPS', 'UPDATE_NETWORK', 'EPSILON', 'EPSILON_DECAY', 'MIN_EPSILON', 'MINIBATCH_SIZE', 'MINIMUM_REPLAY_MEMORY', 'A_LEARNING_RATE', 'C_LEARNING_RATE', 'DISCOUNT_FACTOR', 'MEMORY_SIZE', 'CURRENT_EPISODE']
            parameter_values = [EPISODES, STEPS, UPDATE_NETWORK, EPSILON, EPSILON_DECAY, MIN_EPSILON, MINIBATCH_SIZE, MINIMUM_REPLAY_MEMORY, A_LEARNING_RATE, C_LEARNING_RATE, DISCOUNT_FACTOR, MEMORY_SIZE, episode]
            parameter_dictionary = dict(zip(parameter_keys, parameter_values))
            with open(path+str(episode)+'.json', 'w') as outfile:
                json.dump(parameter_dictionary, outfile)
        
        # Save rewards
        with open('/home/katolab/experiment_data/AC_data/reward_ac.csv','a+') as csvRWRD:
            csvRWRD_writer = csv.writer(csvRWRD,dialect='excel')
            csvRWRD_writer.writerow([episode, episode_step, episode_reward])
        csvRWRD.close()
            
    input()
    env.close()
