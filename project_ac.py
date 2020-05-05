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

# Hyperparameters
MINIMUM_REPLAY_MEMORY = 1000
EPSILON = 1
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.001
DISCOUNT = 0.99
EPISODES = 2000
STEPS = 100

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

    continue_execution = False
    #fill this if continue_execution=True
    resume_epoch = '100' # change to epoch to continue from
    resume_path = path + resume_epoch
    weights_path = resume_path + '.h5'
    monitor_path = resume_path
    params_json  = resume_path + '.json'
    
    env._max_episode_steps = STEPS # env returns done after _max_episode_steps
    env = gym.wrappers.Monitor(env, outdir,force=not continue_execution, resume=continue_execution)

    actor_critic = ac.ActorCritic(env, sess)

    stepCounter = 0
    min_distance = 20
    max_reward = 0
    
    start_time = time.time()

    for episode in range(EPISODES):
        done = False
        cur_state = env.reset()
        episode_reward = 0
        episode_step = 0
        
        while not done:
            action = actor_critic.act(cur_state, EPSILON)
            
            action = np.array(action, dtype=np.float32)
            #print(action[0] )
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

        #some bookkeeping
        """
        if(episode_reward > 400 and episode_reward > max_reward):
            actor.model.save_weights(str(episode_reward)+".h5")
        max_reward = max(max_reward, episode_reward)
        """
        print("Episode:" + str(episode) + " - " + str(episode_step) + "/" + str(STEPS) + " steps |" + " Episodic Reward: " + str(episode_reward) + " | Max Reward Achieved:" + str(max_reward) + " | EPSILON: " + str(EPSILON))
        
        with open('/home/katolab/experiment_data/reward.csv','a+') as csvRWRD:
            csvRWRD_writer = csv.writer(csvRWRD,dialect='excel')
            csvRWRD_writer.writerow([episode_step, episode_reward])
        csvRWRD.close()
            
    input()
    env.close()
