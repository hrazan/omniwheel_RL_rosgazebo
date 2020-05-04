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
import tensorflow.compat.v1 as tf

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
    env = gym.make('GazeboProjectAcOmnirobot-v0')
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

    if not continue_execution:
        #Each time we take a sample and update our weights it is called a mini-batch.
        #Each time we run through the entire dataset, it's called an epoch.
        #PARAMETER LIST
        episodes = 2000
        steps = 100
        updateTargetNetwork = 1000
        epsilon = 1
        epsilon_decay = 0.97
        batch_size = 64
        discountFactor = 0.99
        memorySize = 1000000
        #actor_net_structure = [1000,1000,1000]
        #critic_net_structure = [1000,1000,1000]
        #actor_learningRate = 0.00001
        #critic_learningRate = 0.00001
        learningRate = 0.00001
        current_episode = 0

        actor_critic = ac.ActorCritic(env, sess, memorySize, discountFactor, learningRate, batch_size)
    else:
        #Load weights, monitor info and parameter info.
        #ADD TRY CATCH fro this else
        with open(params_json) as outfile:
            d = json.load(outfile)
            epochs = d.get('epochs')
            steps = d.get('steps')
            updateTargetNetwork = d.get('updateTargetNetwork')
            explorationRate = d.get('explorationRate')
            minibatch_size = d.get('minibatch_size')
            learnStart = d.get('learnStart')
            learningRate = d.get('learningRate')
            discountFactor = d.get('discountFactor')
            memorySize = d.get('memorySize')
            network_inputs = d.get('network_inputs')
            network_outputs = d.get('network_outputs')
            network_structure = d.get('network_structure')
            current_epoch = d.get('current_epoch')

        actor_critic = ac.ActorCritic(env, sess, memorySize, discountFactor, learningRate, batch_size)

        actor_critic.loadWeights(weights_path)

        clear_monitor_files(outdir)
        copy_tree(monitor_path,outdir)

    env = gym.wrappers.Monitor(env, outdir,force=not continue_execution, resume=continue_execution)

    cur_state = env.reset()
    action = env.action_space.sample()
    stepCounter = 0

    for eps in xrange(current_episode+1, episodes+1, 1):
        cur_state = env.reset()
        cumulated_reward = 0
        done = False
        episode_step = 0
        
        # run intil env returns done
        while not done:
            cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
            action = actor_critic.act(cur_state, epsilon)
            #print('actor_critic.act(cur_state, epsilon): ' + str(action))
            action = action.reshape((1, env.action_space.shape[0]))
            #print('action.reshape((1, env.action_space.shape[0]): ' + str(action))

            new_state, reward, done, _ = env.step(action)
            new_state = new_state.reshape((1, env.observation_space.shape[0]))

            actor_critic.remember(cur_state, action, reward, new_state, done)
            actor_critic.train()

            cur_state = new_state
            
            stepCounter += 1
            if stepCounter % updateTargetNetwork == 0:
                actor_critic.update_target()
                print ("updating target network")
		
        epsilon *= epsilon_decay
        epsilon = max (0.05, epsilon)
            
    input()
    env.close()
