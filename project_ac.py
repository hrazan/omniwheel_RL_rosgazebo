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

        actor_critic = ac.ActorCritic(env, memorySize, discountFactor, learningRate, batch_size)
    else:
        #Load weights, monitor info and parameter info.
        #ADD TRY CATCH fro this else
        with open(params_json) as outfile:
            d = json.load(outfile)
            epochs = d.get('episodes')
            steps = d.get('steps')
            updateTargetNetwork = d.get('updateTargetNetwork')
            epsilon = d.get('epsilon')
            epsilon_decay = d.get('epsilon_decay')
            batch_size = d.get('batch_size')
            learningRate = d.get('learningRate')
            discountFactor = d.get('discountFactor')
            memorySize = d.get('memorySize')
            #network_inputs = d.get('network_inputs')
            #network_outputs = d.get('network_outputs')
            #network_structure = d.get('network_structure')
            current_episode = d.get('current_episode')

        actor_critic = ac.ActorCritic(env, sess, memorySize, discountFactor, learningRate, batch_size)

        actor_critic.loadWeights(weights_path)

        clear_monitor_files(outdir)
        copy_tree(monitor_path,outdir)

    env._max_episode_steps = steps # env returns done after _max_episode_steps
    env = gym.wrappers.Monitor(env, outdir,force=not continue_execution, resume=continue_execution)

    stepCounter = 0
    min_distance = 20
    
    start_time = time.time()

    for eps in xrange(current_episode+1, episodes+1, 1):
        cur_state = env.reset()
        if np.any(np.isinf(cur_state)): print('ERROR! 0')
        cumulated_reward = 0
        done = False
        episode_step = 0
        
        # run intil env returns done
        while not done:
            cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
            if np.any(np.isinf(cur_state)): print('ERROR! 1')
            #print('state: ' + str(cur_state))
            action = actor_critic.act(cur_state, epsilon)
            #print('actor_critic.act(cur_state, epsilon): ' + str(action))
            action = action.reshape((1, env.action_space.shape[0]))
            if np.any(np.isnan(action)): print('ERROR! 2')
            #print('action.reshape((1, env.action_space.shape[0]): ' + str(action))

            new_state, reward, done, _ = env.step(action)
            new_state = new_state.reshape((1, env.observation_space.shape[0]))

            actor_critic.remember(cur_state, action, reward, new_state, done)
            actor_critic.train()
            
            cumulated_reward += reward

            cur_state = new_state
            #print(cur_state[0][-1])
            if cur_state[0][-1] < min_distance:
                min_distance = cur_state[0][-1]
            
            episode_step += 1
            stepCounter += 1
            if stepCounter % updateTargetNetwork == 0:
                actor_critic.update_target()
                print ("updating target network")
        
        #env.reset_vel()
        
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        
        print ("EP " + str(eps) + " - " + format(episode_step) + "/" + str(steps) + " steps | e=" + str(round(epsilon, 2)) + " | R: " + str(cumulated_reward) + " | To goal: " + str(min_distance)+ " | Time: %d:%02d:%02d" % (h, m, s))
        
        """
        if (eps)%100==0:
            #save model weights and monitoring data every 100 epochs.
            actor_critic.saveModel(path+str(epoch)+'.h5')
            env._flush()
            copy_tree(outdir,path+str(epoch))
            #save simulation parameters.
            parameter_keys = ['episodes','steps','updateTargetNetwork','epsilon','epsilon_decay', 'batch_size','learningRate','discountFactor','memorySize','current_episode']
            parameter_values = [episodes, steps, updateTargetNetwork, epsilon, epsilon_decay, batch_size, learningRate, discountFactor, memorySize, eps]
            parameter_dictionary = dict(zip(parameter_keys, parameter_values))
            with open(path+str(eps)+'.json', 'w') as outfile:
                json.dump(parameter_dictionary, outfile)
		"""
		
        with open('/home/katolab/experiment_data/AC_data/reward_ac.csv','a+') as csvRWRD:
            csvRWRD_writer = csv.writer(csvRWRD,dialect='excel')
            csvRWRD_writer.writerow([episode_step, cumulated_reward])
        csvRWRD.close()
		
        epsilon *= epsilon_decay
        epsilon = max (0.05, epsilon)
            
    input()
    env.close()
