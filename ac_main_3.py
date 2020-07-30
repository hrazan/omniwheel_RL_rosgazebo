#!/usr/bin/env python

import gym
from gym import wrappers
import gym_gazebo
import time
from distutils.dir_util import copy_tree
import os
import json
import liveplot
import ac_actorcritic as ac
import csv
import numpy as np
import tensorflow.compat.v1 as tf
import random
import ac_memory as memory
import pandas as pd

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
    action_dim = env.action_space.shape[0]
    observation_dim = env.observation_space.shape
    
    main_outdir = '/home/katolab/experiment_data/AC_data_3/'
    outdir = main_outdir + 'gazebo_gym_experiments/'
    path = main_outdir + 'project_dqn_ep'
    
    continue_execution = False
    if not continue_execution: os.makedirs(outdir)
    plotter = liveplot.LivePlot(outdir)
    
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
        EPISODES = 1000
        STEPS = 50
        UPDATE_NETWORK = 1 # once per number of steps
        MINIBATCH_SIZE = 64
        MINIMUM_REPLAY_MEMORY = 64
        A_LEARNING_RATE = 0.0001
        C_LEARNING_RATE = 0.0001
        REWARD_SCALE = 1
        DISCOUNT_FACTOR = 0.99
        MEMORY_SIZE = 100000
        A_HIDDEN_LAYER = [512,512,512]
        C_HIDDEN_LAYER = [[],[],[512,512,512]] # [[before merging critic],[before merging actor],[after merging]]
        CURRENT_EPISODE = 0
        TARGET_DISCOUNT = 0.001 # [0,1] 0: don't update target weights, 1: update target wieghts 100% from model weights
        MEMORIES = None

    else:
        #Load weights, monitor info and parameter info.
        with open(params_json) as outfile:
            d = json.load(outfile)
            EPISODES = d.get('EPISODES')
            STEPS = d.get('STEPS')
            UPDATE_NETWORK = d.get('UPDATE_NETWORK')
            MINIBATCH_SIZE = d.get('MINIBATCH_SIZE')
            MINIMUM_REPLAY_MEMORY = d.get('MINIMUM_REPLAY_MEMORY')
            A_LEARNING_RATE = d.get('A_LEARNING_RATE')
            C_LEARNING_RATE = d.get('C_LEARNING_RATE')
            REWARD_SCALE = d.get('REWARD_SCALE')
            DISCOUNT_FACTOR = d.get('DISCOUNT_FACTOR')
            MEMORY_SIZE = d.get('MEMORY_SIZE')
            A_HIDDEN_LAYER = d.get('A_HIDDEN_LAYER')
            C_HIDDEN_LAYER = d.get('C_HIDDEN_LAYER')
            CURRENT_EPISODE = d.get('CURRENT_EPISODE')
            TARGET_DISCOUNT = d.get('TARGET_DISCOUNT')
            MEMORIES = pd.read_csv(main_outdir + 'experience.csv', index_col=0, dtype = {'reward':np.float64, 'done':np.float32})
            
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
    actor_critic = ac.ActorCritic(env, actor, critic, DISCOUNT_FACTOR, MINIBATCH_SIZE, MEMORY_SIZE, TARGET_DISCOUNT, continue_execution, MEMORIES)
    
    if continue_execution : actor_critic.loadWeights(actor_weights_path, critic_weights_path)
    
    env._max_episode_steps = STEPS # env returns done after _max_episode_steps
    env = gym.wrappers.Monitor(env, outdir,force=not continue_execution, resume=continue_execution)

    stepCounter = 0
    min_distance = 20
    max_reward = 0
    
    start_time = time.time()
    
    env.set_start_mode("static") #"random" or "static"
    
    #start iterating from 'current epoch'
    for episode in xrange(CURRENT_EPISODE+1, EPISODES+1, 1):
        done = False
        cur_state = env.reset()
        action_memory = memory.Memory(STEPS)
        episode_reward = 0
        episode_step = 0
        new_episode = True
        while not done:
            action, action_step = actor_critic.act(cur_state, new_episode)
            next_state, reward, done, _ = env.step(action_step)

            episode_reward += reward

            # Add experience to replay memory
            actor_critic.replay_memory.addMemory(cur_state, action, reward*REWARD_SCALE, next_state, done)
            action_memory.addMemory(cur_state, action, reward, next_state, done)

            cur_state = next_state
            
            episode_step += 1
            stepCounter += 1

            if len(actor_critic.replay_memory.exp.index) >= MINIMUM_REPLAY_MEMORY:
                actor_critic.train('random')
            
            if stepCounter%UPDATE_NETWORK == 0:
                actor_critic.updateTarget()
                
            new_episode = done

        resetVel = False
        while not resetVel:
            try:
                env.reset_vel()
                resetVel = True
            except:
                pass
        
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        
        if env.subgoal_as_dist_to_goal < min_distance:
            min_distance = env.subgoal_as_dist_to_goal
            action_memory.exp.to_csv(outdir + 'min_distance.csv')
        if max_reward < episode_reward:
            max_reward = episode_reward
            action_memory.exp.to_csv(outdir + 'max_reward.csv')
        
        print("EP:" + str(episode) + " - " + str(episode_step) + "/" + str(STEPS) + " steps |" + " Reward: " + str(episode_reward) + " | Max Reward: " + str(max_reward) + " | Min Distance: " + str(min_distance) + "| Time: %d:%02d:%02d" % (h, m, s))
        
        if (episode)%100==0:            
            #save model weights and monitoring data every 100 epochs.
            actor_critic.saveModel(path+str(episode)+'_actor.h5', path+str(episode)+'_critic.h5')
            env._flush()
            copy_tree(outdir,path+str(episode)+'_actor')
            copy_tree(outdir,path+str(episode)+'_critic')
            
            #save simulation parameters.
            parameter_keys = ['EPISODES', 'STEPS', 'UPDATE_NETWORK', 'MINIBATCH_SIZE', 'MINIMUM_REPLAY_MEMORY', 'A_LEARNING_RATE', 'C_LEARNING_RATE', 'REWARD_SCALE', 'DISCOUNT_FACTOR', 'MEMORY_SIZE', 'A_HIDDEN_LAYER', 'C_HIDDEN_LAYER', 'CURRENT_EPISODE', 'TARGET_DISCOUNT']
            parameter_values = [EPISODES, STEPS, UPDATE_NETWORK, MINIBATCH_SIZE, MINIMUM_REPLAY_MEMORY, A_LEARNING_RATE, C_LEARNING_RATE, REWARD_SCALE, DISCOUNT_FACTOR, MEMORY_SIZE, A_HIDDEN_LAYER, C_HIDDEN_LAYER, episode, TARGET_DISCOUNT]
            parameter_dictionary = dict(zip(parameter_keys, parameter_values))
            with open(path+str(episode)+'.json', 'w') as outfile:
                json.dump(parameter_dictionary, outfile)
            
            #save experiences data
            actor_critic.replay_memory.exp.to_csv(main_outdir + 'experience.csv')
            
            # Show rewards graph
            plotter.plot(env, outdir)
        
        # Save rewards
        with open(main_outdir + 'reward_ac.csv','a+') as csvRWRD:
            csvRWRD_writer = csv.writer(csvRWRD,dialect='excel')
            csvRWRD_writer.writerow([episode, episode_step, episode_reward, env.subgoal_as_dist_to_goal])
        csvRWRD.close()
            
    env.close()
