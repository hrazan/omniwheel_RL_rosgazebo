#!/usr/bin/env python

'''
Based on:
https://github.com/vmayoral/basic_reinforcement_learning
https://gist.github.com/wingedsheep/4199594b02138dd427c22a540d6d6b8d
'''
import gym
from gym import wrappers
import gym_gazebo
import time
from distutils.dir_util import copy_tree
import os
import json
import liveplot
import deepq_project as deepq
import csv

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

    #REMEMBER!: turtlebot_nn_setup.bash must be executed.
    env = gym.make('GazeboProjectNnTurtlebot-v0')
    #outdir = '/tmp/gazebo_gym_experiments/'
    #path = '/tmp/project_dqn_ep'
    outdir = '/home/katolab/experiment_data/NN_data_1/gazebo_gym_experiments/'
    path = '/home/katolab/experiment_data/NN_data_1/project_dqn_ep'
    plotter = liveplot.LivePlot(outdir)

    continue_execution = False
    #fill this if continue_execution=True
    resume_epoch = '700' # change to epoch to continue from
    resume_path = path + resume_epoch
    weights_path = resume_path + '.h5'
    monitor_path = resume_path
    params_json  = resume_path + '.json'

    if not continue_execution:
        #Each time we take a sample and update our weights it is called a mini-batch.
        #Each time we run through the entire dataset, it's called an epoch.
        #PARAMETER LIST
        epochs = 2000
        steps = 100
        updateTargetNetwork = 1000
        explorationRate = 1
        minibatch_size = 64
        learnStart = 64
        learningRate = 0.00001
        discountFactor = 0.99
        memorySize = 1000000
        network_inputs = 109
        network_outputs = 5
        network_structure = [1000,1000,1000]
        current_epoch = 0

        deepQ = deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure)
    else:
        #Load weights, monitor info and parameter info.
        #ADD TRY CATCH fro this else
        with open(params_json) as outfile:
            d = json.load(outfile)
            epochs = 2000
            steps = 100 #d.get('steps')
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

        deepQ = deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure)

        deepQ.loadWeights(weights_path)

        clear_monitor_files(outdir)
        copy_tree(monitor_path,outdir)

    env._max_episode_steps = steps # env returns done after _max_episode_steps
    env = gym.wrappers.Monitor(env, outdir,force=not continue_execution, resume=continue_execution)

    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False
    stepCounter = 0
    highest_reward = 0
    highest_reward_file = []

    start_time = time.time()

    #start iterating from 'current epoch'.
    for epoch in xrange(current_epoch+1, epochs+1, 1):
        observation = env.reset()
        cumulated_reward = 0
        done = False
        episode_step = 0

        # run until env returns done
        while not done:
            # env.render()
            qValues = deepQ.getQValues(observation)
            #print(observation)
            
            action = deepQ.selectAction(qValues, explorationRate)

            newObservation, reward, done, info = env.step(action)

            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            deepQ.addMemory(observation, action, reward, newObservation, done)

            if stepCounter >= learnStart:
                if stepCounter <= updateTargetNetwork:
                    deepQ.learnOnMiniBatch(minibatch_size, False)
                else :
                    deepQ.learnOnMiniBatch(minibatch_size, True)

            observation = newObservation

            if done:
                last100Scores[last100ScoresIndex] = episode_step
                last100ScoresIndex += 1
                if last100ScoresIndex >= 100:
                    last100Filled = True
                    last100ScoresIndex = 0
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)
                if not last100Filled:
                    print ("EP " + str(epoch) + " - " + format(episode_step + 1) + "/" + str(steps) + " Episode steps - Cumulated R: " + str(cumulated_reward) + "   Eps=" + str(round(explorationRate, 2)) + "     Time: %d:%02d:%02d" % (h, m, s))
                else :
                    """
                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)
                    """
                    print ("EP " + str(epoch) + " - " + format(episode_step + 1) + "/" + str(steps) + " Episode steps - last100 Steps : " + str((sum(last100Scores) / len(last100Scores))) + " - Cumulated R: " + str(cumulated_reward) + "   Eps=" + str(round(explorationRate, 2)) + "     Time: %d:%02d:%02d" % (h, m, s))
                    if (epoch)%100==0:
                        #save model weights and monitoring data every 100 epochs.
                        deepQ.saveModel(path+str(epoch)+'.h5')
                        env._flush()
                        copy_tree(outdir,path+str(epoch))
                        #save simulation parameters.
                        parameter_keys = ['epochs','steps','updateTargetNetwork','explorationRate','minibatch_size','learnStart','learningRate','discountFactor','memorySize','network_inputs','network_outputs','network_structure','current_epoch']
                        parameter_values = [epochs, steps, updateTargetNetwork, explorationRate, minibatch_size, learnStart, learningRate, discountFactor, memorySize, network_inputs, network_outputs, network_structure, epoch]
                        parameter_dictionary = dict(zip(parameter_keys, parameter_values))
                        with open(path+str(epoch)+'.json', 'w') as outfile:
                            json.dump(parameter_dictionary, outfile)

            stepCounter += 1
            if stepCounter % updateTargetNetwork == 0:
                deepQ.updateTargetNetwork()
                print ("updating target network")

            episode_step += 1

        with open('/home/katolab/experiment_data/reward_1.csv','a+') as csvRWRD:
            csvRWRD_writer = csv.writer(csvRWRD,dialect='excel')
            csvRWRD_writer.writerow([episode_step, cumulated_reward])
        csvRWRD.close()

        explorationRate *= 0.997 #epsilon decay
        # explorationRate -= (2.0/epochs)
        explorationRate = max (0.05, explorationRate)

        if epoch % 100 == 0:
            plotter.plot(env)
            
    #csvRWRD.close()
    input()
    env.close()
