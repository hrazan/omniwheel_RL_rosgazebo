#!/usr/bin/env python
import gym
from gym import wrappers
import gym_gazebo
import time
import numpy
import random
import time
import rospy

import q_qlearn as qlearn
import liveplot
import csv

def render():
    render_skip = 0 #Skip first X episodes.
    render_interval = 50 #Show render Every Y episodes.
    render_episodes = 10 #Show Z episodes every rendering.

    if (x%render_interval == 0) and (x != 0) and (x > render_skip):
        env.render()
    elif ((x-render_episodes)%render_interval == 0) and (x != 0) and (x > render_skip) and (render_episodes < x):
        env.render(close=True)

if __name__ == '__main__':

    env = gym.make('GazeboProjectTurtlebot-v0')

    outdir = '/tmp/gazebo_gym_experiments'
    env = gym.wrappers.Monitor(env, outdir, force=True)
    plotter = liveplot.LivePlot(outdir)
    
    last_time_steps = numpy.ndarray(0)

    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                    alpha=0.2, gamma=0.8, epsilon=0.0)

    initial_epsilon = qlearn.epsilon

    epsilon_discount = 0.9986

    start_time = time.time()
    total_goals = 0
    total_succeed_steps = 0
    total_episodes = 1001
    highest_reward = -10000000
    fewest_steps = 10000000

    best_act = []
    best_act_time = []
    best_position = []
    best_distance = []
    best_state_file = []
    best_reward_file = []

    for x in range(total_episodes):
        done = False

        cumulated_reward = 0 #Should going forward give more reward then L/R ?

        observation = env.reset()

        #render() #defined above, not env.render()

        state = ''.join(map(str, observation))
        i = 0
        act = []
        act_time = []
        position = []
        distance = []
        state_file = []
        reward_file = []

        #print('1:',env.pose)
        while not done:
            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            #print('2:',env.pose)

            # Execute the action and get feedback

            observation, reward, done, info = env.step(action)
            cumulated_reward += reward

            nextState = ''.join(map(str, observation))

            #qlearn.learn(state, action, reward, nextState)

            env._flush(force=True)

            act_time.append(env.action_time)
            
            act.append(action)
            state_file.append(state)
            reward_file.append(reward)
            position.append([(env.startpose.x,env.startpose.y),(env.pose.x,env.pose.y)])
            distance.append([numpy.sqrt(((env.startpose.x-env.pose.x)*(env.startpose.x-env.pose.x))+((env.startpose.y-env.pose.y)*(env.startpose.y-env.pose.y)))])

            if not done:
                state = nextState
                i += 1
            else:
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break


        if (x>0 and x%100==0):
            plotter.plot(env)
            csvQOpen = open('/home/razan/experiment_data/qtable.csv','w')
            csvq = csv.writer(csvQOpen, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
            for qstate,qval in qlearn.q.items():
                csvq.writerow([qstate[0],qstate[1],qval])
            csvQOpen.close()
            print qlearn.q

        if env.goal == True : 
            total_goals += 1
            total_succeed_steps += i+1
            if (i+1) <= fewest_steps:
                highest_reward = cumulated_reward
                fewest_steps = i+1
                best_act = act[:]
                best_act_time = act_time[:]
                best_position = position[:]
                best_distance = distance[:]
                best_state_file = state_file[:]
                best_reward_file = reward_file[:]

            # Make txt file of best actions
            with open('/home/razan/experiment_data/actions.txt','w') as txtfile:
	              for item in best_act:
		                txtfile.write("%s" % item)
            txtfile.close()

            # Make csv file of best actions' detail
            csvOpen = open('/home/razan/experiment_data/actions_details.csv','w')
            writer = csv.writer(csvOpen, dialect='excel')
            for act_num in range(len(best_act)):
                writer.writerow([best_state_file[act_num], best_act[act_num], best_reward_file[act_num], best_act_time[act_num]/1000000, best_position[act_num], best_distance[act_num]])
            csvOpen.close()
            

        print("Number of steps: "+str(i+1))
        print("Fewest succeed steps: "+str(fewest_steps))
        print("Highest succeed reward: "+str(highest_reward))
        print("Total goals: "+str(total_goals))
        if total_goals==0:
            print("Average succeed steps: 0")
        else:
            print("Average succeed steps: "+str(total_succeed_steps/total_goals))

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print ("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s))

    #Github table content
    print ("\n|"+str(total_episodes)+"|"+str(qlearn.alpha)+"|"+str(qlearn.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount))

    l = last_time_steps.tolist()
    l.sort()

    print("Highest succeed reward: "+str(highest_reward))
    print("Fewest steps: "+str(fewest_steps))
    if total_goals==0:
        print("Average succeed steps: 0")
    else:
        print("Average succeed steps: "+str(total_succeed_steps/total_goals))
    print("Total goals: "+str(total_goals))
    #print(qlearn.q)

    input()

    env.close()
