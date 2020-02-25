# omniwheel_RL_rosgazebo
## Make It Works
### 1. Install The [Gym-Gazebo](https://github.com/erlerobot/gym-gazebo)
### 2. Write below code to ~/gym-gazebo/gym_gazebo/__init__.py :
```
register(
    id='GazeboProjectTurtlebot-v0',
    entry_point='gym_gazebo.project_envs.project:ProjectEnv',
    #entry_point='gym_gazebo.envs.turtlebot:ProjectEnv',
    # More arguments here
)
register(
    id='GazeboProjectNnTurtlebot-v0',
    entry_point='gym_gazebo.project_envs.project:ProjectNnEnv',
    #entry_point='gym_gazebo.envs.turtlebot:ProjectNnEnv',
    max_episode_steps=1000,
    # More arguments here
)
```
### 3. Copy project_setup.bash to ~/gym-gazebo/gym_gazebo/envs/installation
project_setup.bash
```
#!/bin/bash

if [ -z "$GAZEBO_MODEL_PATH" ]; then
  bash -c 'echo "export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:"`pwd`/../gym_gazebo/project_envs/assets/models >> ~/.bashrc'
else
  bash -c 'sed "s,GAZEBO_MODEL_PATH=[^;]*,'GAZEBO_MODEL_PATH=`pwd`/../gym_gazebo/project_envs/assets/models'," -i ~/.bashrc'
fi

#add turtlebot launch environment variable
if [ -z "$PROJECT" ]; then
  bash -c 'echo "export PROJECT="`pwd`/../gym_gazebo/project_envs/assets/worlds/project.world >> ~/.bashrc'
else
  bash -c 'sed "s,PROJECT=[^;]*,'PROJECT=`pwd`/../gym_gazebo/project_envs/assets/worlds/project.world'," -i ~/.bashrc'
fi
 
exec bash # reload bash
```
### 4. Copy [omniwheel_RL_rosgazebo_envs](https://github.com/hrazan/omniwheel_RL_rosgazebo_envs) to ~gym-gazebo/gym_gazebo then rename it as "project_envs"