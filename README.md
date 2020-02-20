# omniwheel_RL_rosgazebo
# in ~/gym-gazebo/gym_gazebo/__init__.py :
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

# copy project_setup.bash to ~/gym-gazebo/gym_gazebo/envs/installation
# copy submodule "omniwheel_RL_rosgazebo_envs @ 07f31b3" to ~gym-gazebo/gym_gazebo then rename it as "project_envs"