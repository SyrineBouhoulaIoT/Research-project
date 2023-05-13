import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
import itertools


# Set up logging configuration
# log_file = 'simulation.log'
# if os.path.exists(log_file):
#     os.remove(log_file)


# logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
"""
logging here is used to track each expirement , now using a normal text file , 
later on will change the configs and ucomment top code to do the loggign properly in a .log file 
"""

GRID_SIZE = 30
N_STATES = GRID_SIZE ** 2  # We have 900 possible grid positions
EPSILON = 0.1
ALPHA = 0.5
GAMMA = 0.9
MAX_ENERGY = 10000
RESOURCE_VALUE = 100
SIZE = 2.5  # micro b
RATE = 50  # micro b
STEP_SIZE = 1
"""
the grid will be defined like this : 
x  : 0 --------------------> 30
  0  
y |
  |
  |
  |
  |
  |
  |
  |
  30   
"""
"""
these are our variable values , can be changed to whaterver we want depending on the example 
for the sake of simplicity each step will have a value of 1 for now , we can update it to 0.5 later on 
"""


class Robot:
    def __init__(self, name, pos_x, pos_y, action, workload, resources):
        self.name = name
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.actions = action
        self.workload = workload  # lamda workload value λ  : 0 --> 10
        self.resources = resources  # robot resources : res  0 --> 100  : equivalent to MIPS in this case
        self.energy = MAX_ENERGY  # 10K JOULES ENERGY FIXED VALUE

    def get_state(self):
        position = (self.pos_x, self.pos_y)
        remaining_energy = self.energy
        available_resource = self.resources

        return [position, remaining_energy, available_resource]

    def step(self, direction):
        if direction == 'u' and self.pos_y > 0:
            self.pos_y -= 1
        elif direction == 'l' and self.pos_x > 0:
            self.pos_x -= 1
        elif direction == 'r' and self.pos_x < GRID_SIZE:
            self.pos_x += 1
        elif direction == 'd' and self.pos_y < GRID_SIZE:
            self.pos_y += 1
        else:
            return None

    def offload(self, target_robot):
        self.workload -= 1
        target_robot.workload += 1
        return None

    def calculate_delay(self):
        processing_delay = self.workload / self.resources
        transmission_delay = SIZE / RATE

        return processing_delay + transmission_delay

    def calculate_energy_consumption(self):
        return self.workload / self.resources

    def reward(self):
        return 1 / (self.calculate_delay() + self.calculate_energy_consumption())


def state_index(robot, Q_table):
    robot_state = robot.get_state()
    for index, state in enumerate(Q_table.values()):
        if robot_state == state:
            return index
    return None


def initialize_Q_table(num_states, num_agents, num_actions):
    Q_table = {}

    for state in range(num_states):
        Q_table[state] = {}
        for agent in range(num_agents):
            Q_table[state][agent] = np.zeros(num_actions)

    return Q_table


def main():
    num_agents = 3
    num_actions = 5  # ['left', 'right', 'up', 'down', 'offload']
    num_states = N_STATES
    num_episodes = 10000

    Q_table = initialize_Q_table(num_states, num_agents, num_actions)

    # Initialize robots for the simulation
    robots = [
        Robot("Robot 1", 0, 0, ['l', 'r', 'u', 'd', 'o'], 5, RESOURCE_VALUE),
        Robot("Robot 2", 15, 15, ['l', 'r', 'u', 'd', 'o'], 5, RESOURCE_VALUE),
        Robot("Robot 3", 29, 29, ['l', 'r', 'u', 'd', 'o'], 5, RESOURCE_VALUE)
    ]
    # here we can initialise with random values as well instead of set fixed coordinates always for our 3 robots
    with open('./LOG_FIle.txt', 'a') as f:
        avgrewards = {}
        avgrewards['1'] = []
        avgrewards['2'] = []
        avgrewards['3'] = []

        for episode in range(num_episodes):

            f.write(f'Starting episode {episode + 1}\n')
            for agent, robot in enumerate(robots):
                episode_rewards = []
                current_state = state_index(robot, Q_table)
                if current_state is None:
                    current_state = len(Q_table)
                    Q_table[current_state] = {agent: np.zeros(num_actions) for agent in range(num_agents)}

                if np.random.rand() < EPSILON:
                    action = np.random.choice(robot.actions)
                else:
                    action = robot.actions[np.argmax(Q_table[current_state][agent])]

                if action in ['l', 'r', 'u', 'd']:
                    robot.step(action)
                elif action == 'o':
                    min_workload_robot = min(robots, key=lambda r: r.workload if r != robot else float("inf"))
                    robot.offload(min_workload_robot)

                new_state = state_index(robot, Q_table)
                if new_state is None:
                    new_state = len(Q_table)
                    Q_table[new_state] = {agent: np.zeros(num_actions) for agent in range(num_agents)}

                reward = robot.reward()
                episode_rewards.append(reward)
                if episode % 10 == 0:
                    if agent == 1:
                        avgrewards['1'].append(sum(episode_rewards) / len(episode_rewards))
                    elif agent == 2:
                        avgrewards['2'].append(sum(episode_rewards) / len(episode_rewards))
                    else:
                        avgrewards['3'].append(sum(episode_rewards) / len(episode_rewards))
                    avg=np.mean(episode_rewards[-10:])
                    #print("episode{}/{} ,agent{}  avgscore : {}".format(episode,num_episodes,agent,avg))
                Q_table[current_state][agent] = (1 - ALPHA) * Q_table[current_state][agent] + ALPHA * (
                        reward + GAMMA * np.max(Q_table[new_state][agent]))
                #   logging.info(f'Episode {episode + 1}, Robot {agent + 1}, Action: {action}, Reward: {reward}, New State: {new_state}')
                f.write(
                    f'Episode {episode + 1}, Robot {agent + 1}, Action: {action}, Reward: {reward}, New State: {new_state}\n')

        print(avgrewards)

        # print(Q_table)

        # create a new figure and axis
        fig, ax = plt.subplots()

        # loop over the dictionary keys
        for key in avgrewards.keys():
            # plot the values for this key
            ax.plot(avgrewards[key], label=key)

        # add a legend to the plot
        ax.legend()

        # show the plot
        plt.show()
    f.close()


if __name__ == "__main__":
    # Define the hyperparameters to be tuned
    param_grid = {
        'EPSILON': [0.1, 0.2, 0.3],
        'ALPHA': [0.1, 0.5, 1.0],
        'GAMMA': [0.5, 0.9, 0.95],
        'MAX_ENERGY': [1000, 10000, 100000],
        'RESOURCE_VALUE': [10, 100, 1000],
    }

    # Create a list of all possible combinations of hyperparameters
    param_combinations = list(itertools.product(*param_grid.values()))


    # Define a function to run the experiment for each set of hyperparameters
    def run_experiment(params):
        # Set the hyperparameters
        global EPSILON, ALPHA, GAMMA, MAX_ENERGY, RESOURCE_VALUE

        EPSILON, ALPHA, GAMMA, MAX_ENERGY, RESOURCE_VALUE = params

        # Run the experiment
        rewards = main()
        # Summing all the rewards
        return sum(rewards)


    rewards = {}
    # Run the experiments for all hyperparameter combinations
    for i, params in enumerate(param_combinations):
        # TO-DO: Find the Maximum experiment and pick the right hyperparameters
        rewards[i] = run_experiment(params)
    index = max(rewards.items(), key=lambda x: x[1])[0]
    print(index)
    print(f"Optimal Hyperparameters are: {param_combinations[index]} with a total reward of: {rewards[index]}")

