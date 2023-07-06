import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import json
import hashlib
import pickle
RESOURCE_VALUE = 3
WORKLOADMAX = 3
# GRID_SIZE = 2
Energy_range = 3
N_STATES = (RESOURCE_VALUE * WORKLOADMAX* Energy_range) ** 3
N_ACTIONS = (3 ** 3)   # nmbr action ** nmbr robot
EPSILON = 1  # Initial exploration rate
EPSILON_DECAY = 0.99999  # Decay rate for exploration rate
ALPHA = 0.001  # Learning rate
GAMMA = 0.95  # Discount factor
SIZE = 2.5  # micro b
RATE = 50  # micro b


class Robot:
    # def __init__(self, name, pos_x, pos_y, direction, off_decision, workload, resources, energy):
    def __init__(self, name, off_decision, workload, resources, energy):

        self.name = name
        # self.pos_x = pos_x
        # self.pos_y = pos_y
        # self.direction = direction
        self.off_decision = off_decision
        self.workload = workload  # lambda workload value λ: 0 --> 10
        self.resources = resources  # robot resources: res 0 --> 100: equivalent to MIPS in this case
        self.energy = energy

    def reset(self):
        # self.pos_x = random.randint(0, GRID_SIZE)
        # self.pos_y = random.randint(0, GRID_SIZE)
        self.energy = random.choice([0, 45,90])
        self.resources = random.choice([0, 45,90])
        self.workload = random.randint(1,3 )   # lambda workload value λ: 0 --> 10
        # state = [(self.pos_x, self.pos_y), self.resources, self.workload, self.energy]
        state = [self.workload, self.resources, self.energy]

        return state


    def get_state(self):
        # position = (self.pos_x, self.pos_y)
        available_resource = self.resources
        workload = self.workload
        energy = self.energy
        # return [position, available_resource, workload, energy]
        return [workload, available_resource, energy]

    def action(self):
        return [self.off_decision]
        # return [self.direction, self.off_decision]

    def random_action(self):
        # self.direction = random.choice(['l', 'r', 'u', 'd'])
        self.off_decision = random.choice(['1', '2', '3'])
        return self.off_decision  # Wrap the single action in a list




def get_new_state(state, action, robots):
    new_state = [s.copy() for s in state]  # Create a copy of the state to avoid modifying the original list
    print("Original state:", new_state)
    print("Action:", action)

    for i, robot in enumerate(robots):
        target_robot_index = int(action[i]) - 1  # Calculate the index of the target robot
        target_robot = robots[target_robot_index]  # Get the target robot object

        # Decrease the workload of the current robot
        new_state[i][0] -= 1
        print(f"Robot {i+1}: Workload decreased by 1")

        # Increase the workload of the target robot in the state
        new_state[target_robot_index][0] += 1
        print(f"Target Robot {target_robot.name}: Workload increased by 1")

    print("New state:", new_state)
    return new_state



def get_best_action(state, robots, Q_table):
    state_key = tuple(state)  # Convert the state to a tuple

    state_hash = hash_state(state_key)  # Hash the state key    # print("state key", state_key)
    state_actions = Q_table[state_hash]['actions']
    print("state actions", state_actions)
    if state_actions:
        best_action_key = max(state_actions, key=lambda action: state_actions[action]['Q-value'])
        best_action = state_actions[best_action_key]['vector']
        print("best action", best_action)
        return best_action
    else:
        action_vec = [robot.random_action() for robot in robots]
        print("action from else statement", action_vec)
        return action_vec


def reward(new_state):
    total_reward = 0

    for state in new_state:
        workload = state[0]
        print("workload", workload)
        resources = state[1]
        print("resources", resources)
        energy = state[2]
        print("energy", energy)

        if resources == 0 and energy == 0 and workload > 0:
            reward = workload * -100
        elif resources == 0 or energy == 0 and workload ==0:
            reward = -100
        elif workload == 0:
            reward = 10
        else:
            reward = resources / workload + (energy - workload * 5)
        print("single reward", reward)

        total_reward += reward
        print("total_reward", total_reward)

    return total_reward



def hash_state(state_key):
    state_bytes = pickle.dumps(state_key)
    state_hash = hashlib.sha256(state_bytes).hexdigest()
    return state_hash

def hash_state(action_key):
    action_bytes= pickle.dumps(action_key)
    actionhash = hashlib.sha256(action_bytes).hexdigest()
    return actionhash
def state_index(state, Q_table):
    state_key = tuple(state)  # Convert the state to a tuple
    state_hash = hash_state(state_key)  # Hash the state key    # print("state key", state_key)
    if state_hash in Q_table:
        index = Q_table[state_hash]['index']
        print("State found in the Q_table with index", index)
    else:
        index = len(Q_table)
        print("State not found in the Q_table. Added new index", index)

        Q_table[state_hash] = {
            'index': index,
            'vector': state,
            'actions': {}
        }
    # print("state:Q_table[state_key] :", Q_table[state_key])
    # print("actions of this state:", Q_table[state_key]['actions'])
    return state, state_hash


def action_index(action_vec, state, Q_table):
    state_key = tuple(state)  # Convert the state to a tuple
    state_hash = hash_state(state_key)  # Hash the state key    # print("state key", state_key)

    action_key= tuple(action_vec)  # Convert the state to a tuple
    action_hash= hash_state(action_key)  # Hash the state key    # print("state key", state_key)

    state_actions = Q_table[state_hash]['actions']

    if action_hash in state_actions:
        index = state_actions[action_hash]['index']
        print("Action found in the Q_table with index", index)
    else:
        index = len(state_actions)
        print("Action not found in the Q_table. Added new index", index)

        state_actions[action_hash] = {
            'index': index,
            'vector': action_vec,
            'Q-value': 0.0
        }
    # print("action:state_actions[action_key]: ", state_actions[action_key])
    return action_hash





def main(epsilon):
    # Initialize Q-table
    Q_table = {}

    num_agents = 3
    num_states = N_STATES
    num_actions = N_ACTIONS
    num_episodes = 100000000
    average_rewards = []  # List to store average rewards every 100 episodes

    print("num_states", num_states)
    for episode in range(num_episodes):
        step_counter = 0  # Counter for steps taken in the current episode

        robots = [
            Robot("1", ['1', '2', '3'], 0, 0, 0),
            Robot("2", ['1', '2', '3'], 0, 0, 0),
            Robot("3", ['1', '2', '3'], 0, 0, 0)
        ]

        max_steps = 10
        satisfied = True
        rewards = []

        state = [robot.reset() for robot in robots]
        print("random state new episode", state)
        state, state_key = state_index(state, Q_table)
        # print("current state ",state)


        while satisfied:
            if step_counter == max_steps:
                print("end of steps")
                break

            step_counter += 1

            print("Episode:", episode, "Step:", step_counter)

            state, state_key = state_index(state, Q_table)
            # print("current_state ", current_state)

            if np.random.rand() < epsilon:
                action_vec = [robot.random_action() for robot in robots]
                print("random action", action_vec)
            else:
                action_vec = get_best_action(state, robots, Q_table)
            action_key = action_index(action_vec, state, Q_table)

            new_state = get_new_state(state, action_vec, robots)

            state, new_state_key = state_index(new_state, Q_table)

            if not all(1 <= robot[0] <= 3 for robot in new_state):
                satisfied = False
                totreward = -100
                print("Penalty:", totreward)
            else:
                # totreward = sum([robot.reward(new_state) for robot in robots])
                totreward = reward(new_state)
                print("totreward", totreward)
                rewards.append(totreward)

                Q_table[state_key]['actions'][action_key]['Q-value'] = (1 - ALPHA) * Q_table[state_key]['actions'][action_key]['Q-value'] + ALPHA * (totreward + GAMMA * max(Q_table.get(new_state_key, {}).get('actions', {}).get(action_key, {}).get('Q-value', 0.0), 0.0))

            if not satisfied:
                break

        # total_reward = sum(rewards)  # Calculate the total reward for the episode
        # print("totaaaaaaaaaaaaaaaaal", total_reward)

        epsilon *= EPSILON_DECAY

        # Calculate average reward every 1000 episodes
        if episode % 1000 == 0:
            average_reward = sum(rewards[-1000:]) / 1000  # Calculate the average reward
            average_rewards.append(average_reward)
            rewards = []  # Reset the rewards list for the next 100 episodes

    # Plot average rewards every 1000 episodes
    episodes = range(1000, num_episodes + 1, 1000)
    plt.plot(episodes, average_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward per 1000 Episodes')
    plt.show()

if __name__ == "__main__":
    epsilon = EPSILON  # Initial exploration rate
    main(epsilon)
