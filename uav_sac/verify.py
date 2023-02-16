import numpy as np
from dataclasses import dataclass
import os
import math
from typing import List, Dict, Callable
from uav_sac.networks.conv2d_model import GaussianPolicy, QNetwork
from uav_sac.environments.simple2duav import Simple2DUAV, FullState
from uav_sac.environments.uav_explorer import PlaneEnv
from uav_sac.environments.belief2d import Belief2D
from uav_sac.sac import SAC
from uav_sac.training_config import training_config_from_json
from uav_sac.utils import load_random_animation
import torch
import copy
import argparse
import matplotlib.pyplot as plt
from matplotlib import font_manager
import json

font_dirs = ['/home/jaas224/fonts_for_figures']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

font = {'family': 'Times New Roman',
        'weight': 'bold',
        'size': 22}

plt.rc('font', **font)


def display_results(result_dict, n, title="Simple2DUAV Trajectory", display=False, save_path=None):
    start_image = result_dict['start_image']
    final_image = result_dict['final_image']
    trajectories = result_dict['trajectory']

    # plot the initial map
    mesh = plt.pcolormesh(np.array(start_image).T, cmap="RdYlGn", alpha=0.2)
    # cbar = plt.colorbar(mesh)
    # cbar.set_label("Value (Iz(s))")
    plt.title("Starting Value Map")
    plt.xlabel("Position (X)")
    plt.ylabel("Position (Y)")
    plt.xlim([0, Simple2DUAV.ydim])
    plt.ylim([0, Simple2DUAV.xdim])
    plt.clim(0, 1)
    if display:
        plt.show()
    elif save_path:
        plt.savefig(f"{save_path}startmap_{n}.png",
                    dpi=300,
                    bbox_inches="tight")
    plt.clf()

    # plot the trajectory
    mesh = plt.pcolormesh(np.array(final_image).T, cmap="RdYlGn", alpha=0.2)
    cbar = plt.colorbar(mesh)
    cbar.set_label("Confidence P(C(x,y))")
    plt.title(title)
    plt.xlabel("Position (X)")
    plt.ylabel("Position (Y)")
    plt.xlim([0, Simple2DUAV.ydim])
    plt.ylim([0, Simple2DUAV.xdim])
    plt.clim(0, 1)
    for i, p in enumerate(trajectories.values()):
        plt.scatter(p[0][1], p[0][0], marker='o')
        plt.plot(np.array(p).T[1], np.array(p).T[0], label=f"{i}")
    if display:
        plt.show()
    elif save_path:
        plt.savefig(f"{save_path}{n}.png", dpi=300, bbox_inches="tight")
    plt.clf()

class TurnTaker:
    def __init__(self, agent: SAC, planes: Dict[int, PlaneEnv]):
        self.agent = agent
        self.planes = copy.copy(planes)  # dict of planes

    def next_winner(self):
        if len(self.planes):
            qs = self.agent.get_vs([plane.normed_state for plane in self.planes.values()]) # pass all env states as batch
            winner = list(self.planes.keys())[np.argmax(qs)]
            return self.planes.pop(winner)
        return False

class SaveState:
    def __init__(self, plane: PlaneEnv):
        self.plane: PlaneEnv = copy.deepcopy(plane)
        self.actions: List[float] = []


def generate_agent_simulator(agent: SAC, horizon: int) -> Callable:
    '''
    the general algorithm for the open-loop control of the planes using turn taking
    we want to first guess the actions the planes will take and then commit to them 
    all at once together to simulate the real-time-ness of the actions
    '''
    def _turn_taking(planes: Dict[int, PlaneEnv]):
        episode_reward = 0
        crashed = 0
        done = False
        plane_trajs = {plane_id: [] for plane_id in planes.keys()}
        while not done:
            # Step 1: assemble the planes in order of the best next actions
            turn_taker = TurnTaker(agent, planes)
            save_states: Dict[int, SaveState] = {}
            while winner := turn_taker.next_winner():
                save_states[winner.uuid] = SaveState(winner)
                select_action = lambda: agent.select_action(winner.normed_state, evaluate=True)  # Sample action from policy
                winner.measure = winner.belief.measure  # cannot observe map directly, only belief
                # simulate the plane forward {horizon} steps
                for _ in range(horizon):
                    # Step 2: take actions for the best plane for a given horizon
                    action = select_action()
                    _, reward, plane_done, _ = winner.step(action)
                    episode_reward += reward
                    save_states[winner.uuid].actions.append(action)
                    if plane_done:
                        save_states.pop(winner.uuid)
                        if not math.isclose(winner.t, Simple2DUAV.maxtime, rel_tol=2*Simple2DUAV.dt):
                            crashed += 1
                        done = len(save_states) == 0
                        break
                    plane_trajs[winner.uuid].append([winner.plane.x, winner.plane.y])
            # Step 3: take planned actions for each plane simultaneously
            planes = {}
            for uuid,save_state in save_states.items():
                for step in range(horizon):
                    save_state.plane.step(save_state.actions[step])
                planes[uuid] = save_state.plane
        return plane_trajs, episode_reward, crashed
    return _turn_taking


def generate_greedy_simulator():
    def _run(planes):
        Simple2DUAV.dt = 0.4
        episode_reward = 0
        crashed = 0
        done = False
        plane_trajs = [[] for _ in range(len(planes))]
        action_set = np.arange(-Simple2DUAV.action_max+0.0000001, Simple2DUAV.action_max-0.0000001, math.pi/36)
        while not done:
            turns = [(i,p) for i,p in planes.items()]
            planes[list(planes.keys())[0]].belief.step(Simple2DUAV.dt) # step the map at the start of each turn
            for i, plane in turns:
                rs = []
                for a in action_set:
                    current_state = copy.deepcopy(plane.full_state)
                    _, r, _, _ = plane.step(a)
                    rs.append(r)
                    plane.reset_full_state_from(current_state)
                action = action_set[np.argmax(np.array(rs))]
                _, reward, plane_done, _ = plane.step(action)
                episode_reward += reward
                if plane_done:
                    planes.pop(i)
                    if not math.isclose(plane.t, plane.maxtime, rel_tol=2*Simple2DUAV.dt):
                        crashed += 1
                plane_trajs[i].append([plane.x, plane.y])
            done = len(planes) == 0
        Simple2DUAV.dt = 0.2
        return plane_trajs, episode_reward, crashed
    return _run

class FixedPolicy():
    def __init__(self, planes):
        self.planes = planes
        self.dirs = {
            plane: "down" if plane.y > plane.ydim/2 else "up" for plane in planes.values()
        }

    def select_action(self, plane):
        '''
        lawnmower pattern
        '''
        xpad = 4
        xmin = xpad
        xmax = Simple2DUAV.xdim - xpad
        right = -2*math.pi/18
        soft_right = -1*math.pi/36
        left = 2*math.pi/18
        soft_left = 1*math.pi/36
        if self.dirs[plane] == 'up' and plane.x > xmax and not math.isclose(plane.yaw, math.pi, rel_tol=0.05): # turn right
            action = right
        elif self.dirs[plane] == 'up' and plane.x < xmin and not math.isclose(plane.yaw, 0, rel_tol=0.05): # turn left
            action = left
        elif self.dirs[plane] == 'down' and plane.x > xmax and not math.isclose(plane.yaw, math.pi, rel_tol=0.05): # turn right
            action = right
        elif self.dirs[plane] == 'down' and plane.x < xmin and not math.isclose(plane.yaw, 0, rel_tol=0.05): # turn left
            action = left
        else:
            action = 0
            # keep near pi
            if plane.yaw > math.pi/2 and plane.yaw < math.pi: # left
                action = soft_left
            elif plane.yaw < 3*math.pi/2 and plane.yaw > math.pi: # right
                action = soft_right
            # keep near 0
            elif plane.yaw < math.pi/2 and plane.yaw > 0: # right
                action = soft_right
            elif plane.yaw > 3*math.pi/2 and plane.yaw < math.pi*2: # left
                action = soft_left
        return action

def generate_fixed_simulator():
    def _run(planes):
        episode_reward = 0
        crashed = 0
        done = False
        plane_trajs = [[] for _ in range(len(planes))]
        policy = FixedPolicy(planes)

        while not done:
            turns = [(i, p) for i, p in planes.items()]
            planes[list(planes.keys())[0]].belief.step(Simple2DUAV.dt)  # step the map at the start of each turn
            for i, plane in turns:
                next_state, reward, plane_done, _ = plane.step(policy.select_action())
                episode_reward += reward
                if plane_done:
                    planes.pop(i)
                    if not math.isclose(plane.t, plane.maxtime, rel_tol=2*Simple2DUAV.dt):
                        crashed += 1
                plane_trajs[i].append([plane.x, plane.y])
            done = len(planes) == 0
        return plane_trajs, episode_reward, crashed
    return _run

class Report():
    def __init__(self):
        self.rewards = []
        self.error_deltas = []
        self.crash_count = 0

    def append(self, reward: float, error: List[float], crash: bool):
        self.rewards.append(reward)
        self.error_deltas.append(error[0]-error[1])
        if crash:
            self.crash_count += 1

    @property
    def num_reports(self):
        return len(self.rewards)

    @property
    def average_delta_error(self):
        return sum(self.error_deltas)/self.num_reports

    @property
    def average_reward(self):
        return sum(self.rewards)/self.num_reports

    @property
    def stddev_reward(self):
        return (sum([((x - self.average_reward) ** 2) for x in self.rewards]) / self.num_reports) ** 0.5

    @property
    def crash_rate(self):
        return self.crash_count/self.num_reports


def verify_models(cfg, simulator, save_path=False, display=False) -> Report:
    envs = []
    for e in range(cfg.episode.num_planes):
        env = Simple2DUAV(load_random_animation(), cfg)
        envs.append(env)

    report = Report()
    results = []
    for n in range(cfg.episode.verification_episodes):
        # reset the envs
        result = {}
        envs[0].reset()
        belief_space = envs[0].belief
        animation = load_random_animation()
        for e in envs:
            e.reset(animation=animation, belief_space=belief_space)

        if n % 2 == 0:
            result['start_image'] = copy.copy(np.take(animation[-1], 2, axis=2).T).tolist()

        planes = {env.uuid: env for env in envs}  # preserves index even after deletion
        start_error = envs[0].error
        plane_trajs, episode_reward, crash = simulator(planes)
        end_error = envs[0].error
        report.append(episode_reward, (start_error, end_error), crash)

        if n % 2 == 0:
            belief = np.take(envs[0].belief.img, 2, axis=2)
            result['final_image'] = copy.copy(belief.T).tolist()
            result['score'] = episode_reward
            result['trajectory'] = copy.copy(plane_trajs)
            results.append(result)
            display_results(result, n, display=display, save_path=save_path)
    with open(f"{save_path}results.json", 'w') as outfile:
        json.dump(results, outfile)
    return report

def compare_simulators(gamma, simulators, save_path=False, display=False):
    env = Simple2DUAV(gamma)

    results = []
    for n in range(5):
        # reset the envs
        result = {}
        belief_space = env.belief
        env.reset(belief_space)

        belief = np.take(belief_space.img, 2, axis=2)
        result['start_image'] = copy.copy(belief.T).tolist()
        start_state = copy.copy(env.state).tolist()

        for sim_name, simulator in simulators.items():
            env.reset_state_from(np.array(start_state))
            plane = {0: env}
            plane_trajs, episode_reward, _ = simulator(plane)

            belief = np.take(belief_space.img, 2, axis=2)
            result['final_image'] = copy.copy(belief.T).tolist()
            result['score'] = episode_reward
            result['trajectory'] = copy.copy(plane_trajs)
            results.append(result)
            display_results(result, n, display=display, save_path=f"{save_path}{sim_name}_", title=f"{sim_name} Trajectory")

def main():
    parser = argparse.ArgumentParser(description='Pytorch Simple2DUAV Model Verification Args')
    parser.add_argument('-c', '--cfg_file', required=True,
                        help='config file for the training run, required for model creation')
    parser.add_argument('-m', '--model_path', required=True,
                        help='model to run verification on')
    args = parser.parse_args()
    cfg = training_config_from_json(args.cfg_file)

    env = Simple2DUAV(load_random_animation(), cfg)
    agent = SAC(env.obs_state_len, env.action_space, cfg.hyperparams,
                map_input=(env.observation_space.shape[2],
                           env.observation_space.shape[0]-1,
                           env.observation_space.shape[1]))
    agent.load_checkpoint(args.model_path)

    simulator = generate_agent_simulator(agent, cfg.horizon)
    report = verify_models(args.hyper_params.gamma, 1, 10, simulator, save_path="current_verification/", display=False)
    print(f"Agent average reward over 10 runs {report.average_reward}")

if __name__ == '__main__':
    main()