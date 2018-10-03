#!/usr/bin/env python
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import time
import numpy as np

from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        '-s',
        '--scenario',
        default='simple.py',
        help='Path of the scenario Python script.')
    parser.add_argument(
        '--fps',
        type=float,
        default=20.0,
        help='frames per second to render')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(
        world,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
        done_callback=scenario.done,
        info_callback=None,
        shared_viewer=False)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env, i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    last_frame = time.time()
    done = False
    delay = 1 / args.fps
    step = 1
    traj_rew = np.zeros(env.n)
    while not done:
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        # render all agent views
        env.render()
        elapsed = time.time() - last_frame
        time.sleep(max(0, delay - elapsed))
        last_frame = time.time()
        done = all(done_n)
        # display rewards
        print("Step %d" % step)
        for agent in env.world.agents:
           print(agent.name + " reward: %0.3f" % env._get_reward(agent))
        step += 1
        traj_rew += reward_n
    print('Done! Trajectory reward %s' % traj_rew)
    print('Use Enter or Ctrl+D to exit.')
    input()
