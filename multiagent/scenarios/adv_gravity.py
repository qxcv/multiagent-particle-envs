import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class GravityWorld(World):
    def __init__(self):
        super().__init__()

        # add agents
        self.agents = [Agent() for i in range(1)]

        control_agent = self.agents[0]
        control_agent.name = 'controller'
        control_agent.collide = True
        control_agent.silent = True
        # slow acceleration & low friction
        control_agent.damping = 0.01
        control_agent.accel = 0.2

        # add landmarks
        self.landmarks = [Landmark() for i in range(2)]
        sun_landmark, goal_landmark = self.landmarks

        sun_landmark.name = 'sun'
        sun_landmark.collide = False
        sun_landmark.movable = False
        sun_landmark.gravity_coeff = 0.05
        sun_landmark.size = 0.2

        goal_landmark.name = 'goal'
        goal_landmark.collide = False
        goal_landmark.movable = False
        goal_landmark.size = 0.2
        goal_landmark.gravity_coeff = 0.05

        self.time = 0
        self.max_steps = 500

    def reset(self):
        control_agent = self.agents[0]
        control_agent.color = np.array([0.25, 0.25, 0.25])
        sun, goal = self.landmarks
        sun.color = np.array([0.75, 0.25, 0.25])
        goal.color = np.array([0.25, 0.75, 0.25])

        control_agent.state.p_pos = np.array([-1, 0]) \
            + np.random.uniform(-0.01, 0.01, self.dim_p)
        control_agent.state.p_vel = np.array([0.1, 0.25]) \
            + np.random.uniform(-0.01, 0.01, self.dim_p)
        control_agent.state.c = np.zeros(self.dim_c)

        sun.state.p_pos = np.array([0, 0])
        sun.state.p_vel = np.zeros(self.dim_p)
        goal.state.p_pos = np.array([1, 0])
        goal.state.p_vel = np.zeros(self.dim_p)

    def reward(self, agent):
        control_agent = self.agents[0]
        sun_landmark, goal_landmark = self.landmarks
        goal_dist = np.linalg.norm(control_agent.state.p_pos - goal_landmark.state.p_pos)
        sun_dist = np.linalg.norm(control_agent.state.p_pos - sun_landmark.state.p_pos)
        hit_goal = self._entities_overlap(control_agent, goal_landmark) + 0.0
        hit_sun = self._entities_overlap(control_agent, sun_landmark) + 0.0
        rew = self.max_steps * hit_goal - self.max_steps * hit_sun + min(1, 1 - 0.4 * goal_dist)
        print('rew =', rew, 'and vel =', np.linalg.norm(control_agent.state.p_vel))  # XXX

        # now we return the actual reward for the control agent, or negative
        # reward for the adversary
        if agent is self.agents[0]:
            return rew
        elif agent is self.agents[1]:
            # adversary
            return -rew
        raise Exception(
            "Unknown agent %s (I have %d known agents, and should have 2)" %
            (agent, len(self.agents)))

    def step(self):
        rv = super().step()
        # advance the clock
        self.time += 1
        return rv

    def _entities_overlap(self, ent_a, ent_b):
        # assumes circles (safe for now)
        dist = np.linalg.norm(ent_a.state.p_pos - ent_b.state.p_pos)
        min_sep = ent_a.size + ent_b.size
        return dist < min_sep

    def done(self):
        if self.time > self.max_steps:
            return True
        control_agent = self.agents[0]
        sun_landmark, goal_landmark = self.landmarks
        return self._entities_overlap(control_agent, sun_landmark) \
            or self._entities_overlap(control_agent, goal_landmark)


class Scenario(BaseScenario):
    def make_world(self):
        world = GravityWorld()
        world.reset()
        return world

    def reset_world(self, world):
        return world.reset()

    def reward(self, agent, world):
        return world.reward(agent)

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)

    def done(self, agent, world):
        return world.done()
