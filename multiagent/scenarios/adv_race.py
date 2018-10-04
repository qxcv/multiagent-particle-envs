import numpy as np
from multiagent.core import Agent, RectangularLandmark
from multiagent.scenario import BaseScenario
from multiagent.scenarios.adv_cliff import CliffWorld, _circ_rect_disp


class RaceWorld(CliffWorld):
    def __init__(self, *, nulladv=False, transfer=False):
        super().__init__()

        if transfer:
            assert nulladv, "can't do transfer without nulladv"

        # add agents
        self.agents = [Agent() for i in range(2)]
        controller, adversary = self.agents

        controller.name = 'controller'
        # don't bother colliding; there's nothing to collide with here
        controller.collide = False
        controller.silent = True
        # train environment has high acceleration, medium friction
        controller.damping = 0.2
        controller.accel = 1
        if transfer:
            # test environment has control noise and lower acceleration
            controller.accel = 0.8
            controller.u_noise = 0.5

        adversary.name = 'adversary'
        adversary.collide = False
        adversary.silent = True
        # damping doesn't matter, so I'll just set to high value
        adversary.damping = 0.8
        # adversary can do half as much as controller
        if nulladv:
            adversary.accel = 0.0
        else:
            adversary.accel = 1 / 2.0 * controller.accel

        # add landmarks
        self.landmarks = [RectangularLandmark()]
        cliff, = self.landmarks

        # this time cliff is in the middle of the map & isn't as large
        cliff.name = 'cliff'
        cliff.collide = False
        cliff.movable = False
        cliff.rect_wh = np.array([1.0, 0.2])

        self.time = 0
        # long num of steps so that we can actually get somewhere
        self.max_steps = 100

    def reset(self):
        self.time = 0

        controller, adversary = self.agents

        controller.color = np.array([0.25, 0.25, 0.25])
        # start at a slightly jittered initial position on top or bottom
        mult = np.random.choice([-1, 1])
        controller.state.p_pos \
            = np.array([-0.5 * mult, mult * (0.2 / 2 + 0.15)]) \
            + np.random.uniform(-0.01, 0.01, self.dim_p)
        # start moving in right direction from top or bottom edge
        controller.state.p_vel = np.array([mult * 0.6, 0]) \
            + np.random.uniform(-0.01, 0.01, self.dim_p)
        controller.state.c = np.zeros(self.dim_c)

        # adversary sits inside the controller
        adversary.color = np.array([0.75, 0.25, 0.25])
        adversary.state.p_pos = controller.state.p_pos.copy()
        adversary.state.p_vel = controller.state.p_vel.copy()
        adversary.state.c = np.zeros(self.dim_c)

        cliff, = self.landmarks
        cliff.color = (1.0, 0.05, 0.05)

        cliff.state.p_pos = np.array([0, 0])
        cliff.state.p_vel = np.zeros(self.dim_p)

    def reward(self, agent):
        controller = self.agents[0]
        cliff, = self.landmarks
        hit_cliff = self._entities_overlap(controller, cliff) + 0.0
        cliff_disp = _circ_rect_disp(controller, cliff)
        cliff_dist = np.linalg.norm(cliff_disp)
        # compute velocity in right direction
        cliff_sign = np.sign(cliff_disp)
        if np.all(cliff_sign) or not np.any(cliff_sign):
            # don't compute forward velocity on corners or inside rect
            forward_vel = 0.0
        else:
            # rotate sign counterclockwise by 90 degrees & project onto agent's
            # velocity
            rot_mat = np.array([
                # 90 degree CCW roation
                [0, -1],
                [1, 0]
            ])
            agent_vel = controller.state.p_vel
            forward_vel = np.dot(agent_vel, rot_mat @ cliff_sign)
        # reward for going along right direction, penalise for hitting cliff,
        # slightly penalise for going far away from cliff
        rew = 5 * forward_vel - self.max_steps * hit_cliff \
            - 0.1 * max(cliff_dist - 0.4, 0) ** 2

        # return actual reward for controller, negative for adversary
        if agent is self.agents[0]:
            return rew
        elif agent is self.agents[1]:
            # adversary
            return -rew
        raise Exception(
            "Unknown agent %s (I have %d known agents, and should have 2)" %
            (agent, len(self.agents)))

    def done(self, agent):
        # we ignore agent arg because termination condition is the same for all
        # agents
        if self.time > self.max_steps:
            return True
        control_agent = self.agents[0]
        cliff_landmark, = self.landmarks
        return self._entities_overlap(control_agent, cliff_landmark)


class Scenario(BaseScenario):
    def make_world(self):
        world = RaceWorld()
        world.reset()
        return world

    def reset_world(self, world):
        return world.reset()

    def reward(self, agent, world):
        return world.reward(agent)

    def benchmark_data(self, agent, world):
        if agent is world.agents[1]:
            # don't record anything for adversary
            return
        # for agent, record (1) reward, (2) whether they hit cliff
        rew = world.reward(agent)
        cliff = world.landmarks[0]
        hit_cliff = world._entities_overlap(agent, cliff)
        rv = (rew, bool(hit_cliff))
        return rv

    def observation(self, agent, world):
        entity_pos = []
        agent = world.agents[0]
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # include displacement from big cliff rectangle
        cliff = world.landmarks[0]
        rect_disp = _circ_rect_disp(agent, cliff)
        rv = np.concatenate([agent.state.p_vel, rect_disp] + entity_pos)
        return rv

    def done(self, agent, world):
        rv = world.done(agent)
        return rv
