import numpy as np
from multiagent.core import World, Agent, Landmark, RectangularLandmark
from multiagent.scenario import BaseScenario


class CliffWorld(World):
    def __init__(self):
        super().__init__()

        # add agents
        self.agents = [Agent() for i in range(2)]
        controller, adversary = self.agents

        controller.name = 'controller'
        # don't bother colliding; there's nothing to collide with here
        controller.collide = False
        controller.silent = True
        # high acceleration, lowish friction
        controller.damping = 0.01
        controller.accel = 0.6

        adversary.name = 'adversary'
        adversary.collide = False
        adversary.silent = True
        # damping doesn't matter, so I'll just set to high value
        adversary.damping = 0.8
        # adversary can actually do a lot, but not as much as controller
        adversary.accel = 3 / 4.0 * controller.accel

        # add landmarks
        self.landmarks = [RectangularLandmark(), Landmark()]
        cliff, goal = self.landmarks

        cliff.name = 'cliff'
        cliff.collide = False
        cliff.movable = False
        cliff.rect_wh = np.array([8, 1])

        goal.name = 'goal'
        goal.collide = False
        goal.movable = False
        goal.size = 0.1

        self.time = 0
        # possibly too long; oh well
        self.max_steps = 80

    def reset(self):
        self.time = 0

        controller, adversary = self.agents

        controller.color = np.array([0.25, 0.25, 0.25])
        # start at a slightly initial position
        controller.state.p_pos = np.array([-1, -0.08]) \
            + np.random.uniform(-0.01, 0.01, self.dim_p)
        # don't move initially
        controller.state.p_vel = np.array([0, 0])
        controller.state.c = np.zeros(self.dim_c)

        # adversary sits inside the controller
        adversary.color = np.array([0.75, 0.25, 0.25])
        adversary.state.p_pos = controller.state.p_pos.copy()
        adversary.state.p_vel = controller.state.p_vel.copy()
        adversary.state.c = np.zeros(self.dim_c)

        cliff, goal = self.landmarks
        # like lava :)
        cliff.color = (1.0, 0.05, 0.05)
        # goal is green this time
        goal.color = np.array([0.25, 0.75, 0.25])

        cliff.state.p_pos = np.array([0, 0.5])
        cliff.state.p_vel = np.zeros(self.dim_p)
        goal.state.p_pos = np.array([1, -0.1])
        goal.state.p_vel = np.zeros(self.dim_p)

    def apply_action_force(self, p_force):
        controller, adversary = self.agents
        # add up controller and adversary controls & noise
        controller_noise = np.random.randn(*controller.action.u.shape) \
            * controller.u_noise if controller.u_noise else 0.0
        adversary_noise = np.random.randn(*adversary.action.u.shape) \
            * adversary.u_noise if adversary.u_noise else 0.0
        p_force[0] = controller.action.u + adversary.action.u \
            + controller_noise + adversary_noise
        # adversary never moves
        p_force[1] = np.zeros(self.dim_p)
        return p_force

    def reward(self, agent):
        control_agent = self.agents[0]
        cliff_landmark, goal_landmark = self.landmarks
        goal_dist = np.linalg.norm(control_agent.state.p_pos -
                                   goal_landmark.state.p_pos)
        cliff_dist = np.linalg.norm(control_agent.state.p_pos -
                                  cliff_landmark.state.p_pos)
        hit_goal = self._entities_overlap(control_agent, goal_landmark) + 0.0
        hit_cliff = self._entities_overlap(control_agent, cliff_landmark) + 0.0
        # this is almost exactly the same reward function as the gravity_adv
        # problem
        rew = self.max_steps * hit_goal - self.max_steps * hit_cliff \
           - min(0.2 * goal_dist, 1)

        # return actual reward for controller, negative for adversary
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
        # move adversary into same place as agent
        controller, adversary = self.agents
        adversary.state.p_pos = controller.state.p_pos.copy()
        adversary.state.p_vel = controller.state.p_vel.copy()
        # advance the clock
        self.time += 1
        return rv

    def _entities_overlap(self, ent_a, ent_b):
        # assumes circles (safe for now)
        shapes = {ent_a.shape, ent_b.shape}
        if shapes == {'circle'}:
            dist = np.linalg.norm(ent_a.state.p_pos - ent_b.state.p_pos)
            min_sep = ent_a.size + ent_b.size
            return dist < min_sep
        if shapes == {'circle', 'rectangle'}:
            if ent_a.shape == 'rectangle':
                # ensure ent_a is always the circle
                ent_a, ent_b = ent_b, ent_a
            displacement = _circ_rect_disp(ent_a, ent_b)
            col_dist = np.linalg.norm(displacement)
            return col_dist < ent_a.size
        raise ValueError("Don't know how to do %s collision" %
                         "-".join(map(str, shapes)))

    def done(self, agent):
        # we ignore agent arg because termination condition is the same for all agents
        if self.time > self.max_steps:
            return True
        control_agent = self.agents[0]
        cliff_landmark, goal_landmark = self.landmarks
        return self._entities_overlap(control_agent, cliff_landmark) \
            or self._entities_overlap(control_agent, goal_landmark)

def _circ_rect_disp(ent_a, ent_b):
    # displacement between rectangle and circle
    assert ent_a.shape == 'circle' and ent_b.shape == 'rectangle'
    cxy = ent_a.state.p_pos
    cx, cy = cxy
    radius = ent_a.size
    rx, ry = ent_b.state.p_pos
    w, h = ent_b.rect_wh
    left, right = rx - w / 2.0, ry + w / 2.0
    top, bot = ry + h / 2.0, ry - h / 2.0
    # find distance to collision point (nearest point inside rect) &
    # check if it's less than radius
    collide_point = np.array([
        np.clip(cx, left, right),
        np.clip(cy, bot, top)
    ])
    return collide_point - cxy


class Scenario(BaseScenario):
    def make_world(self):
        world = CliffWorld()
        world.reset()
        return world

    def reset_world(self, world):
        return world.reset()

    def reward(self, agent, world):
        return world.reward(agent)

    def observation(self, agent, world):
        entity_pos = []
        agent = world.agents[0]
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # also compute displacement from big cliff rectangle
        cliff = world.landmarks[0]
        rect_disp = _circ_rect_disp(agent, cliff)
        rv = np.concatenate([agent.state.p_vel, rect_disp] + entity_pos)
        return rv

    def done(self, agent, world):
        rv = world.done(agent)
        return rv
