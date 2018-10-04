import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class GravityWorld(World):
    def __init__(self, *, transfer=False, nulladv=False):
        super().__init__()

        if transfer:
            assert nulladv, "transfer ==> nulladv (or at least it should)"

        # add agents
        self.agents = [Agent() for i in range(2)]
        controller, adversary = self.agents

        controller.name = 'controller'
        controller.collide = True
        controller.silent = True
        # slow acceleration & low friction
        controller.damping = 0.0
        controller.accel = 0.3
        # transfer environment adds a small amount of control noise
        if transfer:
            controller.u_noise = 0.03

        adversary.name = 'adversary'
        adversary.collide = False
        adversary.silent = True
        # damping doesn't matter, so I'll just set to high value
        adversary.damping = 0.8
        if not nulladv:
            # acceleration is almost that of the controller to make things hard
            adversary.accel = 0.6 * controller.accel
        else:
            # nulladv nerfs adversary
            adversary.accel = 0.0

        # add landmarks
        self.landmarks = [Landmark() for i in range(2)]
        sun, goal = self.landmarks

        sun.name = 'sun'
        sun.collide = False
        sun.movable = False
        sun.gravity_coeff = 0.15
        sun.size = 0.2
        if transfer:
            # transfer increases gravitational force of sun by 66%
            sun.gravity_coeff = 0.25

        goal.name = 'goal'
        goal.collide = False
        goal.movable = False
        goal.size = 0.14
        goal.gravity_coeff = 0.15

        self.time = 0
        self.max_steps = 140

    def reset(self):
        self.time = 0

        controller, adversary = self.agents

        controller.color = np.array([0.25, 0.25, 0.25])
        controller.state.p_pos = np.array([-1, 0]) \
            + np.random.uniform(-0.01, 0.01, self.dim_p)
        # bias velocity to go up since DDPG can't handle split policy
        controller.state.p_vel = np.array([0, 0.1]) \
            + np.random.uniform([-0.01, -0.01], [0.01, 0.01], self.dim_p)
        controller.state.c = np.zeros(self.dim_c)

        # adversary sits inside the controller at all times (for visual
        # cleanness)
        adversary.color = np.array([0.75, 0.25, 0.25])
        adversary.state.p_pos = controller.state.p_pos.copy()
        adversary.state.p_vel = controller.state.p_vel.copy()
        adversary.state.c = np.zeros(self.dim_c)

        sun, goal = self.landmarks
        sun.color = np.array([1.0, 0.65, 0.0])
        goal.color = np.array([0.0, 0.45, 0.75])

        sun.state.p_pos = np.array([0, 0])
        sun.state.p_vel = np.zeros(self.dim_p)
        goal.state.p_pos = np.array([1, 0])
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

    def _hit_goal_sun(self):
        control_agent = self.agents[0]
        sun_landmark, goal_landmark = self.landmarks
        hit_goal = self._entities_overlap(control_agent, goal_landmark) + 0.0
        hit_sun = self._entities_overlap(control_agent, sun_landmark) + 0.0
        return hit_goal, hit_sun

    def reward(self, agent):
        control_agent = self.agents[0]
        sun_landmark, goal_landmark = self.landmarks
        goal_dist = np.linalg.norm(
            control_agent.state.p_pos - goal_landmark.state.p_pos)
        hit_goal, hit_sun = self._hit_goal_sun()
        hit_sun = self._entities_overlap(control_agent, sun_landmark) + 0.0
        rew = self.max_steps * hit_goal - self.max_steps * hit_sun \
            - min(0.2 * goal_dist, 1)

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
        # move adversary into same place as agent
        controller, adversary = self.agents
        adversary.state.p_pos = controller.state.p_pos.copy()
        adversary.state.p_vel = controller.state.p_vel.copy()
        # advance the clock
        self.time += 1
        return rv

    def _entities_overlap(self, ent_a, ent_b):
        # assumes circles (safe for now)
        dist = np.linalg.norm(ent_a.state.p_pos - ent_b.state.p_pos)
        min_sep = ent_a.size + ent_b.size
        return dist < min_sep

    def done(self, agent):
        # we ignore agent arg because termination condition is the same for all
        # agents
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
        # get positions of all landmarks in this *first agent's* reference
        # frame this means that both agents get the same observation
        entity_pos = []
        agent = world.agents[0]
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)

    def benchmark_data(self, agent, world):
        if agent is world.agents[1]:
            # don't record anything for adversary
            return
        # for agent, record (1) reward, (2) whether they hit goal, (3) whether
        # they hit sun
        rew = world.reward(agent)
        hit_goal, hit_sun = world._hit_goal_sun()
        rv = (rew, bool(hit_goal), bool(hit_sun))
        return rv

    def done(self, agent, world):
        rv = world.done(agent)
        return rv
