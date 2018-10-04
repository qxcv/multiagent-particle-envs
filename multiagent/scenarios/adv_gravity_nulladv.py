from multiagent.scenarios.adv_gravity import Scenario as GravityScenario


class Scenario(GravityScenario):
    """Same as GravityScenario, but with adversary modified to make them unable
    to accelerate. Useful for testing and for training baselines (e.g. the
    plain DDPG baseline with no adversary)."""

    def make_world(self):
        world = super().make_world()
        #  turn off adversary's ability to accelerate at all
        world.agents[1].accel = 0
        return world
