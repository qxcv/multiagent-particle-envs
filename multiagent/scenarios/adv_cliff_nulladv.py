from multiagent.scenarios.adv_cliff import Scenario as CliffScenario


class Scenario(CliffScenario):
    """Same as cliff scenario, but with adversary modified to make them unable
    to accelerate (like adv_gravity_nulladv)."""

    def make_world(self):
        world = super().make_world()
        world.agents[1].accel = 0
        return world
