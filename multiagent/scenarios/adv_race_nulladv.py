from multiagent.scenarios.adv_race import Scenario as RaceScenario


class Scenario(RaceScenario):
    """Same as race scenario, but with adversary modified to make them unable
    to accelerate (like adv_gravity_nulladv)."""

    def make_world(self):
        world = super().make_world()
        world.agents[1].accel = 0
        return world
