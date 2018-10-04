from multiagent.scenarios import adv_gravity


class Scenario(adv_gravity.Scenario):
    """Same as GravityScenario, but with adversary modified to make them unable
    to accelerate. Useful for testing and for training baselines (e.g. the
    plain DDPG baseline with no adversary)."""

    def make_world(self):
        world = adv_gravity.GravityWorld(nulladv=True, transfer=True)
        world.reset()
        return world
