from multiagent.scenarios import adv_cliff


class Scenario(adv_cliff.Scenario):
    """Same as cliff scenario, but with adversary modified to make them unable
    to accelerate (like adv_gravity_nulladv)."""

    def make_world(self):
        world = adv_cliff.CliffWorld(nulladv=True, transfer=True)
        world.reset()
        return world
