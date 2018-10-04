from multiagent.scenarios import adv_race


class Scenario(adv_race.Scenario):
    """Same as race scenario, but with adversary modified to make them unable
    to accelerate (like adv_gravity_nulladv)."""

    def make_world(self):
        world = adv_race.RaceWorld(nulladv=True, transfer=True)
        world.reset()
        return world
