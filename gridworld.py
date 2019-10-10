from gym_minigrid.minigrid import MiniGridEnv
from gym_minigrid.minigrid import Grid
from gym_minigrid.minigrid import IDX_TO_OBJECT
from gym_minigrid.minigrid import OBJECT_TO_IDX
from gym_minigrid.minigrid import Ball
from gym_minigrid.minigrid import Wall


from typing import Tuple
from typing import List


class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(self, command: str, size=8, agent_start_pos=(0, 0), agent_start_dir=0):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.mission = command
        self.num_available_objects = len(IDX_TO_OBJECT.keys())
        self.available_objects = set(OBJECT_TO_IDX.keys())

        super().__init__(grid_size=size, max_steps=4*size*size, see_through_walls=True)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    def place_random_object(self):
        raise NotImplementedError()

    def place_object(self, object_type: str, object_color: str, location: Tuple[int, int]):
        if object_type == "ball":
            object = Ball(object_color)
            self.place_obj(object, top=location, size=(1, 1))
        elif object_type == "wall":
            object = Wall(object_color)
            self.place_obj(object, top=location, size=(1, 1))

    def place_objects(self, objects: List[Tuple[str, str, Tuple[int, int]]]):
        for object_name, object_color, object_position in objects:
            self.place_object(object_name, object_color, object_position)
