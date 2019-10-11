from gym_minigrid.minigrid import MiniGridEnv
from gym_minigrid.minigrid import Grid
from gym_minigrid.minigrid import IDX_TO_OBJECT
from gym_minigrid.minigrid import OBJECT_TO_IDX
from gym_minigrid.minigrid import Ball
from gym_minigrid.minigrid import Wall


from typing import Tuple
from typing import List
import os
import imageio


class GridWorld(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(self, command: str, save_directory: str, size=8, agent_start_pos=(0, 0), agent_start_dir=0):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.mission = command
        self.num_available_objects = len(IDX_TO_OBJECT.keys())
        self.available_objects = set(OBJECT_TO_IDX.keys())
        self.save_directory = save_directory

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

    def save_situation(self, file_path: str) -> str:
        assert file_path.endswith('.png'), "Invalid file name passed to save_situation, must end with .png."
        save_location = os.path.join(self.save_directory, file_path)
        success = self.render().img.save(save_location)
        if not success:
            print("WARNING: image with name {} failed to save.".format(file_path))
            return ''
        else:
            return save_location

    def visualize_sequence(self, action_sequence: List[int]) -> str:
        """
        Save an image of each situation and make a gif out of the sequence to visualize the command of the
        environment.
        :param action_sequence: list of integers representing actions (as per Actions in minigrid.py).
        :return: directory where the images and gif are saved.
        """

        # Initialize directory with current command as its name.
        mission_dir = self.mission.replace(' ', '_')
        full_dir = os.path.join(self.save_directory, mission_dir)
        if not os.path.exists(full_dir):
            os.mkdir(full_dir)
        save_location = self.save_situation(os.path.join(mission_dir, 'initial.png'))
        filenames = [save_location]

        # Loop over actions and take them.
        for i, action_dir in enumerate(action_sequence):
            current_filename = os.path.join(mission_dir, 'situation_' + str(i) + '.png')

            # Stay.
            if action_dir == -1:
                save_location = self.save_situation(current_filename)
            # Move forward.
            else:
                self.agent_dir = action_dir
                self.step(self.actions.forward)
                save_location = self.save_situation(current_filename)
            filenames.append(save_location)

        # Make a gif of the action sequence.
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        movie_dir = os.path.join(self.save_directory, mission_dir)
        imageio.mimsave(os.path.join(movie_dir, 'movie.gif'), images, fps=5)
        return movie_dir
