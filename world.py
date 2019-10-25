from collections import namedtuple
import itertools
import os
import numpy as np
from typing import Tuple
from typing import List
from typing import Dict
from gym_minigrid.minigrid import MiniGridEnv
from gym_minigrid.minigrid import Grid
from gym_minigrid.minigrid import IDX_TO_OBJECT
from gym_minigrid.minigrid import OBJECT_TO_IDX
from gym_minigrid.minigrid import Circle
from gym_minigrid.minigrid import Square
from gym_minigrid.minigrid import Cylinder
from gym_minigrid.minigrid import DIR_TO_VEC
import imageio

from helpers import one_hot
from helpers import generate_possible_object_names

SemType = namedtuple("SemType", "name")
Position = namedtuple("Position", "column row")
Object = namedtuple("Object", "size color shape")
PositionedObject = namedtuple("PositionedObject", "object position vector", defaults=(None, None, None))
Variable = namedtuple("Variable", "name sem_type")
fields = ("action", "is_transitive", "manner", "adjective_type", "noun")
Weights = namedtuple("Weights", fields, defaults=(None, ) * len(fields))

ENTITY = SemType("noun")
COLOR = SemType("color")
SIZE = SemType("size")
EVENT = SemType("verb")

Direction = namedtuple("Direction", "name")
NORTH = Direction("north")
SOUTH = Direction("south")
WEST = Direction("west")
EAST = Direction("east")
FORWARD = Direction("forward")

Action = namedtuple("Action", "name")
PICK_UP = Action("pick_up")
MOVE_FORWARD = Action("move_forward")
STAY = Action("stay")
DROP = Action("drop")

DIR_TO_INT = {
    NORTH: 3,
    SOUTH: 1,
    WEST: 2,
    EAST: 0
}

INT_TO_DIR = {direction_int: direction for direction, direction_int in DIR_TO_INT.items()}

ACTION_TO_INT = {
    STAY: -1,
    MOVE_FORWARD: 2,
    PICK_UP: 3,
    DROP: 4
}

SIZE_TO_INT = {
    "small": 1,
    "average": 2,
    "big": 3
}

Command = namedtuple("Command", "action event")
UNK_TOKEN = 'UNK'


class Term(object):
    """
    Holds terms that can be parts of logical forms and take as arguments variables that the term can operate over.
    E.g. for the phrase 'Brutus stabs Caesar' the term is stab(B, C) which will be represented by the string
    "(stab B:noun C:noun)".
    """

    def __init__(self, function: str, args: tuple, weights=None, meta=None, specs=None):
        self.function = function
        self.arguments = args
        self.weights = weights
        self.meta = meta
        self.specs = specs

    def replace(self, var_to_find: Variable, replace_by_var: Variable):
        """Find a variable `var_to_find` the arguments and replace it by `replace_by_var`."""
        return Term(
            function=self.function,
            args=tuple(replace_by_var if variable == var_to_find else variable for variable in self.arguments),
            specs=self.specs,
            meta=self.meta
        )

    def to_predicate(self, predicate: dict):
        assert self.specs is not None
        output = self.function
        if self.specs.noun:
            predicate["noun"] = output
        elif self.specs.adjective_type == SIZE:
            predicate["size"] = output
        elif self.specs.adjective_type == COLOR:
            predicate["color"] = output

    def __repr__(self):
        parts = [self.function]
        for variable in self.arguments:
            parts.append("{}:{}".format(variable.name, variable.sem_type.name))
        return "({})".format(" ".join(parts))


class LogicalForm(object):
    """
    Holds neo-Davidsonian-like logical forms (http://ling.umd.edu//~alxndrw/LectureNotes07/neodavidson_intro07.pdf).
    An object LogicalForm(variables=[x, y, z], terms=[t1, t2]) may represent
    lambda x, y, z: and(t1(x, y, z), t2(x, y, z)) (depending on which terms involve what variables).
    """

    def __init__(self, variables: Tuple[Variable], terms: Tuple[Term]):
        self.variables = variables
        self.terms = terms
        if len(variables) > 0:
            self.head = variables[0]

    def bind(self, bind_var: Variable):
        """
        Bind a variable to its head, e.g for 'kick the ball', 'kick' is the head and 'the ball' will be bind to it.
        Or in the case of NP -> JJ NP, bind the JJ (adjective) to the head of the noun-phrase.
        E.g. 'the big red square', bind 'big' to 'square'.
        :param bind_var:
        :return:
        """
        sub_var, variables_out = self.variables[0], self.variables[1:]
        # assert sub_var.sem_type == bind_var.sem_type
        terms_out = [term.replace(sub_var, bind_var) for term in self.terms]
        return LogicalForm(variables=(bind_var,) + variables_out, terms=tuple(terms_out))

    def select(self, variables: list, exclude=frozenset()):
        """Select and return the sub-logical form of the variables in the variables list."""
        queue = list(variables)
        used_vars = set()
        terms_out = []
        while len(queue) > 0:
            var = queue.pop()
            deps = [term for term in self.terms if term.function not in exclude and term.arguments[0] == var]
            for term in deps:
                terms_out.append(term)
                used_vars.add(var)
                for variable in term.arguments[1:]:
                    if variable not in used_vars:
                        queue.append(variable)

        vars_out = [var for var in self.variables if var in used_vars]
        terms_out = list(set(terms_out))
        return LogicalForm(tuple(vars_out), tuple(terms_out))

    def to_predicate(self, objects):
        assert len(self.variables) == 1
        predicate = {"noun": "", "size": "", "color": ""}
        [term.to_predicate(predicate) for term in self.terms]
        object_str = predicate["size"]
        if predicate["color"]:
            object_str += ' ' + predicate["color"]
        object_str += ' ' + predicate["noun"]
        object_str = object_str.strip()
        if object_str in objects.keys():
            object_locations = objects[object_str]
        else:
            object_locations = {}

        return object_locations

    def __repr__(self):
        return "LF({})".format(" ^ ".join([repr(term) for term in self.terms]))


class Situation(object):
    def __init__(self, grid_size: int, agent_position: Position, agent_direction: Direction,
                 placed_objects: List[PositionedObject], carrying=None):
        self.grid_size = grid_size
        self.agent_pos = agent_position  # position is [col, row] (i.e. [x-axis, y-axis])
        self.agent_direction = agent_direction
        self.placed_objects = placed_objects
        self.carrying = carrying  # The object the agent is carrying

    def to_dict(self):
        return {
            "agent_position": Position(column=self.agent_pos[0], row=self.agent_pos[1]),
            "agent_direction": self.agent_direction,
            "grid_size": self.grid_size,
            # TODO: this has to be list of (Object(size, color, shape), Position(row, col))
            "objects": self.placed_objects,
            "carrying": self.carrying
        }


class ObjectVocabulary(object):
    """
    Constructs an object vocabulary. Each object will be calculated by the following:
    [size color shape] and where size is on an ordinal scale of 1 (smallest) to 4 (largest),
    colors and shapes are orthogonal vectors [0 1] and [1 0] and the result is a concatenation:
    e.g. the biggest red circle: [4 0 1 0 1], the smallest blue square: [1 1 0 1 0]
    """

    def __init__(self, shape_nouns: List[str], color_adjectives: List[str], size_adjectives: List[str]):
        """
        # TODO: think about unk (do we need it in object vocab?)
        :param shape_nouns: a list of string names for nouns.
        :param color_adjectives: a list of string names for colors.
        :param size_adjectives: a list of size adjectives ranging from smallest at idx 0 to largest at idx -1.
        """
        self.shape_nouns = shape_nouns
        self.n_nouns = len(shape_nouns)
        self.color_adjectives = color_adjectives
        self.n_color_adjectives = len(color_adjectives)
        self.idx_to_shapes_and_colors = self.shape_nouns + self.color_adjectives
        self.shapes_and_colors_to_idx = {token: i for i, token in enumerate(self.idx_to_shapes_and_colors)}
        self.size_adjectives = size_adjectives
        # Also size specification for 'average' size, e.g. if adjectives are small and big, 3 sizes exist.
        self.n_size_adjectives = len(size_adjectives) + 1
        self.size_adjectives.insert(self.n_size_adjectives // 2, 'average')
        self.size_to_int = {size: i + 1 for i, size in enumerate(self.size_adjectives)}
        self.n_objects = self.n_nouns * self.n_color_adjectives
        self.object_vectors = self.generate_objects()

    def generate_objects(self) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        TODO: think about unk in self.object_to_object_vector
        :param objects: List of objects specified as (size_ajd, color_ajd, shape_noun)
        :return:
        """
        object_to_object_vector = {}
        for size, color, shape in itertools.product(self.size_adjectives, self.color_adjectives, self.shape_nouns):
            object_vector = one_hot(self.n_objects, self.shapes_and_colors_to_idx[color]) + \
                            one_hot(self.n_objects, self.shapes_and_colors_to_idx[shape])
            object_vector *= self.size_to_int[size]
            if shape not in object_to_object_vector.keys():
                object_to_object_vector[shape] = {}
            if color not in object_to_object_vector[shape].keys():
                object_to_object_vector[shape][color] = {}
            object_to_object_vector[shape][color][size] = object_vector
        return object_to_object_vector


class World(MiniGridEnv):
    """
    TODO(lauraruis):
    When constructed generates object vectors for all possible object combinations based on
    size color and shape. Also creates an empty grid with an agent at an initial position. Call initialize to
    place objects and place the agent at a particular position.
    """

    def __init__(self, grid_size: int,
                 shape_nouns: list,
                 color_adjectives: list, size_adjectives: list, save_directory: str):
        self.grid_size = grid_size
        self.num_object_attributes = len(shape_nouns) * len(color_adjectives)
        self.object_vocabulary = ObjectVocabulary(shape_nouns=shape_nouns, color_adjectives=color_adjectives,
                                                  size_adjectives=size_adjectives)
        self.agent_start_pos = (0, 0)  # TODO: col row right?
        self.agent_start_dir = DIR_TO_INT[EAST]  # TODO: has to be int?
        self.mission = None  # TODO: add mission later?
        self.num_available_objects = len(IDX_TO_OBJECT.keys())
        self.available_objects = set(OBJECT_TO_IDX.keys())
        self.save_directory = save_directory

        # Keeps track of all objects currently placed on the grid, needed for specification of a situation.
        self.placed_object_list = []

        # Hash table for looking up locations of objects based on partially formed references (e.g. find the location(s)
        # of a red cylinder when the grid has both a big red cylinder and a small red cylinder.)
        self.object_lookup_table = {}
        super().__init__(grid_size=grid_size, max_steps=4 * grid_size * grid_size, see_through_walls=True)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

    def sample(self):
        """
        Create a grid world by sampling a random number of objects and place them at random positions
        in the grid world. Each object gets assigned a binary vector of attributes assigned to it. The agent
        gets positioned randomly on the grid world.
        # TODO(lauraruis): sample and initialize can be integrated into one function by making a function that just
        # samples the objects and passes it to initialize.
        # TODO(lauraruis): sample based on nonce words but aligned with way of defining objects
        # TODO: size * ([shape and color vec])
        :return: Situation
        """
        raise NotImplementedError()

    def initialize(self, objects: List[Tuple[Object, Position]], agent_position: Position, agent_direction: Direction,
                   carrying: Object=None):
        """
        Create a grid world by placing the objects that are passed as an argument at the specified locations and the
        agent at the specified location.
        # TODO: namedtuples as input or getting made outside of user influence? E.g. args col and row
        # TODO: option to add different grid size here?
        """
        self.clear_situation()
        self.agent_dir = DIR_TO_INT[agent_direction]
        self.place_agent_at(agent_position)
        for current_object, current_position in objects:
            self.place_object(current_object, current_position)
        if carrying:
            carrying_object = self.create_object(carrying,
                                                 self.object_vocabulary.object_vectors[carrying.shape]
                                                 [carrying.color][carrying.size])
            self.carrying = carrying_object
            self.carrying.cur_pos = np.array([-1, -1])
            self.carrying.cur_pos = self.agent_pos

    def create_object(self, object: Object, object_vector: np.ndarray):
        if object.shape == "circle":
            return Circle(object.color, size=SIZE_TO_INT[object.size], vector_representation=object_vector,
                          object_representation=object)
        elif object.shape == "square":
            return Square(object.color, vector_representation=object_vector, object_representation=object)
        elif object.shape == "cylinder":
            return Cylinder(object.color, size=SIZE_TO_INT[object.size], vector_representation=object_vector,
                            object_representation=object)
        else:
            raise NotImplementedError()

    def position_taken(self, position: Position):
        return self.grid.get(position.column, position.row) is not None

    def within_grid(self, position: Position):
        return (position.row and position.column) < self.grid_size

    def place_agent_at(self, position: Position):
        if not self.position_taken(position):
            self.place_agent(top=(position.column, position.row), size=(1, 1), rand_dir=False)
        else:
            raise ValueError("Trying to place agent on cell that is already taken.")

    def place_object(self, object: Object, position: Position):
        if not self.within_grid(position):
            raise IndexError("Trying to place object '{}' outside of grid of size {}.".format(
                object.shape, self.grid_size))
        # Object already placed at this location
        if self.position_taken(position):
            print("WARNING: attempt to place two objects at location ({}, {}), but overlapping objects not "
                  "supported. Skipping object.".format(position.row, position.column))
        object_vector = self.object_vocabulary.object_vectors[object.shape][object.color][object.size]
        positioned_object = PositionedObject(object=object, position=position, vector=object_vector)
        self.place_obj(self.create_object(object, object_vector),
                       top=(position.column, position.row), size=(1, 1))

        # Add to list that keeps track of all objects currently positioned on the grid.
        self.placed_object_list.append(positioned_object)

        # Adjust the object lookup table accordingly.
        self._add_object_to_lookup_table(positioned_object)

    def _add_object_to_lookup_table(self, positioned_object: PositionedObject):
        object_size = positioned_object.object.size
        object_color = positioned_object.object.color
        object_shape = positioned_object.object.shape

        # Generate all possible names
        object_names = generate_possible_object_names(size=object_size, color=object_color, shape=object_shape)
        for possible_object_name in object_names:
            if possible_object_name not in self.object_lookup_table.keys():
                self.object_lookup_table[possible_object_name] = {}

            # This part allows for multiple exactly the same objects (e.g. 2 small red circles) to be on the grid.
            if positioned_object.object not in self.object_lookup_table[possible_object_name].keys():
                self.object_lookup_table[possible_object_name][positioned_object.object] = []
            self.object_lookup_table[possible_object_name][positioned_object.object].append(positioned_object.position)

    def _remove_object(self, target_position: Position) -> PositionedObject:
        # remove from placed_object_list
        target_object = None
        for i, positioned_object in enumerate(self.placed_object_list):
            if positioned_object.position == target_position:
                target_object = self.placed_object_list[i]
                del self.placed_object_list[i]
                break

        # remove from object_lookup Table
        self._remove_object_from_lookup_table(target_object)

        # remove from gym grid
        self.grid.get(target_position.column, target_position.row)
        self.grid.set(target_position.column, target_position.row, None)

        return target_object

    def _remove_object_from_lookup_table(self, positioned_object: PositionedObject):
        possible_object_names = generate_possible_object_names(positioned_object.object.size,
                                                               positioned_object.object.color,
                                                               positioned_object.object.shape)
        for possible_object_name in possible_object_names:
            self.object_lookup_table[possible_object_name][positioned_object.object].remove(positioned_object.position)

    def move_object(self, old_position: Position, new_position: Position):
        # Remove object from old position
        old_positioned_object = self._remove_object(old_position)
        if not old_positioned_object:
            raise ValueError("Trying to move an object from an empty grid location (row {}, col {})".format(
                old_position.row, old_position.column))

        # Add object at new position
        self.place_object(old_positioned_object.object, new_position)

    def pick_up_object(self, recorded_situations: List[Situation]):
        """
        Picking up an object in gym-minigrid means removing it and saying the agent is carrying it.
        :return:
        """
        assert self.grid.get(*self.agent_pos) is not None, "Trying to pick up an object at an empty cell."
        self.step(self.actions.pickup)
        if self.carrying:
            self._remove_object(Position(column=self.agent_pos[0], row=self.agent_pos[1]))
            recorded_situations.append(self.get_current_situation())

    def drop_object(self, recorded_situations: List[Situation]):
        assert self.carrying is not None, "Trying to drop something but not carrying anything."
        self.place_object(self.carrying.object_representation, Position(column=self.agent_pos[0],
                                                                        row=self.agent_pos[1]))
        self.carrying = None
        recorded_situations.append(self.get_current_situation())

    def push_object(self, heavyness: int):
        raise NotImplementedError()

    def direction_to_goal(self, goal: Position):
        difference_vec = np.array([goal.column - self.agent_pos[0], goal.row - self.agent_pos[1]])
        difference_vec[difference_vec < 0] = 0
        col_difference = difference_vec[0]
        row_difference = difference_vec[1]
        if col_difference and row_difference:
            return "SE", self.actions.left
        elif col_difference and not row_difference:
            return "NE", self.actions.right
        elif row_difference and not col_difference:
            return "SW", self.actions.right
        else:
            return "NW", self.actions.left

    def go_to_position(self, position: Position, recorded_situations: List[Situation], manner: str):

        # Zigzag somewhere until in line with the goal, then just go straight for the goal
        if manner == "while zigzagging" and not self.agent_in_line_with_goal(position):
            # find direction of goal
            direction_to_goal, first_move = self.direction_to_goal(position)
            previous_step = first_move
            if direction_to_goal == "NE" or direction_to_goal == "SE":
                self.take_step_forward(EAST)
            else:
                self.take_step_forward(WEST)
            recorded_situations.append(self.get_current_situation())
            while not self.agent_in_line_with_goal(position):
                # turn in opposite direction of previous step and take take step
                if previous_step == self.actions.left:
                    self.step(self.actions.right)
                else:
                    self.step(self.actions.left)
                self.step(self.actions.forward)
                recorded_situations.append(self.get_current_situation())

            # Finish the route not zigzagging
            while self.agent_pos[0] > position.column:
                self.take_step_forward(direction=WEST)
                recorded_situations.append(self.get_current_situation())
            while self.agent_pos[0] < position.column:
                self.take_step_forward(direction=EAST)
                recorded_situations.append(self.get_current_situation())
            while self.agent_pos[1] > position.row:
                self.take_step_forward(direction=NORTH)
                recorded_situations.append(self.get_current_situation())
            while self.agent_pos[1] < position.row:
                self.take_step_forward(direction=SOUTH)
                recorded_situations.append(self.get_current_situation())
        else:
            # Look left and right if cautious
            if manner == "cautiously":
                self.step(action=self.actions.left)
                recorded_situations.append(self.get_current_situation())
                self.step(action=self.actions.right)
                recorded_situations.append(self.get_current_situation())
                self.step(action=self.actions.right)
                recorded_situations.append(self.get_current_situation())
                self.step(action=self.actions.left)
                recorded_situations.append(self.get_current_situation())
                self.step(action=self.actions.left)
                recorded_situations.append(self.get_current_situation())
                self.step(action=self.actions.right)
                recorded_situations.append(self.get_current_situation())

            # Calculate the route to the object on the grid
            while self.agent_pos[0] > position.column:
                if manner == "while spinning":
                    self.step(action=self.actions.left)
                    recorded_situations.append(self.get_current_situation())
                    self.take_step_in_direction(direction=WEST)
                else:
                    self.take_step_forward(direction=WEST)
                recorded_situations.append(self.get_current_situation())

                # Stop after each step
                if manner == "hesitantly":
                    recorded_situations.append(self.get_current_situation())

                # Spin to the left
                if manner == "while spinning":
                    self.step(action=self.actions.left)
                    recorded_situations.append(self.get_current_situation())
            while self.agent_pos[0] < position.column:
                if manner == "while spinning":
                    self.step(action=self.actions.left)
                    recorded_situations.append(self.get_current_situation())
                    self.take_step_in_direction(direction=EAST)
                else:
                    self.take_step_forward(direction=EAST)
                recorded_situations.append(self.get_current_situation())

                # Stop after each step
                if manner == "hesitantly":
                    recorded_situations.append(self.get_current_situation())
            while self.agent_pos[1] > position.row:
                if manner == "while spinning":
                    self.step(action=self.actions.left)
                    recorded_situations.append(self.get_current_situation())
                    self.take_step_in_direction(direction=NORTH)
                else:
                    self.take_step_forward(direction=NORTH)
                recorded_situations.append(self.get_current_situation())

                # Stop after each step
                if manner == "hesitantly":
                    recorded_situations.append(self.get_current_situation())
            while self.agent_pos[1] < position.row:
                # Spin to the left
                if manner == "while spinning":
                    self.step(action=self.actions.left)
                    recorded_situations.append(self.get_current_situation())
                    self.take_step_in_direction(direction=SOUTH)
                else:
                    self.take_step_forward(direction=SOUTH)
                recorded_situations.append(self.get_current_situation())

                # Stop after each step
                if manner == "hesitantly":
                    recorded_situations.append(self.get_current_situation())

    def agent_in_line_with_goal(self, goal: Position):
        return goal.column == self.agent_pos[0] or goal.row == self.agent_pos[1]

    def take_step_forward(self, direction: Direction):
        """
        Turn to some direction and take a step forward.
        """
        self.agent_dir = DIR_TO_INT[direction]
        self.step(action=self.actions.forward)

    def take_step_in_direction(self, direction: Direction):
        """
        Take a step in some direction without turning to that direction.
        """
        dir_vec = DIR_TO_VEC[DIR_TO_INT[direction]]
        self.agent_pos = self.agent_pos + dir_vec

    def get_current_situation(self) -> Situation:
        if self.carrying:
            carrying = self.carrying.object_representation
        else:
            carrying = None
        return Situation(grid_size=self.grid_size,
                         agent_position=Position(column=self.agent_pos[0], row=self.agent_pos[1]),
                         agent_direction=INT_TO_DIR[self.agent_dir], placed_objects=self.placed_object_list.copy(),
                         carrying=carrying)

    def save_situation(self, file_name) -> str:
        self.mission = "test"
        save_location = os.path.join(self.save_directory, file_name)
        assert save_location.endswith('.png'), "Invalid file name passed to save_situation, must end with .png."
        success = self.render().img.save(save_location)
        if not success:
            print("WARNING: image with name {} failed to save.".format(file_name))
            return ''
        else:
            return save_location

    def visualize_sequence(self, action_sequence: List[Situation]) -> str:
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
        filenames = []
        # Loop over actions and take them.
        for i, (action, action_int) in enumerate(action_sequence):
            current_filename = os.path.join(mission_dir, 'situation_' + str(i) + '.png')

            # Stay.
            if action == STAY:
                save_location = self.save_situation(current_filename)
            elif action == PICK_UP:
                self.step(self.actions.pickup)
                save_location = self.save_situation(current_filename)
            elif action == DROP:
                self.step(self.actions.drop)
                save_location = self.save_situation(current_filename)
            # Move forward.
            else:
                if action_int >= 0:
                    self.agent_dir = action_int
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

    def clear_situation(self):
        self.object_lookup_table.clear()
        self.placed_object_list.clear()
        self.reset()

    def initialize_object_vocabulary(self):
        # TODO
        raise NotImplementedError()

    def set_mission(self, mission: str):
        self.mission = mission

