from collections import namedtuple
import itertools
import numpy as np
import random
from typing import Tuple
from typing import List
from typing import Dict

from helpers import topo_sort
from helpers import plan_step
from helpers import one_hot
from helpers import generate_possible_object_names

SemType = namedtuple("SemType", "name")
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

DIR_TO_INT = {
    NORTH: 3,
    SOUTH: 1,
    WEST: 2,
    EAST: 0,
    FORWARD: -1
}

Action = namedtuple("Action", "name")
PICK_UP = Action("pick_up")
MOVE_FORWARD = Action("move_forward")
STAY = Action("stay")
DROP = Action("drop")

ACTION_TO_INT = {
    STAY: -1,
    MOVE_FORWARD: 2,
    PICK_UP: 3,
    DROP: 4
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

    def replace(self, var_to_find, replace_by_var):
        return Term(
            self.function,
            tuple(replace_by_var if variable == var_to_find else variable for variable in self.arguments),
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

    def bind(self, bind_var):
        """
        Bind a variable to its head, e.g. for 'kick the ball', 'kick' is the head and 'the ball' will be bind to it.
        :param bind_var:
        :return:
        """
        sub_var, variables_out = self.variables[0], self.variables[1:]
        # assert sub_var.sem_type == bind_var.sem_type
        terms_out = [t.replace(sub_var, bind_var) for t in self.terms]
        return LogicalForm((bind_var,) + variables_out, tuple(terms_out))

    def select(self, variables, exclude=frozenset()):
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


class ObjectVocabulary(object):
    """
    Constructs an object vocabulary. Each object will be calculated by the following:
    size_adjective * (color_adjective + shape_noun), e.g. big * (red + circle) could be
    2 * ([0 1 0] + [1 0 0]) = [2 2 0].
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

    def generate_objects(self) -> Dict[str, Dict[str, Dict[str, list]]]:
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


class World(object):
    """
    TODO(lauraruis): why is this a class
    """

    def __init__(self, grid_size: int, n_attributes: int, min_objects: int, max_objects: int, shape_nouns: list,
                 color_adjectives: list, size_adjectives: list):
        self.grid_size = grid_size
        self.n_attributes = n_attributes
        self.min_objects = min_objects
        assert max_objects < grid_size ** 2
        self.max_objects = max_objects
        self.object_vocabulary = ObjectVocabulary(shape_nouns=shape_nouns, color_adjectives=color_adjectives,
                                                  size_adjectives=size_adjectives)

    def sample(self):
        """
        Create a grid world by sampling a random number of objects and place them at random positions
        in the grid world. Each object gets assigned a binary vector of attributes assigned to it. The agent
        gets positioned randomly on the grid world.
        # TODO(lauraruis): sample and initialize can be integrated into one function by making a function that just
        # samples the objects and passes it to initialize.
        :return: Situation
        """
        grid = np.zeros((self.grid_size, self.grid_size, self.n_attributes))
        num_objects = np.random.randint(self.min_objects, self.max_objects)
        num_placed_objects = 0
        occupied_rows = set()
        occupied_cols = set()
        placed_objects = []
        while num_placed_objects < num_objects:
            row, column = np.random.randint(self.grid_size, size=2)

            # Object already placed at this location
            if grid[row, column, :].any():
                continue

            # An object is represented by a binary vector
            grid[row, column, :] = np.random.binomial(1, 0.5, size=self.n_attributes)
            occupied_cols.add(column)
            occupied_rows.add(row)
            placed_objects.append(("", "", (column, row)))  # TODO: add random binomial here too?
            num_placed_objects += 1
        agent_col = np.random.choice([i for i in range(self.grid_size) if i not in occupied_cols])
        agent_row = np.random.choice([i for i in range(self.grid_size) if i not in occupied_rows])
        return Situation(grid, agent_pos=(agent_col, agent_row), objects=placed_objects)

    def initialize(self, objects: List[Tuple[str, str, str, Tuple[int, int]]], agent_pos=(0, 0)):
        """
        Create a grid world by placing the objects that are passed as an argument at the specified locations and the
        agent at the specified location.
        # TODO: make agent named tuple
        :return: Situation
        """
        grid = np.zeros((self.grid_size, self.grid_size, self.n_attributes))
        num_placed_objects = 0
        placed_objects = {}
        placed_objects_list = []
        for (object_name, object_color, object_size, object_position) in objects:
            column, row = object_position
            if (row or column) > self.grid_size:
                raise IndexError("Trying to place object '{}' outside of grid of size {}.".format(
                    object_name, self.grid_size))

            # Object already placed at this location
            if grid[row, column, :].any():
                print("WARNING: attempt to place two objects at location ({}, {}), but overlapping objects not "
                      "supported. Skipping object.".format(row, column))
                continue

            # An object is represented by a binary vector
            object_vector = self.object_vocabulary.object_vectors[object_name][object_color][object_size]
            grid[row, column, :] = object_vector
            object_names = generate_possible_object_names(size=object_size, color=object_color, shape=object_name)
            placed_objects_list.append((object_name, object_color, object_size, (column, row), object_vector))
            for possible_object_name in object_names:
                if possible_object_name not in placed_objects.keys():
                    placed_objects[possible_object_name] = {}
                placed_objects[possible_object_name][' '.join([object_size, object_color, object_name])] = (column, row)

            num_placed_objects += 1
        return Situation(grid, agent_pos, agent_direction=EAST, objects=placed_objects, placed_objects=placed_objects_list)


class Situation(object):
    def __init__(self, grid, agent_pos, agent_direction, objects, placed_objects):
        self.grid = grid
        self.grid_size = grid.shape[0]
        self.agent_pos = agent_pos  # position is [col, row] (i.e. [x-axis, y-axis])
        self.agent_direction = agent_direction
        self.objects = objects
        self.placed_objects = placed_objects

    def step(self, direction):
        next_pos = plan_step(self.agent_pos, DIR_TO_INT[direction])
        if next_pos.min() < 0 or next_pos.max() >= self.grid_size:
            print("WARNING: trying to move outside of grid.")
            next_pos = self.agent_pos
        return Situation(self.grid, next_pos, agent_direction=direction, objects=self.objects,
                         placed_objects=self.placed_objects)

    def demonstrate(self, logical_form):
        events = [variable for variable in logical_form.variables if variable.sem_type == EVENT]
        seq_constraints = [term.arguments for term in logical_form.terms if term.function == "seq"]
        ordered_events = topo_sort(events, seq_constraints)
        result = [(Command(STAY, None), self, ACTION_TO_INT[STAY])]
        for event in ordered_events:
            sub_logical_form = logical_form.select([event], exclude={"seq"})
            plan_part = result[-1][1].plan(event, sub_logical_form)
            if plan_part is None:
                return None
            result += plan_part
        return result

    def plan(self, event, logical_form):
        event_lf = logical_form.select([event], exclude={"patient"})
        args = [term.arguments[1] for term in logical_form.terms if term.function == "patient"]

        # Find the action verb if it exists.
        action = None
        is_transitive = False
        if event_lf.head.sem_type == EVENT:
            for term in event_lf.terms:
                if term.specs.action:
                    action = term.specs.action
                    is_transitive = term.specs.is_transitive

        # Find the manner adverb if it exists.  TODO: there can only be a manner if there is a verb?
        manner = [term.specs.manner for term in event_lf.terms if term.specs.manner]
        manner = manner.pop() if manner else None

        assert len(args) <= 1
        if len(args) == 0:
            return [(Command(STAY, event), self, ACTION_TO_INT[STAY])]
        else:
            arg_logical_form = logical_form.select([args[0]])
            object_locations = arg_logical_form.to_predicate(self.objects)
            if not object_locations:
                return None
            object_name, goal = random.sample(object_locations.items(), 1).pop()
            # TODO: if transitive verb, do something at goal location.
            return self.route_to((goal[0], goal[1]), event, is_transitive=is_transitive, object_name=object_name)

    def route_to(self, goal, event, is_transitive=False, object_name=""):
        """

        :param goal: location in form (x-axis, y-axis) (i.e. (column, row)
        :param event:
        :param is_transitive:
        :return:
        """
        path = []
        current_state = self
        while current_state.agent_pos[0] > goal[0]:
            current_state = current_state.step(WEST)
            path.append((Command(MOVE_FORWARD, None), current_state, DIR_TO_INT[WEST]))
        while current_state.agent_pos[0] < goal[0]:
            current_state = current_state.step(EAST)
            path.append((Command(MOVE_FORWARD, None), current_state, DIR_TO_INT[EAST]))
        while current_state.agent_pos[1] > goal[1]:
            current_state = current_state.step(NORTH)
            path.append((Command(MOVE_FORWARD, None), current_state, DIR_TO_INT[NORTH]))
        while current_state.agent_pos[1] < goal[1]:
            current_state = current_state.step(SOUTH)
            path.append((Command(MOVE_FORWARD, None), current_state, DIR_TO_INT[SOUTH]))
        if not is_transitive:
            path.append((Command(STAY, event), current_state, ACTION_TO_INT[STAY]))
            assert (goal == current_state.agent_pos).all(), "Route finding to goal failed."
        else:
            path.append((Command(PICK_UP, event), current_state, ACTION_TO_INT[PICK_UP]))
            if not self.against_wall(current_state.agent_pos):
                object_location = current_state.agent_pos
                current_state = current_state.step(self.agent_direction)
                current_state = current_state.move_object(object_location, current_state.agent_pos, object_name)
                path.append((Command(MOVE_FORWARD, event), current_state, DIR_TO_INT[self.agent_direction]))
                path.append((Command(DROP, event), current_state, ACTION_TO_INT[DROP]))
        return path

    def move_object(self, old_location, new_location, object_name):
        objects = self.objects
        num_attributes = len(self.grid[0][0])
        replacement_vector = np.zeros(num_attributes)
        object_vector = self.grid[old_location[1]][old_location[0]]
        self.grid[old_location[1]][old_location[0]] = replacement_vector
        self.grid[new_location[1]][new_location[0]] = object_vector

        possible_object_names = generate_possible_object_names(*object_name.split())
        for possible_object_name in possible_object_names:
            objects[possible_object_name][object_name] = new_location
        return Situation(self.grid, self.agent_pos, agent_direction=self.agent_direction, objects=objects,
                         placed_objects=self.placed_objects)

    def against_wall(self, current):
        grid_size = self.grid_size
        return current[0] + 1 == grid_size or current[1] + 1 == grid_size
