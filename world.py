from collections import namedtuple
import numpy as np
from typing import Tuple
from typing import List

from helpers import topo_sort
from helpers import plan_step
from gridworld import EmptyEnv

SemType = namedtuple("SemType", "name")
Variable = namedtuple("Variable", "name sem_type")

ENTITY = SemType("noun")
EVENT = SemType("verb")

Direction = namedtuple("Direction", "name")
NORTH = Direction("north")
SOUTH = Direction("south")
WEST = Direction("west")
EAST = Direction("east")
STAY = Direction("stay")

DIR_TO_INT = {
    NORTH: 3,
    SOUTH: 1,
    WEST: 2,
    EAST: 0
}

Command = namedtuple("Command", "direction event")

FACE_NORTH = 3
FACE_WEST = 2
FACE_SOUTH = 1
FACE_EAST = 0
MOVE_FORWARD = 2


class Term(object):
    """
    Holds terms that can be parts of logical forms and take as arguments variables that the term can operate over.
    E.g. for the phrase 'Brutus stabs Caesar' the term is stab(B, C) which will be represented by the string
    "(stab B:noun C:noun)".
    """

    def __init__(self, function: str, args: tuple, weights=None, meta=None):
        self.function = function
        self.arguments = args
        self.weights = weights
        self.meta = meta

    def replace(self, var_to_find, replace_by_var):
        return Term(
            self.function,
            tuple(replace_by_var if variable == var_to_find else variable for variable in self.arguments),
            self.weights,
            self.meta
        )

    def to_predicate(self):
        assert self.weights is not None

        def predicate_fn(features):
            return (self.weights * features[..., :]).sum(axis=-1)

        return predicate_fn

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
            self.head = variables[0]  # TODO: what is head

    def bind(self, bind_var):
        sub_var, variables_out = self.variables[0], self.variables[1:]
        assert sub_var.sem_type == bind_var.sem_type
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

    def to_predicate(self):
        assert len(self.variables) == 1
        term_predicates = [term.to_predicate() for term in self.terms]

        def predicate_fn(features):
            term_scores = [term_to_predicate(features) for term_to_predicate in term_predicates]
            return np.product(term_scores, axis=0)

        return predicate_fn

    def __repr__(self):
        return "LF({})".format(" ^ ".join([repr(term) for term in self.terms]))


class World(object):
    """
    TODO(lauraruis): why is this a class
    """

    def __init__(self, grid_size: int, n_attributes: int, min_objects: int, max_objects: int):
        self.grid_size = grid_size
        self.n_attributes = n_attributes
        self.min_objects = min_objects
        assert max_objects < grid_size ** 2
        self.max_objects = max_objects

    def sample(self):
        """
        Create a grid world by sampling a random number of objects and place them at random positions
        in the grid world. Each object gets assigned a binary vector of attributes assigned to it. The agent
        gets positioned randomly on the grid world.
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

    def initialize(self, objects: List[Tuple[str, str, Tuple[int, int]]], agent_pos=(0, 0)):
        """
        Create a grid world by sampling a random number of objects and place them at random positions
        in the grid world. Each object gets assigned a binary vector of attributes assigned to it. The agent
        gets positioned randomly on the grid world.
        TODO: make objects named tuple
        :return: Situation
        """
        grid = np.zeros((self.grid_size, self.grid_size, self.n_attributes))
        num_placed_objects = 0
        placed_objects = []
        for (object_name, object_color, object_position) in objects:
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
            grid[row, column, :] = np.random.binomial(1, 0.5, size=self.n_attributes)
            placed_objects.append((object_name, object_color, (column, row)))  # TODO: add random binomial here too?
            num_placed_objects += 1
        return Situation(grid, agent_pos, objects=placed_objects)


class Situation(object):
    def __init__(self, grid, agent_pos, objects):
        self.grid = grid
        self.grid_size = grid.shape[0]
        self.agent_pos = agent_pos  # position is [col, row] (i.e. [x-axis, y-axis])
        self.objects = objects

    def step(self, action):
        if action == STAY:
            next_pos = self.agent_pos
        else:
            next_pos = plan_step(self.agent_pos, DIR_TO_INT[action])
            if next_pos.min() < 0 or next_pos.max() >= self.grid_size:
                print("WARNING: trying to move outside of grid.")
                next_pos = self.agent_pos

        return Situation(self.grid, next_pos, self.objects)

    def demonstrate(self, logical_form):
        events = [variable for variable in logical_form.variables if variable.sem_type == EVENT]
        seq_constraints = [term.arguments for term in logical_form.terms if term.function == "seq"]
        ordered_events = topo_sort(events, seq_constraints)
        result = [(None, self)]
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
        # comb_weights = np.sum([t.weights for t in event_lf.terms], axis=0)
        # TODO eww
        action = [term.weights[0] for term in event_lf.terms if term.weights[0] is not None]
        manner = [term.weights[1] for term in event_lf.terms if term.weights[1] is not None]
        has_arg, = [term.weights[2] for term in event_lf.terms if term.weights[2] is not None]
        action = action[0] if action != [] else None
        manner = manner[0] if manner != [] else None
        assert len(args) <= 1
        if len(args) == 0:
            return [(Command(STAY, event), self)]
        else:
            arg_logical_form = logical_form.select([args[0]])
            arg_predicate = arg_logical_form.to_predicate()
            scores = arg_predicate(self.grid)
            candidates = np.asarray((scores > 0).nonzero()).T.tolist()
            if len(candidates) == 0:
                return None
            goal = candidates[np.random.randint(len(candidates))]
            return self.route_to((goal[1], goal[0]), event)

    def route_to(self, goal, event):
        """

        :param goal: location in form (x-axis, y-axis) (i.e. (column, row)
        :param event:
        :return:
        """
        path = []
        current_state = self
        while current_state.agent_pos[0] > goal[0]:
            current_state = current_state.step(WEST)
            path.append((Command(WEST, None), current_state))
        while current_state.agent_pos[0] < goal[0]:
            current_state = current_state.step(EAST)
            path.append((Command(EAST, None), current_state))
        while current_state.agent_pos[1] > goal[1]:
            current_state = current_state.step(NORTH)
            path.append((Command(NORTH, None), current_state))
        while current_state.agent_pos[1] < goal[1]:
            current_state = current_state.step(SOUTH)
            path.append((Command(SOUTH, None), current_state))
        path.append((Command(STAY, event), current_state))
        assert (goal == current_state.agent_pos).all(), "Route finding to goal failed."
        return path
