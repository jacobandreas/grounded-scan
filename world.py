from collections import namedtuple
import numpy as np

SemType = namedtuple("SemType", "name")
Variable = namedtuple("Variable", "name sem_type")

ENTITY = SemType("n")
EVENT = SemType("v")

Direction = namedtuple("Direction", "name")
NORTH = Direction("north")
SOUTH = Direction("south")
WEST = Direction("west")
EAST = Direction("east")
STAY = Direction("stay")

GRID_SIZE = 10
N_ATTRIBUTES = 8
MIN_OBJECTS = 5
MAX_OBJECTS = 10

Command = namedtuple("Command", "direction event")

# TODO cleaner
VAR_COUNTER = [0]
def free_var(sem_type):
    name = "x{}".format(VAR_COUNTER[0])
    VAR_COUNTER[0] += 1
    return Variable(name, sem_type)

# TODO faster
def topo_sort(items, constraints):
    items = list(items)
    constraints = list(constraints)
    out = []
    while len(items) > 0:
        roots = [
            i for i in items 
            if not any(c[1] == i for c in constraints)
        ]
        assert len(roots) > 0, (items, constraints)
        to_pop = roots[0]
        items.remove(to_pop)
        constraints = [c for c in constraints if c[0] != to_pop]
        out.append(to_pop)
    return out

def rand_weights():
    return 2 * (np.random.random(N_ATTRIBUTES) - 0.5)

def accept_weights():
    return np.ones(N_ATTRIBUTES)

class Term(object):
    def __init__(self, function, args, weights=None, meta=None):
        self.function = function
        self.args = args
        self.weights = weights
        self.meta = meta
        
    def replace(self, find, replace):
        return Term(
            self.function,
            tuple(replace if var == find else var for var in self.args),
            self.weights,
            self.meta
        )

    def to_predicate(self):
        assert self.weights is not None
        def pred(features):
            return (self.weights * features[..., :]).sum(axis=-1)
        return pred

    def __repr__(self):
        parts = [self.function]
        for var in self.args:
            parts.append("{}:{}".format(var.name, var.sem_type.name))
        return "({})".format(" ".join(parts))

class Lf(object):
    def __init__(self, variables, terms):
        self.variables = variables
        self.terms = terms
        if len(variables) > 0:
            self.head = variables[0]

    def bind(self, bind_var):
        sub_var, variables_out = self.variables[0], self.variables[1:]
        assert sub_var.sem_type == bind_var.sem_type
        terms_out = [t.replace(sub_var, bind_var) for t in self.terms]
        return Lf((bind_var,) + variables_out, tuple(terms_out))

    def select(self, variables, exclude=frozenset()):
        queue = list(variables)
        used_vars = set()
        terms_out = []
        while len(queue) > 0:
            var = queue.pop()
            deps = [t for t in self.terms if t.function not in exclude and t.args[0] == var]
            for term in deps:
                terms_out.append(term)
                used_vars.add(var)
                for v in term.args[1:]:
                    if v not in used_vars:
                        queue.append(v)

        vars_out = [v for v in self.variables if v in used_vars]
        terms_out = list(set(terms_out))
        return Lf(vars_out, terms_out)

    def to_predicate(self):
        assert len(self.variables) == 1
        term_predicates = [t.to_predicate() for t in self.terms]
        def pred(features):
            term_scores = [p(features) for p in term_predicates]
            return np.product(term_scores, axis=0)
        return pred

    def __repr__(self):
        out = [repr(t) for t in self.terms]
        return "Lf({})".format(" ^ ".join(out))


class World(object):
    def __init__(self):
        pass

    def sample(self):
        grid = np.zeros((GRID_SIZE, GRID_SIZE, N_ATTRIBUTES))
        n_objects = np.random.randint(MIN_OBJECTS, MAX_OBJECTS)
        placed_objects = 0
        while placed_objects < n_objects:
            r, c = np.random.randint(GRID_SIZE, size=2)
            if grid[r, c, :].any():
                continue
            attrs = np.random.binomial(1, 0.5, size=N_ATTRIBUTES)
            grid[r, c, :] = attrs
            placed_objects += 1
        agent_pos = tuple(np.random.randint(GRID_SIZE, size=2))
        return Situation(grid, agent_pos)

class Situation(object):
    def __init__(self, grid, agent_pos):
        self.grid = grid
        self.agent_pos = agent_pos

    def step(self, action):
        if action == NORTH and self.agent_pos[0] > 0:
            next_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
        elif action == SOUTH and self.agent_pos[0] < GRID_SIZE - 1:
            next_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
        elif action == WEST and self.agent_pos[1] > 0:
            next_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
        elif action == EAST and self.agent_pos[1] < GRID_SIZE - 1:
            next_pos = (self.agent_pos[0], self.agent_pos[1] + 1)
        else:
            next_pos = self.agent_pos
        return Situation(self.grid, next_pos)

    def demonstrate(self, lf):
        events = [v for v in lf.variables if v.sem_type == EVENT]
        seq_constraints = [t.args for t in lf.terms if t.function == "seq"]
        ordered_events = topo_sort(events, seq_constraints)
        result = [(None, self)]
        for event in ordered_events:
            sub_lf = lf.select([event], exclude={"seq"})
            plan_part = result[-1][1].plan(event, sub_lf)
            if plan_part is None:
                return None
            result += plan_part
        return result

    def plan(self, event, lf):
        event_lf = lf.select([event], exclude={"patient"})
        args = [t.args[1] for t in lf.terms if t.function == "patient"]
        #comb_weights = np.sum([t.weights for t in event_lf.terms], axis=0)
        # TODO eww
        action = [t.weights[0] for t in event_lf.terms if t.weights[0] is not None]
        manner = [t.weights[1] for t in event_lf.terms if t.weights[1] is not None]
        has_arg, = [t.weights[2] for t in event_lf.terms if t.weights[2] is not None]
        action = action[0] if action != [] else None
        manner = manner[0] if manner != [] else None
        assert len(args) <= 1
        if len(args) == 0:
            return [(Command(STAY, event), self)]
        else:
            arg_lf = lf.select([args[0]])
            arg_pred = arg_lf.to_predicate()
            scores = arg_pred(self.grid)
            candidates = np.asarray((scores > 0).nonzero()).T.tolist()
            if len(candidates) == 0:
                return None
            goal = candidates[np.random.randint(len(candidates))]
            return self.route_to(goal, event)

    def route_to(self, goal, event):
        r_first = np.random.randint(2)
        path = []
        curr_state = self
        while curr_state.agent_pos[0] > goal[0]:
            curr_state = curr_state.step(NORTH)
            path.append((Command(NORTH, None), curr_state))
        while curr_state.agent_pos[0] < goal[0]:
            curr_state = curr_state.step(SOUTH)
            path.append((Command(SOUTH, None), curr_state))
        while curr_state.agent_pos[1] > goal[1]:
            curr_state = curr_state.step(WEST)
            path.append((Command(WEST, None), curr_state))
        while curr_state.agent_pos[1] < goal[1]:
            curr_state = curr_state.step(EAST)
            path.append((Command(EAST, None), curr_state))
        path.append((Command(STAY, event), curr_state))
        return path

