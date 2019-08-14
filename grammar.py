#!/usr/bin/env python3

from collections import defaultdict, namedtuple
import numpy as np

NT = namedtuple("Nonterminal", "name")
T = namedtuple("Terminal", "name")

ROOT = NT("ROOT")
VP = NT("VP")
VV_intransitive = NT("VV_intrans")
VV_transitive = NT("VV_trans")
RB = NT("RB")
DP = NT("DP")
NP = NT("NP")
NN = NT("NN")
JJ = NT("JJ")

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

Command = namedtuple("Command", "direction event")

MAX_RECURSION = 2
N_ATTRIBUTES = 8
MIN_OBJECTS = 5
MAX_OBJECTS = 10
GRID_SIZE = 10

def rand_weights():
    return 2 * (np.random.random(N_ATTRIBUTES) - 0.5)

def accept_weights():
    return np.ones(N_ATTRIBUTES)

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

class Rule(object):
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs

    def instantiate(self, *args, **kwargs):
        raise NotImplementedError()

class LexicalRule(Rule):
    def instantiate(self, meta=None, **kwargs):
        # TODO a little fishy to have recursion meta here rather than in wrapper
        var = free_var(self.sem_type)
        return Lf(
            (var,),
            (Term(self.name, (var,), weights=self.weights, meta=meta),)
        )

class Root(Rule):
    def __init__(self):
        super().__init__(ROOT, [VP])

    def instantiate(self, child, **kwargs):
        return child

class RootConj(Rule):
    def __init__(self):
        super().__init__(ROOT, [VP, T("and"), ROOT])

    def instantiate(self, child1, child2, **kwargs):
        return Lf(
            child1.variables + child2.variables, 
            child1.terms + child2.terms + (Term("seq", (child1.head, child2.head)),)
        )

class VpWrapper(Rule):
    def __init__(sef):
        super().__init__(VP, [RB, VP])

    def instantiate(self, rb, vp, meta, **kwargs):
        bound = rb.bind(vp.head)
        assert bound.variables[0] == vp.head
        return Lf(vp.variables + bound.variables[1:], vp.terms + bound.terms)

class VpIntransitive(Rule):
    def __init__(self):
        super().__init__(VP, [VV_intransitive])

    def instantiate(self, vv, **kwargs):
        return vv

class VpTransitive(Rule):
    def __init__(self):
        super().__init__(VP, [VV_transitive, DP])

    def instantiate(self, vv, dp, **kwargs):
        role = Term('patient', (vv.head, dp.head))
        return Lf(vv.variables + dp.variables, vv.terms + dp.terms + (role,))

class Dp(Rule):
    def __init__(self):
        super().__init__(DP, [T("a"), NP])

    def instantiate(self, np, **kwargs):
        return np

class NpWrapper(Rule):
    def __init__(self):
        super().__init__(NP, [JJ, NP])

    def instantiate(self, jj, np, meta=None, **kwargs):
        bound = jj.bind(np.head)
        assert bound.variables[0] == np.head
        return Lf(np.variables + bound.variables[1:], np.terms + bound.terms)

class Np(Rule):
    def __init__(self):
        super().__init__(NP, [NN])

    def instantiate(self, nn, **kwargs):
        return nn

class VvTransitive(LexicalRule):
    def __init__(self, word, weights):
        super().__init__(VV_transitive, [T(word)])
        self.name = word
        self.sem_type = EVENT
        self.weights = weights

class VvIntransitive(LexicalRule):
    def __init__(self, word, weights):
        super().__init__(VV_intransitive, [T(word)])
        self.name = word
        self.sem_type = EVENT
        self.weights = weights

class Rb(LexicalRule):
    def __init__(self, word, weights):
        super().__init__(RB, [T(word)])
        self.name = word
        self.sem_type = EVENT
        self.weights = weights

class Nn(LexicalRule):
    def __init__(self, word, weights):
        super().__init__(NN, [T(word)])
        self.name = word
        self.weights = weights
        self.sem_type = ENTITY

class Jj(LexicalRule):
    def __init__(self, word, weights):
        super().__init__(JJ, [T(word)])
        self.name = word
        self.weights = weights
        self.sem_type = ENTITY

class Grammar(object):
    RULES = [
        Root(),
        RootConj(),
        VpWrapper(),
        VpIntransitive(),
        VpTransitive(),
        Dp(),
        NpWrapper(),
        Np(),
        VvIntransitive("twist", ("twist", None, False)),
        VvIntransitive("shout", ("shout", None, False)),
        VvIntransitive("jump", ("jump", None, False)),
        VvTransitive("touch", ("touch", None, True)),
        VvTransitive("push", ("push", None, True)),
        VvTransitive("break", ("break", None, True)),
        Rb("quickly", (None, "quickly", None)),
        Rb("slowly", (None, "slowly", None)),
        Rb("moderately", (None, "moderately", None)),
        Nn("circle", rand_weights()),
        Nn("square", rand_weights()),
        Nn("triangle", rand_weights()),
        Nn("object", accept_weights()),
        Jj("big", rand_weights()),
        Jj("small", rand_weights()),
        Jj("red", rand_weights()),
        Jj("green", rand_weights()),
        Jj("blue", rand_weights()),
    ]

    HOLD_OUT_OBJECT = np.random.binomial(1, 0.5, size=N_ATTRIBUTES)
    HOLD_OUT_ADVERB = "jump"
    HOLD_OUT_ADJECTIVE = "blue"
    HOLD_OUT_APPLICATION = ("big", np.random.binomial(1, 0.5, size=N_ATTRIBUTES))
    HOLD_OUT_COMPOSITION = ("small", "red")
    HOLD_OUT_RECURSION = ("big", 1)

    CATEGORIES = {
        "manner": {"quickly", "slowly", "moderately"},
        "shape": {"circle", "square", "triangle"},
        "size": {"big", "small"},
        "color": {"red", "green", "blue"},
    }

    def __init__(self):
        nonterminals = {rule.lhs for rule in self.RULES}
        rules = {n: [] for n in nonterminals}
        for rule in self.RULES:
            rules[rule.lhs].append(rule)
        self.nonterminals = nonterminals
        self.rules = rules

    def sample(self, symbol=ROOT, last_rule=None, recursion=0):
        if isinstance(symbol, T):
            return symbol
        nt_rules = self.rules[symbol]
        if recursion == MAX_RECURSION - 1:
            nt_rules = [r for r in nt_rules if r != last_rule]
        rule = nt_rules[np.random.randint(len(nt_rules))]
        next_recursion = recursion + 1 if rule == last_rule else 0
        return Derivation(
            rule,
            tuple(self.sample(r, rule, next_recursion) for r in rule.rhs),
            meta={"recursion": recursion}
        )

    def category(self, function):
        for k, v in self.CATEGORIES.items():
            if function in v:
                return k
        return None

    def coherent(self, lf):
        for variable in lf.variables:
            functions = [t.function for t in lf.terms if variable in t.args]
            categories = [self.category(f) for f in functions]
            categories = [c for c in categories if c is not None]
            if len(categories) != len(set(categories)):
                return False
        return True

    def assign_split(self, command, demo):
        lf = command.meaning()

        if any(t.function == self.HOLD_OUT_ADVERB for t in lf.terms):
            return "adverb"

        if any(t.function == self.HOLD_OUT_ADJECTIVE for t in lf.terms):
            return "adjective"

        if any(
            t1.function == self.HOLD_OUT_COMPOSITION[0] 
            and t2.function == self.HOLD_OUT_COMPOSITION[1]
            and t1.args == t2.args
            for t1 in lf.terms
            for t2 in lf.terms
        ):
            return "composition"

        if any(
            t.function == self.HOLD_OUT_RECURSION[0]
            and t.meta is not None
            and t.meta["recursion"] == self.HOLD_OUT_RECURSION[1]
            for t in lf.terms
        ):
            return "recursion"

        for command, situation in demo:
            if command is None or command.event is None:
                continue
            event_lf = lf.select([command.event])
            args = [
                t.args[1] for t in event_lf.terms 
                if t.function == "patient" and t.args[0] == command.event
            ]
            if len(args) == 0:
                continue
            arg_var, = args
            arg_object = situation.grid[situation.agent_pos]

            if (arg_object == self.HOLD_OUT_OBJECT).all():
                return "object"

            if (
                any(
                    t.function == self.HOLD_OUT_APPLICATION[0] 
                    for t in lf.terms
                    if t.args[0] == arg_var
                )
                and (arg_object == self.HOLD_OUT_APPLICATION[1]).all()
            ):
                return "application"

        return "main"

class Derivation(object):
    def __init__(self, rule, children, meta=None):
        self.rule = rule
        self.children = children
        self.meta = meta

    def words(self):
        out = []
        for child in self.children:
            if isinstance(child, T):
                out.append(child.name)
            else:
                out += child.words()
        return tuple(out)

    # TODO canonical variable names, not memoization
    def meaning(self):
        if not hasattr(self, "_cached_lf"):
            child_meanings = [
                child.meaning() 
                for child in self.children 
                if isinstance(child, Derivation)
            ]
            meaning = self.rule.instantiate(*child_meanings, meta=self.meta)
            self._cached_lf = meaning
        return self._cached_lf

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

def main():
    g = Grammar()
    w = World()
    out = []
    complete = set()
    to_generate = 5000
    while len(complete) < to_generate:

        command = g.sample()
        if command.words() in complete:
            continue
        meaning = command.meaning()
        if not g.coherent(meaning):
            continue
        for j in range(100):
            situation = w.sample()
            demo = situation.demonstrate(meaning)
            if demo is not None:
                out.append((command, demo))
                complete.add(command.words())
                if (len(complete) + 1) % 100 == 0:
                    print("{:5d} / {:5d}".format(len(complete) + 1, to_generate))
                break

    splits = defaultdict(list)
    for command, demo in out:
        split = g.assign_split(command, demo)
        splits[split].append((command, demo))

    for split, data in splits.items():
        print(split, len(data))

if __name__ == "__main__":
    main()

# TODO path validator
# TODO data splits
# TODO code cleanup
# TODO write to file
