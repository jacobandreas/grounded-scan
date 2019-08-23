#!/usr/bin/env python3

from world import *

from collections import defaultdict, namedtuple
import numpy as np
import pronounceable

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

MAX_RECURSION = 2

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
    def __init__(self):
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

class Vocab(object):
    def __init__(
            self,
            verbs_intrans,
            verbs_trans,
            adverbs,
            nouns,
            adjectives
    ):
        self.verbs_intrans = verbs_intrans
        self.verbs_trans = verbs_trans
        self.adverbs = adverbs
        self.nouns = nouns
        self.adjectives = adjectives

    def rules(self):
        vv_intrans_rules = [
            VvIntransitive(name, (name, None, False)) 
            for name in self.verbs_intrans
        ]
        vv_trans_rules = [
            VvTransitive(name, (name, None, True))
            for name in self.verbs_trans
        ]
        rb_rules = [Rb(name, (None, name, None)) for name in self.adverbs]
        nn_rules = [Nn(name, weight) for name, weight in self.nouns]
        jj_rules = [Jj(name, weight) for name, weight in self.adjectives]

        return (
            vv_intrans_rules + vv_trans_rules + rb_rules + nn_rules + jj_rules
        )

    def random_adverb(self):
        return np.random.choice(self.adverbs)

    def random_adjective(self):
        return self.adjectives[np.random.randint(len(self.adjectives))][0]

    @classmethod
    def sample(
            cls,
            r_intrans=(2, 6),
            r_trans=(2, 6),
            r_adverb=(2, 6),
            r_noun=(2, 6),
            r_adjective=(2, 6)
    ):
        n_intrans = np.random.randint(*r_intrans)
        n_trans = np.random.randint(*r_trans)
        n_adverb = np.random.randint(*r_adverb)
        n_noun = np.random.randint(*r_noun)
        n_adjective = np.random.randint(*r_adjective)

        intrans = [pronounceable.generate_word() for _ in range(n_intrans)]
        trans = [pronounceable.generate_word() for _ in range(n_trans)]
        adverb = [pronounceable.generate_word() for _ in range(n_adverb)]
        noun = [
            (pronounceable.generate_word(), rand_weights())
            for _ in range(n_noun)
        ]
        adjective = [
            (pronounceable.generate_word(), rand_weights())
            for _ in range(n_adjective)
        ]
        return Vocab(intrans, trans, adverb, noun, adjective)

class Grammar(object):
    COMMON_RULES = [
        Root(),
        RootConj(),
        VpWrapper(),
        VpIntransitive(),
        VpTransitive(),
        Dp(),
        NpWrapper(),
        Np(),
        Nn("object", accept_weights()),
    ]

    def __init__(self, vocab):
        rule_list = self.COMMON_RULES + vocab.rules()
        nonterminals = {rule.lhs for rule in rule_list}
        rules = {n: [] for n in nonterminals}
        for rule in rule_list:
            rules[rule.lhs].append(rule)
        self.nonterminals = nonterminals
        self.rules = rules

        random_object = lambda: np.random.binomial(1, 0.5, size=N_ATTRIBUTES)

        self.hold_out_object = random_object()
        self.hold_out_adverb = vocab.random_adverb()
        self.hold_out_adjective = vocab.random_adjective()
        self.hold_out_application = (vocab.random_adjective(), random_object())
        self.hold_out_composition = (vocab.random_adjective(), vocab.random_adjective())
        self.hold_out_recursion = (vocab.random_adjective(), np.random.randint(MAX_RECURSION))
        self.categories = {
            "manner": set(vocab.adverbs),
            "shape": {n for n, _ in vocab.nouns},
            "color": set([v for v, _ in vocab.adjectives[:len(vocab.adjectives)//2]]),
            "size": set([v for v, _ in vocab.adjectives[len(vocab.adjectives)//2:]])
        }

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
        for k, v in self.categories.items():
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

        if any(t.function == self.hold_out_adverb for t in lf.terms):
            return "adverb"

        if any(t.function == self.hold_out_adjective for t in lf.terms):
            return "adjective"

        if any(
            t1.function == self.hold_out_composition[0] 
            and t2.function == self.hold_out_composition[1]
            and t1.args == t2.args
            for t1 in lf.terms
            for t2 in lf.terms
        ):
            return "composition"

        if any(
            t.function == self.hold_out_recursion[0]
            and t.meta is not None
            and t.meta["recursion"] == self.hold_out_recursion[1]
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

            if (arg_object == self.hold_out_object).all():
                return "object"

            if (
                any(
                    t.function == self.hold_out_application[0] 
                    for t in lf.terms
                    if t.args[0] == arg_var
                )
                and (arg_object == self.hold_out_application[1]).all()
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
