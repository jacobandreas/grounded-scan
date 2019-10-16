#!/usr/bin/env python3
from world import LogicalForm
from world import Term
from world import SemType
from world import ENTITY
from world import Variable
from world import Weights
from world import EVENT
from world import COLOR
from world import SIZE

from typing import List
from typing import Tuple
from typing import ClassVar
from collections import namedtuple
import numpy as np

Nonterminal = namedtuple("Nonterminal", "name")
Terminal = namedtuple("Terminal", "name")

ROOT = Nonterminal("ROOT")
VP = Nonterminal("VP")
VV_intransitive = Nonterminal("VV_intransitive")
VV_transitive = Nonterminal("VV_transitive")
RB = Nonterminal("RB")
DP = Nonterminal("DP")
NP = Nonterminal("NP")
NN = Nonterminal("NN")
JJ = Nonterminal("JJ")


# TODO cleaner
VAR_COUNTER = [0]
def free_var(sem_type):
    name = "x{}".format(VAR_COUNTER[0])
    VAR_COUNTER[0] += 1
    return Variable(name, sem_type)


class Rule(object):
    """
    Rule-class of form LHS -> RHS with method instantiate that defines its meaning.
    """
    def __init__(self, lhs: Nonterminal, rhs: List):
        self.lhs = lhs
        self.rhs = rhs
        self.sem_type = None

    def instantiate(self, *args, **kwargs):
        raise NotImplementedError()


class LexicalRule(Rule):
    """
    Rule of form Non-Terminal -> Terminal.
    """
    def __init__(self, lhs: Nonterminal, word: str, specs: Tuple, sem_type: SemType):
        super().__init__(lhs=lhs, rhs=[Terminal(word)])
        self.name = word
        self.sem_type = sem_type
        self.specs = specs

    def instantiate(self, meta=None, **kwargs) -> LogicalForm:
        # TODO a little fishy to have recursion meta here rather than in wrapper
        var = free_var(self.sem_type)
        return LogicalForm(
            variables=(var, ),
            terms=(Term(self.name, (var, ), specs=self.specs, meta=meta), )
        )


class Root(Rule):
    def __init__(self):
        super().__init__(lhs=ROOT, rhs=[VP])

    def instantiate(self, child, **kwargs):
        return child


class RootConj(Rule):
    def __init__(self):
        super().__init__(lhs=ROOT, rhs=[VP, Terminal("and"), ROOT])

    def instantiate(self, left_child, right_child, **kwargs):
        return LogicalForm(
            variables=left_child.variables + right_child.variables,
            terms=left_child.terms + right_child.terms + (Term("seq", (left_child.head, right_child.head)),)
        )


class VpWrapper(Rule):
    def __init__(self):
        super().__init__(lhs=VP, rhs=[RB, VP])

    def instantiate(self, rb, vp, meta, **kwargs):
        bound = rb.bind(vp.head)
        assert bound.variables[0] == vp.head
        return LogicalForm(variables=vp.variables + bound.variables[1:], terms=vp.terms + bound.terms)


class VpIntransitive(Rule):
    def __init__(self):
        super().__init__(lhs=VP, rhs=[VV_intransitive])

    def instantiate(self, vv, **kwargs):
        return vv


class VpTransitive(Rule):
    def __init__(self):
        super().__init__(lhs=VP, rhs=[VV_transitive, DP])

    def instantiate(self, vv, dp, **kwargs):
        role = Term("patient", (vv.head, dp.head))
        return LogicalForm(variables=vv.variables + dp.variables, terms=vv.terms + dp.terms + (role,))


class Dp(Rule):
    def __init__(self):
        super().__init__(lhs=DP, rhs=[Terminal("a"), NP])

    def instantiate(self, np, **kwargs):
        return np


class NpWrapper(Rule):
    def __init__(self):
        super().__init__(lhs=NP, rhs=[JJ, NP])

    def instantiate(self, jj, np, meta=None, **kwargs):
        bound = jj.bind(np.head)
        assert bound.variables[0] == np.head
        return LogicalForm(variables=np.variables + bound.variables[1:], terms=np.terms + bound.terms)


class Np(Rule):
    def __init__(self):
        super().__init__(lhs=NP, rhs=[NN])

    def instantiate(self, nn, **kwargs):
        return nn


class Grammar(object):
    """
    TODO(lauraruis): describe
    """
    COMMON_RULES = [Root(), RootConj(), VpWrapper(), VpIntransitive(), VpTransitive(), Dp(), NpWrapper(), Np(),
                    LexicalRule(lhs=NN, word="object", sem_type=ENTITY, specs=Weights(noun="object"))]

    def __init__(self, vocabulary: ClassVar, n_attributes=8, max_recursion=1):
        rule_list = self.COMMON_RULES + self.lexical_rules(vocabulary.verbs_intrans, vocabulary.verbs_trans,
                                                           vocabulary.adverbs, vocabulary.nouns,
                                                           vocabulary.color_adjectives, vocabulary.size_adjectives)
        nonterminals = {rule.lhs for rule in rule_list}
        self.rules = {nonterminal: [] for nonterminal in nonterminals}
        for rule in rule_list:
            self.rules[rule.lhs].append(rule)
        self.nonterminals = nonterminals

        self.categories = {
            "manner": set(vocabulary.adverbs),
            "shape": {n for n in vocabulary.nouns},
            "color": set([v for v in vocabulary.color_adjectives]),
            "size": set([v for v in vocabulary.size_adjectives])
        }
        self.max_recursion = max_recursion

    def lexical_rules(self, verbs_intrans: List[str], verbs_trans: List[str], adverbs: List[str], nouns: List[str],
                      color_adjectives: List[str], size_adjectives: List[str]) -> List[LexicalRule]:
        """
        Instantiate the lexical rules with the sampled words from the vocabulary.
        """
        vv_intrans_rules = [
            LexicalRule(lhs=VV_intransitive, word=verb, sem_type=EVENT, specs=Weights(action=verb, is_transitive=False))
            for verb in verbs_intrans
        ]
        vv_trans_rules = [
            LexicalRule(lhs=VV_transitive, word=verb, sem_type=EVENT, specs=Weights(action=verb, is_transitive=True))
            for verb in verbs_trans
        ]
        rb_rules = [LexicalRule(lhs=RB, word=word, sem_type=EVENT, specs=Weights(manner=word)) for word in adverbs]
        nn_rules = [LexicalRule(lhs=NN, word=word, sem_type=ENTITY, specs=Weights(noun=word)) for word in nouns]
        jj_rules = []
        jj_rules.extend([LexicalRule(lhs=JJ, word=word, sem_type=ENTITY, specs=Weights(adjective_type=COLOR))
                         for word in color_adjectives])
        jj_rules.extend([LexicalRule(lhs=JJ, word=word, sem_type=ENTITY, specs=Weights(adjective_type=SIZE))
                         for word in size_adjectives])
        return vv_intrans_rules + vv_trans_rules + rb_rules + nn_rules + jj_rules

    def sample(self, symbol=ROOT, last_rule=None, recursion=0):
        """
        Sample a command from the grammar by recursively sampling rules for each symbol.
        :param symbol: current node in constituency tree.
        :param last_rule:  previous rule sampled.
        :param recursion: recursion depth (increases if sample ruled is applied twice).
        :return: Derivation
        """
        # If the current symbol is a Terminal, close current branch and return.
        if isinstance(symbol, Terminal):
            return symbol
        nonterminal_rules = self.rules[symbol]

        # Filter out last rule if max recursion depth is reached.
        if recursion == self.max_recursion - 1:
            nonterminal_rules = [rule for rule in nonterminal_rules if rule != last_rule]

        # Sample a random rule.
        next_rule = nonterminal_rules[np.random.randint(len(nonterminal_rules))]
        next_recursion = recursion + 1 if next_rule == last_rule else 0
        return Derivation(
            next_rule,
            tuple(self.sample(rule, next_rule, next_recursion) for rule in next_rule.rhs),
            meta={"recursion": recursion}
        )

    def category(self, function):
        for category, values in self.categories.items():
            if function in values:
                return category
        return None

    def is_coherent(self, logical_form):
        """
        Returns true for coherent logical forms, false otherwise. A command's logical form is coherent if ..
        TODO(lauraruis) finish comment
        """
        for variable in logical_form.variables:
            functions = [term.function for term in logical_form.terms if variable in term.arguments]
            categories = [self.category(function) for function in functions]
            categories = [c for c in categories if c is not None]
            if len(categories) != len(set(categories)):
                return False
        return True


class Derivation(object):
    def __init__(self, rule, children, meta=None):
        self.rule = rule
        self.children = children
        self.meta = meta

    def words(self) -> tuple:
        """
        Recursively obtain all words of a derivation.
        """
        out = []
        for child in self.children:
            if isinstance(child, Terminal):
                out.append(child.name)
            else:
                out += child.words()
        return tuple(out)

    # TODO canonical variable names, not memoization
    def meaning(self) -> LogicalForm:
        if not hasattr(self, "_cached_logical_form"):
            child_meanings = [
                child.meaning()
                for child in self.children
                if isinstance(child, Derivation)
            ]
            meaning = self.rule.instantiate(*child_meanings, meta=self.meta)
            self._cached_logical_form = meaning
        return self._cached_logical_form
