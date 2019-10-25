#!/usr/bin/env python3
# TODO: max recursion choice per rule
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
from itertools import product


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
    def __init__(self, lhs: Nonterminal, rhs: List, max_recursion=2):
        self.lhs = lhs
        self.rhs = rhs
        self.sem_type = None
        self.max_recursion = max_recursion

    def instantiate(self, *args, **kwargs):
        raise NotImplementedError()


class LexicalRule(Rule):
    """
    Rule of form Non-Terminal -> Terminal.
    """
    def __init__(self, lhs: Nonterminal, word: str, specs: Weights, sem_type: SemType):
        super().__init__(lhs=lhs, rhs=[Terminal(word)], max_recursion=1)
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
    def __init__(self, max_recursion=0):
        super().__init__(lhs=ROOT, rhs=[VP, Terminal("and"), ROOT], max_recursion=max_recursion)

    def instantiate(self, left_child, right_child, **kwargs):
        return LogicalForm(
            variables=left_child.variables + right_child.variables,
            terms=left_child.terms + right_child.terms + (Term("seq", (left_child.head, right_child.head)),)
        )


class VpWrapper(Rule):
    def __init__(self, max_recursion=0):
        super().__init__(lhs=VP, rhs=[VP, RB], max_recursion=max_recursion)

    def instantiate(self, rb, vp, meta, **kwargs):
        bound = rb.bind(vp.head)
        assert bound.variables[0] == vp.head
        return LogicalForm(variables=vp.variables + bound.variables[1:], terms=vp.terms + bound.terms)


class VpIntransitive(Rule):
    def __init__(self):
        super().__init__(lhs=VP, rhs=[VV_intransitive, Terminal("to"), DP])

    def instantiate(self, vv, dp, **kwargs):
        role = Term("patient", (vv.head, dp.head))
        return LogicalForm(variables=vv.variables + dp.variables, terms=vv.terms + dp.terms + (role,))


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
    def __init__(self, max_recursion=0):
        super().__init__(lhs=NP, rhs=[JJ, NP], max_recursion=max_recursion)

    def instantiate(self, jj, np, meta=None, **kwargs):
        bound = jj.bind(np.head)
        assert bound.variables[0] == np.head
        return LogicalForm(variables=np.variables + bound.variables[1:], terms=np.terms + bound.terms)


class Np(Rule):
    def __init__(self):
        super().__init__(lhs=NP, rhs=[NN])

    def instantiate(self, nn, **kwargs):
        return nn


class Derivation(object):
    """
    Holds a constituency tree that makes up a sentence. Can be used to obtain the meaning of a sentence in terms
    of a Logical Form. The meaning of a derivation is made up of the meaning of its children.
    """

    def __init__(self, rule, children=None, meta=None):
        self.rule = rule
        self.lhs = rule.lhs
        self.children = children
        self.meta = meta

    @classmethod
    def from_rules(cls, rules: list, symbol=ROOT, lexicon=None):
        """Recursively form a derivation from a rule list that has been constructed in a depth-first manner,
        use the lexicon for the Lexical Rules at the leafs of the constituency tree."""
        # If the current symbol is a Terminal, close current branch and return.
        if isinstance(symbol, Terminal):
            return symbol
        if symbol not in lexicon.keys():
            next_rule = rules.pop()
        else:
            next_rule = lexicon[symbol].pop()

        return Derivation(
            next_rule,
            tuple(cls.from_rules(rules, symbol=next_symbol, lexicon=lexicon) for next_symbol in next_rule.rhs)
        )

    def words(self) -> tuple:
        """Obtain all words of a derivation by combining the words of all the children."""
        out = []
        for child in self.children:
            if isinstance(child, Terminal):
                out.append(child.name)
            else:
                out += child.words()
        return tuple(out)

    # TODO canonical variable names, not memoization
    def meaning(self) -> LogicalForm:
        """Recursively define the meaning of the derivation by instantiating the meaning of each child."""
        if not hasattr(self, "_cached_logical_form"):
            child_meanings = [
                child.meaning()
                for child in self.children
                if isinstance(child, Derivation)
            ]
            meaning = self.rule.instantiate(*child_meanings, meta=self.meta)
            self._cached_logical_form = meaning
        return self._cached_logical_form


class Template(object):
    """
    A template is a constituency-tree without lexical rules. From a template together with a lexicon, multiple
    constituency trees can be formed.
    """

    def __init__(self):
        self._left_values = []
        self._right_values = []
        self._leftmost_nonterminal = None
        self.rules = []

    def add_value(self, value, expandable):
        if expandable and not self._leftmost_nonterminal:
            self._leftmost_nonterminal = value
        elif self._leftmost_nonterminal:
            self._right_values.append(value)
        else:
            self._left_values.append(value)

    def has_nonterminal(self):
        return self._leftmost_nonterminal is not None

    def get_leftmost_nonterminal(self):
        assert self.has_nonterminal(), "Trying to get a NT but none present in this derivation."
        return self._leftmost_nonterminal

    def expand_leftmost_nonterminal(self, rule, expandables):
        new_derivation = Template()
        new_derivation_symbols = self._left_values + rule.rhs + self._right_values
        new_derivation.rules = self.rules.copy()
        new_derivation.rules.append(rule)
        for value in new_derivation_symbols:
            if value in expandables:
                new_derivation.add_value(value, expandable=True)
            else:
                new_derivation.add_value(value, expandable=False)
        return new_derivation

    def to_derivation(self):
        assert not self.has_nonterminal(), "Trying to write a non-terminal to a string."
        self.rules.reverse()
        return self._left_values, self.rules


class Grammar(object):
    """
    TODO(lauraruis): describe
    """
    COMMON_RULES = [Root(), RootConj(max_recursion=2), VpWrapper(), VpIntransitive(), VpTransitive(), Dp(),
                    NpWrapper(max_recursion=2), Np()]

    def __init__(self, vocabulary: ClassVar, max_recursion=1):
        rule_list = self.COMMON_RULES + self.lexical_rules(vocabulary.verbs_intrans, vocabulary.verbs_trans,
                                                           vocabulary.adverbs, vocabulary.nouns,
                                                           vocabulary.color_adjectives, vocabulary.size_adjectives)
        nonterminals = {rule.lhs for rule in rule_list}
        self.rules = {nonterminal: [] for nonterminal in nonterminals}
        self.vocabulary = vocabulary
        for rule in rule_list:
            self.rules[rule.lhs].append(rule)
        self.nonterminals = nonterminals
        self.expandables = set(rule.lhs for rule in rule_list if not isinstance(rule, LexicalRule))

        self.categories = {
            "manner": set(vocabulary.adverbs),
            "shape": {n for n in vocabulary.nouns},
            "color": set([v for v in vocabulary.color_adjectives]),
            "size": set([v for v in vocabulary.size_adjectives])
        }
        self.word_to_category = {}
        for category, words in self.categories.items():
            for word in words:
                self.word_to_category[word] = category

        self.max_recursion = max_recursion
        self.all_templates = []
        self.all_derivations = []

    @ staticmethod
    def lexical_rules(verbs_intrans: List[str], verbs_trans: List[str], adverbs: List[str], nouns: List[str],
                      color_adjectives: List[str], size_adjectives: List[str]) -> list:
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
            tuple(self.sample(next_symbol, next_rule, next_recursion) for next_symbol in next_rule.rhs),
            meta={"recursion": recursion}
        )

    def generate_all(self, current_template: Template, all_templates: list, rule_use_counter: dict):

        # If the template contains no non-terminals, we close this branch.
        if not current_template.has_nonterminal():
            all_templates.append(current_template.to_derivation())
            return

        # Retrieve the leftmost non-terminal to expand.
        leftmost_nonterminal = current_template.get_leftmost_nonterminal()

        # Get all possible RHS replacements and start a new derivation branch for each of them.
        rules_leftmost_nonterminal = self.rules[leftmost_nonterminal]
        for rule_leftmost_nonterminal in rules_leftmost_nonterminal:

            # Lexical rules are not expandable
            if isinstance(rule_leftmost_nonterminal, LexicalRule):
                continue

            # Each branch gets its own rule usage counter.
            rule_use_counter_copy = rule_use_counter.copy()

            # If this rule has already been applied in the current branch..
            if rule_leftmost_nonterminal in rule_use_counter_copy.keys():

                # ..do not use it again if it has been applied more than a maximum allowed number of times.
                if rule_use_counter[rule_leftmost_nonterminal] >= rule_leftmost_nonterminal.max_recursion:
                    continue
                rule_use_counter_copy[rule_leftmost_nonterminal] += 1
            else:
                rule_use_counter_copy[rule_leftmost_nonterminal] = 1

            # Get the next derivation by replacing the leftmost NT with its RHS.
            next_template = current_template.expand_leftmost_nonterminal(rule_leftmost_nonterminal,
                                                                         self.expandables)

            # Start a new derivation branch for this RHS.
            self.generate_all(next_template, all_templates, rule_use_counter_copy)

    def form_commands_from_template(self, derivation_template: list, derivation_rules: list):
        """
        Replaces all NT's in a template with the possible T's and forms all possible commands with those.
        If multiple the same NT's follow each other, e.g. a JJ JJ JJ NN, for each following JJ the possible words
        will be halved over the possibilities, meaning no words will repeat themselves (e.g. the red red circle),
        this does mean that whenever the max. recursion depth for a rule is larger than the log(n) where n is the number
        of words for that particular rule, this does not have an effect.
        :param derivation_template: list of NT's, e.g. [VV_intrans, 'to', 'a', JJ, JJ, NN, RB]
        :param derivation_rules: list of rules that build up the constituency tree for this template
        :return: all possible combinations where all NT's are replaced by the words from the lexicon.
        """

        # In the templates, replace each lexical rule with the possible words from the lexicon
        replaced_template = []
        previous_symbol = None
        lexicon = {}
        for symbol in derivation_template:
            if isinstance(symbol, Nonterminal):
                possible_words = [s.name for s in self.rules[symbol]]
                for rule in self.rules[symbol]:
                    lexicon[rule.name] = rule
                if previous_symbol == symbol:
                    previous_words = replaced_template.pop()
                    first_half = previous_words[:len(previous_words) // 2]
                    second_half = previous_words[len(previous_words) // 2:]
                    replaced_template.append(first_half)
                    replaced_template.append(second_half)
                else:
                    replaced_template.append(possible_words)
            else:
                lexicon[symbol.name] = symbol
                replaced_template.append([symbol.name])
            previous_symbol = symbol

        # Generate all possible commands from the templates.
        all_commands = [command for command in product(*replaced_template)]
        all_derivations = []
        for command in all_commands:
            command_lexicon = {}
            for word, symbol in zip(command, derivation_template):
                if symbol not in command_lexicon:
                    command_lexicon[symbol] = [lexicon[word]]
                else:
                    command_lexicon[symbol] = [lexicon[word]] + command_lexicon[symbol]
            derivation = Derivation.from_rules(derivation_rules.copy(), symbol=ROOT, lexicon=command_lexicon)
            assert ' '.join(derivation.words()) == ' '.join(command), "Derivation and command not the same."
            all_derivations.append(derivation)
        return all_derivations

    def generate_all_commands(self):

        # Generate all possible templates from the grammar.
        initial_template = Template()
        initial_template.add_value(value=ROOT, expandable=True)
        self.generate_all(current_template=initial_template, all_templates=self.all_templates,
                          rule_use_counter={})

        # For each template, form all possible commands by combining it with the lexicon.
        for derivation_template, derivation_rules in self.all_templates:
            derivations = self.form_commands_from_template(derivation_template, derivation_rules)
            self.all_derivations.extend(derivations)

    def category(self, function):
        return self.word_to_category.get(function)

    def is_coherent(self, logical_form):
        """
        Returns true for coherent logical forms, false otherwise. A command's logical form is coherent the
        arguments of a variable have all unique categories. E.g. in coherent would be: 'the red blue circle'.
        """
        for variable in logical_form.variables:
            functions = [term.function for term in logical_form.terms if variable in term.arguments]
            categories = [self.category(function) for function in functions]
            categories = [c for c in categories if c is not None]
            if len(categories) != len(set(categories)):
                return False
        return True

