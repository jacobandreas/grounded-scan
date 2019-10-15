from grammar import VV_intransitive
from grammar import VV_transitive
from grammar import LexicalRule
from grammar import NN
from grammar import RB
from grammar import JJ
from world import EVENT
from world import ENTITY
from world import COLOR
from world import SIZE
from world import Weights

import pronounceable
import numpy as np
from typing import List


class Vocabulary(object):
    """
    Method containing functionality for vocabulary. Allows for both random sampling of nonce-vocabulary by initializing
    through class method `sample` as well as setting user-defined words through default constructor.
    TODO(lauraruis): weights for nouns and adjectives as argument or generated?
    """
    def __init__(self, verbs_intrans: List[str], verbs_trans: List[str], adverbs: List[str], nouns: List[str],
                 color_adjectives: List[str], size_adjectives: List[str], n_attributes: int = 6):
        self.verbs_intrans = verbs_intrans
        self.verbs_trans = verbs_trans
        self.adverbs = adverbs
        self.nouns = nouns
        self.color_adjectives = color_adjectives
        self.size_adjectives = size_adjectives
        self.adjectives = color_adjectives + size_adjectives
        self.n_attributes = len(self.nouns) * len(self.color_adjectives)

    def lexical_rules(self) -> List[LexicalRule]:
        """
        Instantiate the lexical rules with the sampled words from the vocabulary.
        """
        vv_intrans_rules = [
            LexicalRule(lhs=VV_intransitive, word=verb, sem_type=EVENT, specs=Weights(action=verb, is_transitive=False))
            for verb in self.verbs_intrans
        ]
        vv_trans_rules = [
            LexicalRule(lhs=VV_transitive, word=verb, sem_type=EVENT, specs=Weights(action=verb, is_transitive=True))
            for verb in self.verbs_trans
        ]
        rb_rules = [LexicalRule(lhs=RB, word=word, sem_type=EVENT, specs=Weights(manner=word)) for word in self.adverbs]
        nn_rules = [LexicalRule(lhs=NN, word=word, sem_type=ENTITY, specs=Weights(noun=word)) for word in self.nouns]
        jj_rules = []
        jj_rules.extend([LexicalRule(lhs=JJ, word=word, sem_type=ENTITY, specs=Weights(adjective_type=COLOR))
                         for word in self.color_adjectives])
        jj_rules.extend([LexicalRule(lhs=JJ, word=word, sem_type=ENTITY, specs=Weights(adjective_type=SIZE))
                         for word in self.size_adjectives])
        return vv_intrans_rules + vv_trans_rules + rb_rules + nn_rules + jj_rules

    def random_adverb(self):
        return np.random.choice(self.adverbs)

    def random_adjective(self):
        return self.adjectives[np.random.randint(len(self.adjectives))][0]

    @classmethod
    def sample(cls, range_intrans=(2, 6), range_trans=(2, 6), range_adverb=(2, 6), range_noun=(2, 6),
               range_adjective=(2, 6), n_attributes: int = 8):
        """
        Sample random nonce-words and initialize the vocabulary with these.
        """
        # Sample amount of words to generate within range
        n_intrans = np.random.randint(*range_intrans)
        n_trans = np.random.randint(*range_trans)
        n_adverb = np.random.randint(*range_adverb)
        n_noun = np.random.randint(*range_noun)
        n_adjective = np.random.randint(*range_adjective)

        # Generate random nonce-words
        intransitive_verbs = [pronounceable.generate_word() for _ in range(n_intrans)]
        transitive_verbs = [pronounceable.generate_word() for _ in range(n_trans)]
        adverbs = [pronounceable.generate_word() for _ in range(n_adverb)]
        nouns = [pronounceable.generate_word() for _ in range(n_noun)]
        adjectives = [pronounceable.generate_word() for _ in range(n_adjective)]
        color_adjectives = adjectives[:n_adjective//2]
        size_adjectives = adjectives[n_adjective//2:]
        return Vocabulary(intransitive_verbs, transitive_verbs, adverbs, nouns, color_adjectives, size_adjectives,
                          n_attributes=n_attributes)
