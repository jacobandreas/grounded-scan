import pronounceable
import  numpy as np
from typing import List


class Vocabulary(object):
    """
    TODO: does this need to be a class?
    Method containing functionality for vocabulary. Allows for both random sampling of nonce-vocabulary by initializing
    through class method `sample` as well as setting user-defined words through default constructor.
    TODO(lauraruis): weights for nouns and adjectives as argument or generated?
    """
    def __init__(self, verbs_intrans: List[str], verbs_trans: List[str], adverbs: List[str], nouns: List[str],
                 color_adjectives: List[str], size_adjectives: List[str]):
        self.verbs_intrans = verbs_intrans
        self.verbs_trans = verbs_trans
        self.adverbs = adverbs
        self.nouns = nouns
        self.color_adjectives = color_adjectives
        self.size_adjectives = size_adjectives
        if len(color_adjectives) > 0 and len(size_adjectives) > 0:
            self.adjectives = color_adjectives + size_adjectives
        elif len(color_adjectives) > 0:
            self.adjectives = color_adjectives
        else:
            self.adjectives = size_adjectives
        self.n_attributes = len(self.nouns) * len(self.color_adjectives)

    @classmethod
    def sample(cls, range_intrans=(2, 6), range_trans=(2, 6), range_adverb=(2, 6), range_noun=(2, 6),
               range_adjective=(2, 6)):
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
        color_adjectives = adjectives[:n_adjective // 2]
        size_adjectives = adjectives[n_adjective // 2:]
        return Vocabulary(intransitive_verbs, transitive_verbs, adverbs, nouns, color_adjectives, size_adjectives)
