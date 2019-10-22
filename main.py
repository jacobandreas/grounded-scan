# TODO: build in option to do nonce words
# TODO: actions for transitive verbs
# TODO: fix issues with 'object'
# TODO: read words from file for vocab
# TODO: splits
# TODO: remove unnecessary stuff from minigrid.py
# TODO: make objects either rollable or not
# TODO: generate all rules -> done
# TODO: generate all situations, generate all command, situation -> demonstration pairs
from grammar import Grammar
from dataset import GroundedScan

import argparse
import os


def main():

    parser = argparse.ArgumentParser(description="Grounded SCAN")
    parser.add_argument('--max_recursion', type=int, default=2, help='Max. recursion depth allowed when sampling from '
                                                                     'grammar.')
    parser.add_argument('--n_attributes', type=int, default=8, help='Number of attributes to ..')  # TODO
    parser.add_argument('--examples_to_generate', type=int, default=10, help='Number of command-demonstration examples'
                                                                            ' to generate.')
    parser.add_argument('--grid_size', type=int, default=15, help='Number of rows (and columns) in the grid world.')
    parser.add_argument('--min_objects', type=int, default=8, help='Minimum amount of objects to put in the grid '
                                                                   'world.')
    parser.add_argument('--max_objects', type=int, default=9, help='Maximum amount of objects to put in the grid '
                                                                   'world.')
    parser.add_argument('--read_vocab_from_file', dest='sample_vocab', default=False, action='store_false')
    parser.add_argument('--sample_vocab', dest='sample_vocab', default=False, action='store_true')
    parser.add_argument('--visualization_dir', type=str, default='visualizations', help='Path to a folder in which '
                                                                                        'visualizations should be '
                                                                                        'stored.')
    flags = vars(parser.parse_args())

    # Create directory for visualizations if it doesn't exist.
    if flags['visualization_dir']:
        visualization_path = os.path.join(os.getcwd(), flags['visualization_dir'])
        if not os.path.exists(visualization_path):
            os.mkdir(visualization_path)

    # Sample a vocabulary and a grammar with rules of form NT -> T and T -> {words from vocab}.
    if flags['sample_vocab']:  # TODO
        verbs_intrans = ['walk', 'run', 'jump']
        verbs_trans = ['roll', 'push']
        adverbs = ['quickly', 'slowly', 'while zigzagging', 'while spinning', 'cautiously', 'hesitantly']
        adverbs = ['while zigzagging']
        nouns = ['circle', 'square']
        color_adjectives = ['red', 'blue']
        # Size adjectives sorted from smallest to largest.
        size_adjectives = ['small', 'big']
        grounded_scan = GroundedScan(intransitive_verbs=verbs_intrans, transitive_verbs=verbs_trans, adverbs=adverbs,
                                     nouns=nouns, color_adjectives=color_adjectives, size_adjectives=size_adjectives,
                                     save_directory=flags["visualization_dir"], grid_size=flags["grid_size"])
    else:
        # TODO: read from file
        verbs_intrans = ['walk', 'run', 'jump']
        verbs_trans = ['roll', 'push']
        adverbs = ['quickly', 'slowly', 'while zigzagging', 'while spinning', 'cautiously', 'hesitantly']
        # adverbs = ['while zigzagging']
        nouns = ['circle', 'square']
        color_adjectives = ['red', 'blue']
        # Size adjectives sorted from smallest to largest.
        size_adjectives = ['small', 'big']
        grounded_scan = GroundedScan(intransitive_verbs=verbs_intrans, transitive_verbs=verbs_trans, adverbs=adverbs,
                                     nouns=nouns, color_adjectives=color_adjectives, size_adjectives=size_adjectives,
                                     save_directory=flags["visualization_dir"], grid_size=flags["grid_size"])

    grammar = Grammar(grounded_scan.vocabulary, max_recursion=flags['max_recursion'])

    # Structures for keeping track of examples
    examples = []
    unique_commands = set()

    # Generate examples of a command with a situation mapping to a demonstration.
    while len(unique_commands) < flags['examples_to_generate']:
        command = grammar.sample()
        if command.words() in unique_commands:
            continue
        meaning = command.meaning()
        if not grammar.is_coherent(meaning):
            continue

        # For each command sample a situation of the world and determine a ground-truth demonstration sequence.
        for j in range(2):  # TODO: change

            # Place specific items in the world.
            if not flags['sample_vocab']:
                initial_situation = grounded_scan.sample_situation(num_objects=4)

                # demonstrate the meaning of the command based on the current situation
                demonstration = grounded_scan.demonstrate_command(' '.join(command.words()), meaning, initial_situation)
                grounded_scan.visualize_command(command=' '.join(command.words()), initial_situation=initial_situation,
                                                demonstration=demonstration)
                if demonstration:
                    examples.append((command, initial_situation, demonstration))
                    unique_commands.add(command.words())
                    if (len(unique_commands) + 1) % 100 == 0:
                        print("{:5d} / {:5d}".format(len(unique_commands) + 1, flags['examples_to_generate']))
                    break
            # # Place random items at random locations.
            # else:
            #     situation = world.sample()


if __name__ == "__main__":
    main()

# TODO path validator
# TODO write to file
