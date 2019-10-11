from grammar import Grammar
from vocabulary import Vocabulary
from world import World
from helpers import random_weights
from helpers import visualize_action_sequence

import argparse
import os
from collections import defaultdict


def main():

    parser = argparse.ArgumentParser(description="Grounded SCAN")
    parser.add_argument('--max_recursion', type=int, default=2, help='Max. recursion depth allowed when sampling from '
                                                                     'grammar.')
    parser.add_argument('--n_attributes', type=int, default=8, help='Number of attributes to ..')  # TODO
    parser.add_argument('--examples_to_generate', type=int, default=8, help='Number of command-demonstration examples'
                                                                            ' to generate.')
    parser.add_argument('--grid_size', type=int, default=5, help='Number of rows (and columns) in the grid world.')
    parser.add_argument('--min_objects', type=int, default=1, help='Minimum amount of objects to put in the grid '
                                                                   'world.')
    parser.add_argument('--max_objects', type=int, default=2, help='Maximum amount of objects to put in the grid '
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
    if flags['sample_vocab']:
        vocabulary = Vocabulary.sample()
    else:
        # TODO: read from file
        verbs_intrans = ['walk', 'run', 'jump']
        verbs_trans = ['push', 'kick']
        adverbs = ['quickly', 'slowly']
        nouns = [('circle', random_weights(flags['n_attributes'])),
                 ('cylinder', random_weights(flags['n_attributes'])), ('wall', random_weights(flags['n_attributes']))]
        adjectives = [('big', random_weights(flags['n_attributes'])), ('small', random_weights(flags['n_attributes'])),
                      ('red', random_weights(flags['n_attributes'])), ('blue', random_weights(flags['n_attributes']))]
        vocabulary = Vocabulary(verbs_intrans=verbs_intrans, verbs_trans=verbs_trans, adverbs=adverbs, nouns=nouns,
                                adjectives=adjectives)
    grammar = Grammar(vocabulary, n_attributes=flags['n_attributes'], max_recursion=flags['max_recursion'])

    # Initialize the world
    world = World(grid_size=flags['grid_size'], n_attributes=flags['n_attributes'], min_objects=flags['min_objects'],
                  max_objects=flags['max_objects'])

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
        for j in range(100):

            # Place specific items in the world.
            if not flags['sample_vocab']:
                # TODO: read from file
                objects = [("ball", "red", (2, 2)), ("wall", "blue", (3, 1))]  # positions are [col, row]
                situation = world.initialize(objects, agent_pos=(0, 0))
            # Place random items at random locations.
            else:
                situation = world.sample()

            # A demonstration is a sequence of commands and situations.
            demonstration = situation.demonstrate(meaning)
            if demonstration:
                examples.append((command, demonstration))
                unique_commands.add(command.words())
                if (len(unique_commands) + 1) % 100 == 0:
                    print("{:5d} / {:5d}".format(len(unique_commands) + 1, flags['examples_to_generate']))
                break

    # Assign examples to data splits.
    splits = defaultdict(list)
    for command, demonstration in examples:
        print("\nCommand: " + ' '.join(command.words()))
        print("Meaning: ", command.meaning())

        split = grammar.assign_split(command, demonstration)
        splits[split].append((command, demonstration))

    # Visualize one command.
    visualize_action_sequence(examples[0], flags['visualization_dir'])

    for split, data in splits.items():
        print(split, len(data))


if __name__ == "__main__":
    main()

# TODO path validator
# TODO write to file
