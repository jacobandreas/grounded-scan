# TODO: build in option to do nonce words
# TODO: splits
# TODO: remove unnecessary stuff from minigrid.py
# TODO: implement generate_all_situations for conjuncations (i.e. with multiple targets)
# TODO: make target_commands an enum like Actions in minigrid
# TODO: visualize_data_example in dataset.py
from dataset import GroundedScan
from grammar import Derivation

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
    parser.add_argument('--intransitive_verbs', type=str, default='walk', help='Comma-separated list of '
                                                                               'intransitive verbs.')
    parser.add_argument('--transitive_verbs', type=str, default='push', help='Comma-separated list of '
                                                                             'transitive verbs.')
    parser.add_argument('--adverbs', type=str,
                        default='quickly, slowly, while zigzagging, while spinning, cautiously, hesitantly',
                        help='Comma-separated list of adverbs.')
    parser.add_argument('--nouns', type=str, default='circle,square,cylinder', help='Comma-separated list of nouns.')
    parser.add_argument('--color_adjectives', type=str, default='green,red,blue', help='Comma-separated list of '
                                                                                       'colors.')
    parser.add_argument('--size_adjectives', type=str, default='small,big', help='Comma-separated list of sizes.')
    parser.add_argument('--min_object_size', type=int, default=1, help='Smallest object size.')
    parser.add_argument('--max_object_size', type=int, default=4, help='Biggest object size.')
    flags = vars(parser.parse_args())

    # Create directory for visualizations if it doesn't exist.
    if flags['visualization_dir']:
        visualization_path = os.path.join(os.getcwd(), flags['visualization_dir'])
        if not os.path.exists(visualization_path):
            os.mkdir(visualization_path)

    # Sample a vocabulary and a grammar with rules of form NT -> T and T -> {words from vocab}.
    # grounded_scan = GroundedScan(intransitive_verbs=flags["intransitive_verbs"].split(','),
    #                              transitive_verbs=flags["transitive_verbs"].split(','),
    #                              adverbs=flags["adverbs"].split(','), nouns=flags["nouns"].split(','),
    #                              color_adjectives=flags["color_adjectives"].split(','),
    #                              size_adjectives=flags["size_adjectives"].split(','),
    #                              min_object_size=flags["min_object_size"], max_object_size=flags["max_object_size"],
    #                              save_directory=flags["visualization_dir"], grid_size=flags["grid_size"])

    # Generate all possible commands from the grammar
    # grounded_scan.get_data_pairs()
    # grounded_scan.print_dataset_statistics()
    grounded_scan = GroundedScan.load_dataset_from_file("visualizations/dataset.txt", save_directory="visualizations")
    grounded_scan.print_dataset_statistics()
    # print("Saved dataset to {}".format(dataset_path))

    grounded_scan.visualize_data_examples(10)


if __name__ == "__main__":
    main()
