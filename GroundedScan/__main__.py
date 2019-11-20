# TODO: build in option to do nonce words
# TODO: splits
# TODO: remove unnecessary stuff from minigrid.py
# TODO: implement generate_all_situations for conjuncations (i.e. with multiple targets)
# TODO: make target_commands an enum like Actions in minigrid
# TODO: pushing objects over other objects? (concern about overlapping objects)
# TODO: make agent different thing (different color enough?)
# TODO: make pushing objects starting when adjacent to object (concern regarding overlapping objects?)
# TODO: what to do about pushing something that's on the border
# TODO: make initial situation image part of data examples
# TODO: make message to group with design choices (different situations per referred target, non-overlapping objects)
# TODO: logging instead of printing
from GroundedScan.dataset import GroundedScan
from GroundedScan.dataset_test import run_all_tests
from GroundedScan.world import Situation

import argparse
import os
import logging
import json

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M')
logger = logging.getLogger(__name__)


def main():

    parser = argparse.ArgumentParser(description="Grounded SCAN")

    # General arguments.
    parser.add_argument('--mode', type=str, default='execute_commands',
                        help='Generate (mode=generate) data, run tests (mode=test) or execute commands from a file'
                             '(mode=execute_commands).')
    parser.add_argument('--load_dataset_from', type=str, default='', help='Path to file with dataset.')
    parser.add_argument('--visualization_dir', type=str, default='visualizations', help='Path to a folder in which '
                                                                                        'visualizations should be '
                                                                                        'stored.')
    parser.add_argument('--predicted_commands_file', type=str, default='predict.json',
                        help='Path to a file with predictions.')
    parser.add_argument('--save_dataset_as', type=str, default='dataset.txt', help='Filename to save dataset in.')
    parser.add_argument("--count_equivalent_examples", dest="count_equivalent_examples", default=False,
                        action="store_true")

    # Dataset arguments.
    parser.add_argument('--num_resampling', type=int, default=10, help='Number of time to resample a semantically '
                                                                       'equivalent situation (which will likely result'
                                                                       ' in different situations in terms of object '
                                                                       'locations).')
    parser.add_argument('--visualize_per_template', type=int, default=0, help='How many visualization to generate per '
                                                                              'command template.')
    parser.add_argument('--train_percentage', type=float, default=.8,
                        help='Percentage of examples to put in the training set (rest is test set).')

    # World arguments/
    parser.add_argument('--grid_size', type=int, default=6, help='Number of rows (and columns) in the grid world.')
    parser.add_argument('--min_objects', type=int, default=2, help='Minimum amount of objects to put in the grid '
                                                                   'world.')
    parser.add_argument('--max_objects', type=int, default=8, help='Maximum amount of objects to put in the grid '
                                                                   'world.')
    parser.add_argument('--sample_vocab', dest='sample_vocab', default=False, action='store_true')  # TODO
    parser.add_argument('--min_object_size', type=int, default=1, help='Smallest object size.')
    parser.add_argument('--max_object_size', type=int, default=4, help='Biggest object size.')
    parser.add_argument('--other_objects_sample_percentage', type=float, default=.5,
                        help='Percentage of possible objects distinct from the target to place in the world.')

    # Grammar and Vocabulary arguments
    parser.add_argument('--intransitive_verbs', type=str, default='walk', help='Comma-separated list of '
                                                                               'intransitive verbs.')
    parser.add_argument('--transitive_verbs', type=str, default='push', help='Comma-separated list of '
                                                                             'transitive verbs.')
    parser.add_argument('--adverbs', type=str,
                        default='quickly,slowly,while zigzagging,while spinning,cautiously,hesitantly',
                        help='Comma-separated list of adverbs.')
    parser.add_argument('--nouns', type=str, default='circle,square,cylinder', help='Comma-separated list of nouns.')
    parser.add_argument('--color_adjectives', type=str, default='green,red,blue', help='Comma-separated list of '
                                                                                       'colors.')
    parser.add_argument('--size_adjectives', type=str, default='small,big', help='Comma-separated list of sizes.')
    parser.add_argument('--max_recursion', type=int, default=2, help='Max. recursion depth allowed when sampling from '
                                                                     'grammar.')

    flags = vars(parser.parse_args())

    # Sample a vocabulary and a grammar with rules of form NT -> T and T -> {words from vocab}.
    grounded_scan = GroundedScan(intransitive_verbs=flags["intransitive_verbs"].split(','),
                                 transitive_verbs=flags["transitive_verbs"].split(','),
                                 adverbs=flags["adverbs"].split(','), nouns=flags["nouns"].split(','),
                                 color_adjectives=flags["color_adjectives"].split(','),
                                 size_adjectives=flags["size_adjectives"].split(','),
                                 min_object_size=flags["min_object_size"],
                                 max_object_size=flags["max_object_size"],
                                 save_directory=flags["visualization_dir"], grid_size=flags["grid_size"])

    # Create directory for visualizations if it doesn't exist.
    if flags['visualization_dir']:
        visualization_path = os.path.join(os.getcwd(), flags['visualization_dir'])
        if not os.path.exists(visualization_path):
            os.mkdir(visualization_path)

    if flags['mode'] == 'generate':

        # Generate all possible commands from the grammar
        grounded_scan.get_data_pairs(num_resampling=flags['num_resampling'],
                                     other_objects_sample_percentage=flags['other_objects_sample_percentage'],
                                     visualize_per_template=flags['visualize_per_template'],
                                     train_percentage=flags['train_percentage'])
        grounded_scan.save_dataset_statistics(split="train")
        grounded_scan.save_dataset_statistics(split="test")
        dataset_path = grounded_scan.save_dataset(flags['save_dataset_as'])
        logger.info("Saved dataset to {}".format(dataset_path))
        if flags['count_equivalent_examples']:
            logger.info("Equivalent examples in train and testset: {}".format(grounded_scan.count_equivalent_examples(
                "train", "test")))
        grounded_scan.visualize_data_examples()
    elif flags['mode'] == 'execute_commands':
        assert os.path.exists(flags["predicted_commands_file"]), "Trying to execute commands from non-existing file: "\
                                                                 "{}".format(flags["predicted_commands_file"])
        grounded_scan.visualize_prediction(flags["predicted_commands_file"])
    elif flags['mode'] == 'test':
        logger.info("Running all tests..")
        run_all_tests()
    else:
        raise ValueError("Unknown value for command-line argument 'mode'={}.".format(flags['mode']))


if __name__ == "__main__":
    main()
