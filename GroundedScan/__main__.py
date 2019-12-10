# TODO: build in option to do nonce words (fix todo's with nonce)
# TODO: splits
# TODO: build in few-shot (specified few) generalization option
# TODO: implement generate_all_situations for conjuncations (i.e. with multiple targets)
# TODO: make target_commands an enum like Actions in minigrid
# TODO: pushing objects over other objects? (concern about overlapping objects)
# TODO: make agent different thing (different color enough?)
# TODO: make pushing objects starting when adjacent to object (concern regarding overlapping objects?)
# TODO: what to do about pushing something that's on the border (currently just not pushed, doesn't make sense)
# TODO: logging instead of printing
# TODO: count how often an example ends up in multiple splits
from GroundedScan.dataset import GroundedScan
from GroundedScan.dataset_test import run_all_tests

import argparse
import os
import logging

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
    parser.add_argument('--output_directory', type=str, default='output', help='Path to a folder in which '
                                                                               'all outputs should be '
                                                                               'stored.')
    parser.add_argument('--predicted_commands_file', type=str, default='predict.json',
                        help='Path to a file with predictions.')
    parser.add_argument('--save_dataset_as', type=str, default='dataset.txt', help='Filename to save dataset in.')
    parser.add_argument("--count_equivalent_examples", dest="count_equivalent_examples", default=False,
                        action="store_true")
    parser.add_argument("--only_save_errors", dest="only_save_errors", default=False,
                        action="store_true")

    # Dataset arguments.
    parser.add_argument('--split', type=str, default='uniform', choices=['uniform', 'generalization'])
    parser.add_argument('--num_resampling', type=int, default=10, help='Number of time to resample a semantically '
                                                                       'equivalent situation (which will likely result'
                                                                       ' in different situations in terms of object '
                                                                       'locations).')
    parser.add_argument('--visualize_per_template', type=int, default=0, help='How many visualization to generate per '
                                                                              'command template.')
    parser.add_argument('--train_percentage', type=float, default=.8,
                        help='Percentage of examples to put in the training set (rest is test set).')

    # World arguments.
    parser.add_argument('--grid_size', type=int, default=6, help='Number of rows (and columns) in the grid world.')
    parser.add_argument('--min_other_objects', type=int, default=0, help='Minimum amount of objects to put in the grid '
                                                                         'world.')
    parser.add_argument('--max_objects', type=int, default=2, help='Maximum amount of objects to put in the grid '
                                                                   'world.')
    parser.add_argument('--sample_vocab', dest='sample_vocab', default=False, action='store_true')  # TODO
    parser.add_argument('--min_object_size', type=int, default=1, help='Smallest object size.')
    parser.add_argument('--max_object_size', type=int, default=4, help='Biggest object size.')
    parser.add_argument('--other_objects_sample_percentage', type=float, default=.5,
                        help='Percentage of possible objects distinct from the target to place in the world.')

    # Grammar and Vocabulary arguments
    parser.add_argument('--type_grammar', type=str, default='normal', choices=['simple_intrans', 'simple_trans',
                                                                               'normal', 'adverb', 'full'])
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
    grounded_scan = GroundedScan(
        intransitive_verbs=flags["intransitive_verbs"].split(','),
        transitive_verbs=flags["transitive_verbs"].split(','),
        adverbs=flags["adverbs"].split(','), nouns=flags["nouns"].split(','),
        color_adjectives=flags["color_adjectives"].split(',') if flags["color_adjectives"] else [],
        size_adjectives=flags["size_adjectives"].split(',') if flags["size_adjectives"] else [],
        min_object_size=flags["min_object_size"], max_object_size=flags["max_object_size"],
        save_directory=flags["output_directory"], grid_size=flags["grid_size"], type_grammar=flags["type_grammar"])

    # Create directory for visualizations if it doesn't exist.
    if flags['output_directory']:
        visualization_path = os.path.join(os.getcwd(), flags['output_directory'])
        if not os.path.exists(visualization_path):
            os.mkdir(visualization_path)

    if flags['mode'] == 'generate':

        # Generate all possible commands from the grammar
        grounded_scan.get_data_pairs(num_resampling=flags['num_resampling'],
                                     other_objects_sample_percentage=flags['other_objects_sample_percentage'],
                                     visualize_per_template=flags['visualize_per_template'],
                                     split_type=flags["split"],
                                     train_percentage=flags['train_percentage'],
                                     min_other_objects=flags['min_other_objects'])
        logger.info("Discarding equivalent examples, may take a while...")
        equivalent_examples = grounded_scan.discard_equivalent_examples()
        logger.info("Gathering dataset statistics...")
        grounded_scan.save_dataset_statistics(split="train")
        if flags["split"] == "uniform":
            grounded_scan.save_dataset_statistics(split="test")
        elif flags["split"] == "generalization":
            for split in ["visual", "situational_1", "situational_2", "contextual"]:
                grounded_scan.save_dataset_statistics(split=split)
        dataset_path = grounded_scan.save_dataset(flags['save_dataset_as'])
        grounded_scan.visualize_data_examples()
        logger.info("Saved dataset to {}".format(dataset_path))
        logger.info("Discarded {} examples from the test set that were already in the training set.".format(
            equivalent_examples))
        if flags['count_equivalent_examples']:
            if flags["split"] == "uniform":
                splits_to_count = ["test"]
            elif flags["split"] == "generalization":
                splits_to_count = ["visual", "situational_1", "situational_2", "contextual"]
            else:
                raise ValueError("Unknown option for flag --split: {}".format(flags["split"]))
            for split in splits_to_count:
                logger.info("Equivalent examples in train and testset: {}".format(
                    grounded_scan.count_equivalent_examples("train", split)))
    elif flags['mode'] == 'execute_commands':
        assert os.path.exists(flags["predicted_commands_file"]), "Trying to execute commands from non-existing file: "\
                                                                 "{}".format(flags["predicted_commands_file"])
        grounded_scan.visualize_prediction(flags["predicted_commands_file"], only_save_errors=flags["only_save_errors"])
    elif flags['mode'] == 'test':
        logger.info("Running all tests..")
        run_all_tests()
    elif flags['mode'] == 'error_analysis':
        logger.info("Performing error analysis on file with predictions: {}".format(flags["predicted_commands_file"]))
        grounded_scan.error_analysis(predictions_file=flags["predicted_commands_file"],
                                     output_file=os.path.join(flags["output_directory"], "error_analysis.txt"))
        logger.info("Wrote data to path: {}.".format(os.path.join(flags["output_directory"], "error_analysis.txt")))
        logger.info("Saved plots in directory: {}.".format(flags["output_directory"]))
    else:
        raise ValueError("Unknown value for command-line argument 'mode'={}.".format(flags['mode']))


if __name__ == "__main__":
    main()
