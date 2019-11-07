from dataset import GroundedScan
from grammar import Derivation
from world import Situation
from world import Position
from world import Object
from world import INT_TO_DIR
from world import PositionedObject

import os
import time
import numpy as np

TEST_DIRECTORY = "test_dir"
TEST_PATH = os.path.join(os.getcwd(), TEST_DIRECTORY)
if not os.path.exists(TEST_PATH):
    os.mkdir(TEST_PATH)

TEST_DATASET = GroundedScan(intransitive_verbs=["walk"],
                            transitive_verbs=["push"],
                            adverbs=["TestAdverb1"], nouns=["circle", "cylinder", "square"],
                            color_adjectives=["red", "green"],
                            size_adjectives=["big", "small"],
                            min_object_size=1, max_object_size=4,
                            save_directory=TEST_DIRECTORY, grid_size=15)

TEST_SITUATION = Situation(grid_size=15, agent_position=Position(row=0, column=0), agent_direction=INT_TO_DIR[0],
                           target_object=PositionedObject(object=Object(size=2, color='red', shape='circle'),
                                                          position=Position(row=10, column=4),
                                                          vector=np.array([1, 0, 1])),
                           placed_objects=[PositionedObject(object=Object(size=4, color='green', shape='cylinder'),
                                                            position=Position(row=3, column=12),
                                                            vector=np.array([0, 1, 0]))], carrying=None)


def test_save_and_load_dataset():
    start = time.time()
    TEST_DATASET.get_data_pairs(max_examples=1000)
    TEST_DATASET.save_dataset("test.txt")

    test_grounded_scan = GroundedScan.load_dataset_from_file(os.path.join(TEST_DIRECTORY, "test.txt"),
                                                             TEST_DIRECTORY)

    for statistics_one, statistics_two in zip(TEST_DATASET._data_statistics.items(),
                                              test_grounded_scan._data_statistics.items()):
        key_one, statistic_one = statistics_one
        key_two, statistic_two = statistics_two
        if isinstance(statistic_one, list):
            for stat_one, stat_two in zip(statistic_one, statistic_two):
                assert stat_one == stat_two, "test_save_and_load_dataset FAILED when comparing {} between "
                "saved and loaded dataset.".format(key_one)
        elif isinstance(statistic_one, dict):
            for key, values in statistic_one.items():
                assert statistic_two[key] == values, "test_save_and_load_dataset FAILED when comparing {} between "
                "saved and loaded dataset.".format(key_one)
    os.remove(os.path.join(TEST_DIRECTORY, "test.txt"))
    end = time.time()
    print("test_save_and_load_dataset PASSED in {} seconds".format(end - start))
    return


def test_derivation_from_rules():
    start = time.time()
    derivation, arguments = TEST_DATASET.sample_command()
    rules_list = []
    lexicon = {}
    derivation.to_rules(rules_list, lexicon)
    test = Derivation.from_rules(rules_list, lexicon=lexicon)
    assert ' '.join(test.words()) == ' '.join(derivation.words()), "test_derivation_from_rules FAILED"
    end = time.time()
    print("test_derivation_from_rules PASSED in {} seconds".format(end - start))


def test_derivation_from_string():
    start = time.time()
    derivation, arguments = TEST_DATASET.sample_command()
    derivation_str = derivation.__repr__()
    rules_str, lexicon_str = derivation_str.split(';')
    new_derivation = Derivation.from_str(rules_str, lexicon_str, TEST_DATASET._grammar)
    assert ' '.join(new_derivation.words()) == ' '.join(derivation.words()), "test_derivation_from_string FAILED"
    end = time.time()
    print("test_derivation_from_string PASSED in {} seconds".format(end - start))


def test_demonstrate_target_commands():
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_intrans 'to' DP,ROOT -> VP"
    lexicon_str = "T:walk,NT:VV_intransitive -> walk,T:to,T:a,T:big,NT:JJ -> big,T:circle,NT:NN -> circle"
    derivation = Derivation.from_str(rules_str, lexicon_str, TEST_DATASET._grammar)
    actual_target_commands, _, _ = TEST_DATASET.demonstrate_command(derivation, TEST_SITUATION)
    target_commands, _ = TEST_DATASET.demonstrate_target_commands(derivation, TEST_SITUATION, actual_target_commands)
    assert ','.join(actual_target_commands) == ','.join(target_commands),  \
        "test_demonstrate_target_commands FAILED"
    end = time.time()
    print("test_demonstrate_target_commands PASSED in {} seconds".format(end - start))


if __name__ == "__main__":
    test_save_and_load_dataset()
    test_derivation_from_rules()
    test_derivation_from_string()
    test_demonstrate_target_commands()
    os.rmdir(TEST_DIRECTORY)

