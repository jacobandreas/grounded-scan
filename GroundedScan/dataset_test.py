# TODO: use test framework instead of asserts
from GroundedScan.dataset import GroundedScan
from GroundedScan.grammar import Derivation
from GroundedScan.world import Situation
from GroundedScan.world import Position
from GroundedScan.world import Object
from GroundedScan.world import INT_TO_DIR
from GroundedScan.world import PositionedObject
from GroundedScan.helpers import numpy_array_to_image
from GroundedScan.helpers import image_to_numpy_array

import os
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)

TEST_DIRECTORY = "test_dir"
TEST_PATH = os.path.join(os.getcwd(), TEST_DIRECTORY)
if not os.path.exists(TEST_PATH):
    os.mkdir(TEST_PATH)

EXAMPLES_TO_TEST = 10000

TEST_DATASET = GroundedScan(intransitive_verbs=["walk"],
                            transitive_verbs=["push"],
                            adverbs=["TestAdverb1"], nouns=["circle", "cylinder", "square"],
                            color_adjectives=["red", "blue", "green"],
                            size_adjectives=["big", "small"],
                            min_object_size=1, max_object_size=4,
                            save_directory=TEST_DIRECTORY, grid_size=15, type_grammar="normal")

TEST_SITUATION_1 = Situation(grid_size=15, agent_position=Position(row=7, column=2), agent_direction=INT_TO_DIR[0],
                             target_object=PositionedObject(object=Object(size=2, color='red', shape='circle'),
                                                            position=Position(row=10, column=4),
                                                            vector=np.array([1, 0, 1])),
                             placed_objects=[PositionedObject(object=Object(size=2, color='red', shape='circle'),
                                                              position=Position(row=10, column=4),
                                                              vector=np.array([1, 0, 1])),
                                             PositionedObject(object=Object(size=4, color='green', shape='circle'),
                                                              position=Position(row=3, column=12),
                                                              vector=np.array([0, 1, 0]))], carrying=None)

TEST_SITUATION_2 = Situation(grid_size=15, agent_position=Position(row=7, column=2), agent_direction=INT_TO_DIR[0],
                             target_object=PositionedObject(object=Object(size=4, color='red', shape='circle'),
                                                            position=Position(row=10, column=4),
                                                            vector=np.array([1, 0, 1])),
                             placed_objects=[PositionedObject(object=Object(size=4, color='red', shape='circle'),
                                                              position=Position(row=10, column=4),
                                                              vector=np.array([1, 0, 1])),
                                             PositionedObject(object=Object(size=4, color='green', shape='cylinder'),
                                                              position=Position(row=3, column=12),
                                                              vector=np.array([0, 1, 0]))], carrying=None)

TEST_SITUATION_3 = Situation(grid_size=15, agent_position=Position(row=7, column=2), agent_direction=INT_TO_DIR[0],
                             target_object=None,
                             placed_objects=[PositionedObject(object=Object(size=1, color='red', shape='circle'),
                                                              position=Position(row=10, column=4),
                                                              vector=np.array([1, 0, 1])),
                                             PositionedObject(object=Object(size=2, color='green', shape='circle'),
                                                              position=Position(row=3, column=1),
                                                              vector=np.array([0, 1, 0]))], carrying=None)

TEST_SITUATION_4 = Situation(grid_size=15, agent_position=Position(row=7, column=2), agent_direction=INT_TO_DIR[0],
                             target_object=None,
                             placed_objects=[PositionedObject(object=Object(size=2, color='red', shape='circle'),
                                                              position=Position(row=10, column=4),
                                                              vector=np.array([1, 0, 1])),
                                             PositionedObject(object=Object(size=4, color='red', shape='circle'),
                                                              position=Position(row=3, column=1),
                                                              vector=np.array([0, 1, 0]))], carrying=None)


def test_save_and_load_dataset():
    start = time.time()
    TEST_DATASET.get_data_pairs(max_examples=EXAMPLES_TO_TEST)
    TEST_DATASET.save_dataset("test.txt")
    TEST_DATASET.save_dataset_statistics(split="train")
    TEST_DATASET.save_dataset_statistics(split="test")

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
    for example_one, example_two in zip(TEST_DATASET.get_examples_with_image("train"),
                                        test_grounded_scan.get_examples_with_image("train")):
        assert TEST_DATASET.command_repr(example_one["input_command"]) == test_grounded_scan.command_repr(
            example_two["input_command"]), "test_save_and_load_dataset FAILED"
        assert TEST_DATASET.command_repr(example_one["target_command"]) == test_grounded_scan.command_repr(
            example_two["target_command"]), "test_save_and_load_dataset FAILED"
        assert np.array_equal(example_one["situation_image"], example_two["situation_image"]),\
            "test_save_and_load_dataset FAILED"
    os.remove(os.path.join(TEST_DIRECTORY, "test.txt"))
    end = time.time()
    logger.info("test_save_and_load_dataset PASSED in {} seconds".format(end - start))
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
    logger.info("test_derivation_from_rules PASSED in {} seconds".format(end - start))


def test_derivation_from_string():
    start = time.time()
    derivation, arguments = TEST_DATASET.sample_command()
    derivation_str = derivation.__repr__()
    rules_str, lexicon_str = derivation_str.split(';')
    new_derivation = Derivation.from_str(rules_str, lexicon_str, TEST_DATASET._grammar)
    assert ' '.join(new_derivation.words()) == ' '.join(derivation.words()), "test_derivation_from_string FAILED"
    end = time.time()
    logger.info("test_derivation_from_string PASSED in {} seconds".format(end - start))


def test_demonstrate_target_commands_one():
    """Test that target commands sequence resulting from demonstrate_command is the same as the one executed by
     demonstrate_target_commands"""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_intrans 'to' DP,ROOT -> VP"
    lexicon_str = "T:walk,NT:VV_intransitive -> walk,T:to,T:a,T:small,NT:JJ -> small,T:circle,NT:NN -> circle"
    derivation = Derivation.from_str(rules_str, lexicon_str, TEST_DATASET._grammar)
    actual_target_commands, _, _ = TEST_DATASET.demonstrate_command(derivation, TEST_SITUATION_1)
    command = ' '.join(derivation.words())
    target_commands, _ = TEST_DATASET.demonstrate_target_commands(command, TEST_SITUATION_1, actual_target_commands)
    assert ','.join(actual_target_commands) == ','.join(target_commands),  \
        "test_demonstrate_target_commands_one FAILED"
    end = time.time()
    logger.info("test_demonstrate_target_commands_one PASSED in {} seconds".format(end - start))


def test_demonstrate_target_commands_two():
    """Test that target commands sequence resulting from demonstrate_command for pushing a heavy objectis the same as
     the executed one by demonstrate_target_commands"""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_trans DP,ROOT -> VP"
    lexicon_str = "T:push,NT:VV_transitive -> push,T:a,T:big,NT:JJ -> big,T:circle,NT:NN -> circle"
    derivation = Derivation.from_str(rules_str, lexicon_str, TEST_DATASET._grammar)
    actual_target_commands, _, _ = TEST_DATASET.demonstrate_command(derivation, initial_situation=TEST_SITUATION_2)
    command = ' '.join(derivation.words())
    target_commands, _ = TEST_DATASET.demonstrate_target_commands(command, TEST_SITUATION_2, actual_target_commands)
    assert ','.join(actual_target_commands) == ','.join(target_commands), "test_demonstrate_target_commands_two FAILED"
    end = time.time()
    logger.info("test_demonstrate_target_commands_two PASSED in {} seconds".format(end - start))


def test_demonstrate_target_commands_three():
    """Test that target commands sequence resulting from demonstrate_command for pushing a light object is the same as
     the executed one by demonstrate_target_commands"""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_trans DP,ROOT -> VP"
    lexicon_str = "T:push,NT:VV_transitive -> push,T:a,T:small,NT:JJ -> small,T:circle,NT:NN -> circle"
    derivation = Derivation.from_str(rules_str, lexicon_str, TEST_DATASET._grammar)
    actual_target_commands, _, _ = TEST_DATASET.demonstrate_command(derivation, initial_situation=TEST_SITUATION_1)
    command = ' '.join(derivation.words())
    target_commands, _ = TEST_DATASET.demonstrate_target_commands(command, TEST_SITUATION_1, actual_target_commands)
    assert ','.join(actual_target_commands) == ','.join(target_commands), "test_demonstrate_target_commands_three FAILED"
    end = time.time()
    logger.info("test_demonstrate_target_commands_three PASSED in {} seconds".format(end - start))


def test_demonstrate_command_one():
    """Test pushing a light object (where one target command of 'push <dir>' results in movement of 1 grid)."""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_trans DP,ROOT -> VP"
    lexicon_str = "T:push,NT:VV_transitive -> push,T:a,T:small,NT:JJ -> small,T:circle,NT:NN -> circle"
    derivation = Derivation.from_str(rules_str, lexicon_str, TEST_DATASET._grammar)
    expected_target_commands = "walk,walk,turn right,walk,walk,walk,"\
                               "push,push,push,push"
    actual_target_commands, _, _ = TEST_DATASET.demonstrate_command(derivation, initial_situation=TEST_SITUATION_1)
    assert expected_target_commands == ','.join(actual_target_commands), "test_demonstrate_command_one FAILED"
    end = time.time()
    logger.info("test_demonstrate_command_one PASSED in {} seconds".format(end - start))


def test_demonstrate_command_two():
    """Test pushing a heavy object (where one target command of 'push <dir>' results in movement of 1 grid)."""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_trans DP,ROOT -> VP"
    lexicon_str = "T:push,NT:VV_transitive -> push,T:a,T:small,NT:JJ -> small,T:circle,NT:NN -> circle"
    derivation = Derivation.from_str(rules_str, lexicon_str, TEST_DATASET._grammar)
    expected_target_commands = "walk,walk,turn right,walk,walk,walk," \
                               "push,push,push,push,push,push,push,push"
    actual_target_commands, _, _ = TEST_DATASET.demonstrate_command(derivation, initial_situation=TEST_SITUATION_2)
    assert expected_target_commands == ','.join(actual_target_commands), "test_demonstrate_command_two FAILED"
    end = time.time()
    logger.info("test_demonstrate_command_two PASSED in {} seconds".format(end - start))


def test_demonstrate_command_three():
    """Test walk to a small circle, tests that the function demonstrate command is able to find the target small circle
    even if that circle isn't explicitly set as the target object in the situation (which it wouldn't be at test time).
    """
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_intrans 'to' DP,ROOT -> VP"
    lexicon_str = "T:walk,NT:VV_intransitive -> walk,T:to,T:a,T:small,NT:JJ -> small,T:circle,NT:NN -> circle"
    derivation = Derivation.from_str(rules_str, lexicon_str, TEST_DATASET._grammar)
    expected_target_commands = "walk,walk,turn right,walk,walk,walk"
    actual_target_commands, _, _ = TEST_DATASET.demonstrate_command(derivation, initial_situation=TEST_SITUATION_3)
    assert expected_target_commands == ','.join(actual_target_commands), "test_demonstrate_command_three FAILED"
    end = time.time()
    logger.info("test_demonstrate_command_three PASSED in {} seconds".format(end - start))


def test_demonstrate_command_four():
    """Test walk to a small circle, tests that the function demonstrate command is able to find the target big circle
    even if that circle isn't explicitly set as the target object in the situation (which it wouldn't be at test time).
    """
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_intrans 'to' DP,ROOT -> VP"
    lexicon_str = "T:walk,NT:VV_intransitive -> walk,T:to,T:a,T:big,NT:JJ -> big,T:circle,NT:NN -> circle"
    derivation = Derivation.from_str(rules_str, lexicon_str, TEST_DATASET._grammar)
    expected_target_commands = "turn left,turn left,walk,turn right,walk,walk,walk,walk"
    actual_target_commands, _, _ = TEST_DATASET.demonstrate_command(derivation, initial_situation=TEST_SITUATION_3)
    assert expected_target_commands == ','.join(actual_target_commands), "test_demonstrate_command_four FAILED"
    end = time.time()
    logger.info("test_demonstrate_command_four PASSED in {} seconds".format(end - start))


def test_demonstrate_command_five():
    """Test that when referring to a small red circle and two present in the world, it finds the correct one."""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,NP -> JJ NP,DP -> 'a' NP,VP -> VV_intrans 'to' DP,ROOT -> VP"
    lexicon_str = "T:walk,NT:VV_intransitive -> walk,T:to,T:a,T:red,NT:JJ -> small:JJ -> red,T:small,T:circle,NT:"\
                  "NN -> circle"
    derivation = Derivation.from_str(rules_str, lexicon_str, TEST_DATASET._grammar)
    expected_target_commands = "walk,walk,turn right,walk,walk,walk"
    actual_target_commands, _, _ = TEST_DATASET.demonstrate_command(derivation, initial_situation=TEST_SITUATION_4)
    assert expected_target_commands == ','.join(actual_target_commands), "test_demonstrate_command_five FAILED"
    end = time.time()
    logger.info("test_demonstrate_command_five PASSED in {} seconds".format(end - start))


def test_demonstrate_command_six():
    """Test that when referring to a small red circle but only one red circle is present, demonstrate_commands fails."""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,NP -> JJ NP,DP -> 'a' NP,VP -> VV_intrans 'to' DP,ROOT -> VP"
    lexicon_str = "T:walk,NT:VV_intransitive -> walk,T:to,T:a,T:red,NT:JJ -> small:JJ -> red,T:small,T:circle,NT:"\
                  "NN -> circle"
    derivation = Derivation.from_str(rules_str, lexicon_str, TEST_DATASET._grammar)
    expected_target_commands = ""
    try:
        actual_target_commands, _, _ = TEST_DATASET.demonstrate_command(derivation, initial_situation=TEST_SITUATION_3)
    except AssertionError:
        actual_target_commands = ""
    assert expected_target_commands == ','.join(actual_target_commands), "test_demonstrate_command_six FAILED"
    end = time.time()
    logger.info("test_demonstrate_command_six PASSED in {} seconds".format(end - start))


def test_find_referred_target_one():
    """Test that for particular referred targets, the Derivation class identifies it correctly."""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,NP -> JJ NP,DP -> 'a' NP,VP -> VV_intrans 'to' DP,ROOT -> VP"
    lexicon_str = "T:walk,NT:VV_intransitive -> walk,T:to,T:a,T:red,NT:JJ -> small:JJ -> red,T:small,T:circle,NT:" \
                  "NN -> circle"
    derivation = Derivation.from_str(rules_str, lexicon_str, TEST_DATASET._grammar)
    arguments = []
    derivation.meaning(arguments)
    assert len(arguments) == 1, "test_find_referred_target_one FAILED."
    target_str, target_predicate = arguments.pop().to_predicate()
    assert target_str == "red circle", "test_find_referred_target FAILED."
    assert target_predicate["noun"] == "circle", "test_find_referred_target_one FAILED."
    assert target_predicate["size"] == "small", "test_find_referred_target_one FAILED."
    assert target_predicate["color"] == "red", "test_find_referred_target_one FAILED."
    end = time.time()
    logger.info("test_find_referred_target_one PASSED in {} seconds".format(end - start))


def test_find_referred_target_two():
    """Test that for particular referred targets, the Derivation class identifies it correctly."""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_intrans 'to' DP,ROOT -> VP"
    lexicon_str = "T:walk,NT:VV_intransitive -> walk,T:to,T:a,T:big,NT:JJ -> big,T:circle,NT:NN -> circle"
    derivation = Derivation.from_str(rules_str, lexicon_str, TEST_DATASET._grammar)
    arguments = []
    derivation.meaning(arguments)
    assert len(arguments) == 1, "test_find_referred_target_two FAILED."
    target_str, target_predicate = arguments.pop().to_predicate()
    assert target_str == "circle", "test_find_referred_target_two FAILED."
    assert target_predicate["noun"] == "circle", "test_find_referred_target_two FAILED."
    assert target_predicate["size"] == "big", "test_find_referred_target_two FAILED."
    assert target_predicate["color"] == "", "test_find_referred_target_two FAILED."
    end = time.time()
    logger.info("test_find_referred_target_two PASSED in {} seconds".format(end - start))


def test_generate_possible_targets_one():
    """Test that for particular referred targets, the right possible target objects get generated."""
    start = time.time()
    target_predicate = {"noun": "circle", "color": "red", "size": "big"}
    expected_possible_targets = {(2, "red", "circle"), (3, "red", "circle"), (4, "red", "circle")}
    actual_possible_targets = TEST_DATASET.generate_possible_targets(referred_size=target_predicate["size"],
                                                                     referred_color=target_predicate["color"],
                                                                     referred_shape=target_predicate["noun"])
    for actual_possible_target in actual_possible_targets:
        assert actual_possible_target in expected_possible_targets, "test_generate_possible_targets_one FAILED."
    end = time.time()
    logger.info("test_generate_possible_targets_one PASSED in {} seconds".format(end - start))


def test_generate_possible_targets_two():
    """Test that for particular referred targets, the right possible target objects get generated."""
    start = time.time()
    target_predicate = {"noun": "circle", "color": "", "size": "small"}
    expected_possible_targets = {(1, "red", "circle"), (2, "red", "circle"), (3, "red", "circle"),
                                 (1, "blue", "circle"), (2, "blue", "circle"), (3, "blue", "circle"),
                                 (1, "green", "circle"), (2, "green", "circle"), (3, "green", "circle")}
    actual_possible_targets = TEST_DATASET.generate_possible_targets(referred_size=target_predicate["size"],
                                                                     referred_color=target_predicate["color"],
                                                                     referred_shape=target_predicate["noun"])
    for expected_possible_target, actual_possible_target in zip(expected_possible_targets, actual_possible_targets):
        assert actual_possible_target in expected_possible_targets, "test_generate_possible_targets_two FAILED."
    end = time.time()
    logger.info("test_generate_possible_targets_two PASSED in {} seconds".format(end - start))


def test_generate_situations_one():
    """Test that when a small green circle is referred to there exist no smaller green circles than the target object in
    the world and at least one larger green circle."""
    start = time.time()
    target_shape = "circle"
    target_color = "green"
    target_size = 2
    referred_size = "small"
    referred_color = "green"
    referred_shape = "circle"
    situation_specifications = TEST_DATASET.generate_situations(num_resampling=1)
    relevant_situation = situation_specifications[target_shape][target_color][target_size].pop()
    TEST_DATASET.initialize_world_from_spec(relevant_situation, referred_size=referred_size,
                                            referred_color=referred_color,
                                            referred_shape=referred_shape,
                                            actual_size=target_size,
                                            sample_percentage=0.5
                                            )
    smallest_object = TEST_DATASET._world.object_positions("green circle",
                                                           object_size="small").pop()
    assert smallest_object == relevant_situation["target_position"], "test_generate_situations_one FAILED."
    other_related_objects = TEST_DATASET._world.object_positions("green circle")
    larger_objects = []
    for size, sized_objects in other_related_objects:
        if size < target_size:
            assert not sized_objects, "test_generate_situations_one FAILED."
        elif size > target_size:
            larger_objects.extend(sized_objects)
    assert len(larger_objects) >= 1, "test_generate_situations_one FAILED."
    end = time.time()
    logger.info("test_generate_situations_one PASSED in {} seconds".format(end - start))


def test_generate_situations_two():
    """Test that when a big green circle is referred to there exists no larger green circles and the exists at least
    one smaller green circle."""
    start = time.time()
    target_shape = "circle"
    target_color = "green"
    target_size = 2
    referred_size = "big"
    referred_color = "green"
    referred_shape = "circle"
    situation_specifications = TEST_DATASET.generate_situations(num_resampling=1)
    relevant_situation = situation_specifications[target_shape][target_color][target_size].pop()
    TEST_DATASET.initialize_world_from_spec(relevant_situation, referred_size=referred_size,
                                            referred_color=referred_color,
                                            referred_shape=referred_shape,
                                            actual_size=target_size,
                                            sample_percentage=0.5
                                            )
    largest_object = TEST_DATASET._world.object_positions("green circle",
                                                           object_size="big").pop()
    assert largest_object == relevant_situation["target_position"], "test_generate_situations_two FAILED."
    other_related_objects = TEST_DATASET._world.object_positions("green circle")
    smaller_objects = []
    for size, sized_objects in other_related_objects:
        if size > target_size:
            assert not sized_objects, "test_generate_situations_two FAILED."
        elif size < target_size:
            smaller_objects.extend(sized_objects)
    assert len(smaller_objects) >= 1, "test_generate_situations_two FAILED."
    end = time.time()
    logger.info("test_generate_situations_two PASSED in {} seconds".format(end - start))


def test_generate_situations_three():
    """Test that for particular commands the right situations get matched."""
    start = time.time()
    target_shape = "circle"
    target_color = "green"
    target_size = 2
    referred_size = "big"
    referred_shape = "circle"
    situation_specifications = TEST_DATASET.generate_situations(num_resampling=1)
    relevant_situation = situation_specifications[target_shape][target_color][target_size].pop()
    TEST_DATASET.initialize_world_from_spec(relevant_situation, referred_size=referred_size,
                                            referred_color="",
                                            referred_shape=referred_shape,
                                            actual_size=target_size,
                                            sample_percentage=0.5
                                            )
    largest_object = TEST_DATASET._world.object_positions("circle",
                                                          object_size="big").pop()
    assert largest_object == relevant_situation["target_position"], "test_generate_situations_three FAILED."
    other_related_objects = TEST_DATASET._world.object_positions("circle")
    smaller_objects = []
    for size, sized_objects in other_related_objects:
        if size > target_size:
            assert not sized_objects, "test_generate_situations_three FAILED."
        elif size < target_size:
            smaller_objects.extend(sized_objects)
    assert len(smaller_objects) >= 1, "test_generate_situations_three FAILED."
    end = time.time()
    logger.info("test_generate_situations_three PASSED in {} seconds".format(end - start))


def test_situation_representation_eq():
    start = time.time()
    test_situations = [TEST_SITUATION_1, TEST_SITUATION_2, TEST_SITUATION_3, TEST_SITUATION_4]
    for i, test_situation_1 in enumerate(test_situations):
        for j, test_situation_2 in enumerate(test_situations):
            if i == j:
                assert test_situation_1 == test_situation_2, "test_situation_representation_eq FAILED."
            else:
                assert test_situation_1 != test_situation_2, "test_situation_representation_eq FAILED."
    end = time.time()
    logger.info("test_situation_representation_eq PASSED in {} seconds".format(end - start))


def test_example_representation_eq():
    """Test that the function for comparing examples returns true when exactly the same example is passed twice."""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_intrans 'to' DP,ROOT -> VP"
    lexicon_str = "T:walk,NT:VV_intransitive -> walk,T:to,T:a,T:big,NT:JJ -> big,T:circle,NT:NN -> circle"
    derivation = Derivation.from_str(rules_str, lexicon_str, TEST_DATASET._grammar)
    arguments = []
    derivation.meaning(arguments)
    target_str, target_predicate = arguments.pop().to_predicate()

    target_commands, _, target_action = TEST_DATASET.demonstrate_command(derivation, initial_situation=TEST_SITUATION_1)
    TEST_DATASET.fill_example(derivation.words(), derivation, TEST_SITUATION_1, target_commands, target_action,
                              target_predicate, visualize=False)
    TEST_DATASET.get_data_pairs(max_examples=10, num_resampling=2)
    for split, examples in TEST_DATASET._data_pairs.items():
        for example in examples:
            assert TEST_DATASET.compare_examples(example, example), "test_example_representation_eq FAILED."
    end = time.time()
    logger.info("test_example_representation_eq PASSED in {} seconds".format(end - start))


def test_example_representation():
    """Test that when you save an example in its representation its the same if you parse it again."""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_intrans 'to' DP,ROOT -> VP"
    lexicon_str = "T:walk,NT:VV_intransitive -> walk,T:to,T:a,T:big,NT:JJ -> big,T:circle,NT:NN -> circle"
    derivation = Derivation.from_str(rules_str, lexicon_str, TEST_DATASET._grammar)
    arguments = []
    derivation.meaning(arguments)
    target_str, target_predicate = arguments.pop().to_predicate()

    target_commands, _, target_action = TEST_DATASET.demonstrate_command(derivation, initial_situation=TEST_SITUATION_1)
    TEST_DATASET.fill_example(derivation.words(), derivation, TEST_SITUATION_1, target_commands, target_action,
                              target_predicate, visualize=False, split="train")
    example = TEST_DATASET._data_pairs["train"].pop()
    parsed_command, parsed_derivation, parsed_situation, parsed_target_commands, _, parsed_action = TEST_DATASET.parse_example(
        example
    )
    assert example["command"] == TEST_DATASET.command_repr(parsed_command), "test_example_representation FAILED."
    assert example["derivation"] == TEST_DATASET.derivation_repr(parsed_derivation), "test_example_representation "\
                                                                                     "FAILED."
    situation = Situation.from_representation(example["situation"])
    assert situation == parsed_situation, "test_example_representation FAILED."
    assert example["target_commands"] == TEST_DATASET.command_repr(parsed_target_commands), \
        "test_example_representation FAILED."
    assert example["verb_in_command"] == parsed_action, "test_example_representation FAILED."
    assert example["referred_target"] == ' '.join([target_predicate["size"], target_predicate["color"],
                                         target_predicate["noun"]]), "test_example_representation FAILED."
    end = time.time()
    logger.info("test_example_representation PASSED in {} seconds".format(end - start))


def test_initialize_world():
    """Test that two the same situations get represented in exactly the same image by rendering.py and minigrid.py"""
    start = time.time()
    test_situations = [TEST_SITUATION_1, TEST_SITUATION_2, TEST_SITUATION_3, TEST_SITUATION_4]
    current_situation = TEST_DATASET._world.get_current_situation()
    current_mission = TEST_DATASET._world.mission
    for i, test_situation_1 in enumerate(test_situations):
        for j, test_situation_2 in enumerate(test_situations):
            TEST_DATASET._world.clear_situation()
            TEST_DATASET.initialize_world(test_situation_1)
            situation_1 = TEST_DATASET._world.get_current_situation()
            TEST_DATASET._world.clear_situation()
            TEST_DATASET.initialize_world(test_situation_2)
            situation_2 = TEST_DATASET._world.get_current_situation()
            if i == j:
                assert situation_1 == situation_2, "test_initialize_world FAILED."
            else:
                assert situation_1 != situation_2, "test_initialize_world FAILED."
    TEST_DATASET.initialize_world(current_situation, mission=current_mission)
    end = time.time()
    logger.info("test_initialize_world PASSED in {} seconds".format(end - start))


def test_image_representation_situations():
    """Test that situations are still the same when they need to be in image / numpy RGB array form."""
    start = time.time()
    current_situation = TEST_DATASET._world.get_current_situation()
    current_mission = TEST_DATASET._world.mission
    test_situations = [TEST_SITUATION_1, TEST_SITUATION_2, TEST_SITUATION_3, TEST_SITUATION_4]
    for i, test_situation_1 in enumerate(test_situations):
        for j, test_situation_2 in enumerate(test_situations):
            TEST_DATASET._world.clear_situation()
            TEST_DATASET.initialize_world(test_situation_1)
            np_situation_image_1 = TEST_DATASET._world.render(mode='human').getArray()
            # test = TEST_DATASET._world.render(mode='human').getFullScreen(os.path.join(TEST_DIRECTORY, "test_full_screen.png"))
            # numpy_array_to_image(test, os.path.join(TEST_DIRECTORY, "test_full_screen.png"))
            numpy_array_to_image(np_situation_image_1, os.path.join(TEST_DIRECTORY, "test_im_1.png"))
            np_situation_image_1_reread = image_to_numpy_array(os.path.join(TEST_DIRECTORY, "test_im_1.png"))
            assert np.array_equal(np_situation_image_1,
                                  np_situation_image_1_reread), "test_image_representation_situations FAILED."
            TEST_DATASET._world.clear_situation()
            TEST_DATASET.initialize_world(test_situation_2)
            np_situation_image_2 = TEST_DATASET._world.render().getArray()
            numpy_array_to_image(np_situation_image_2, os.path.join(TEST_DIRECTORY, "test_im_2.png"))
            np_situation_image_2_reread = image_to_numpy_array(os.path.join(TEST_DIRECTORY, "test_im_2.png"))
            assert np.array_equal(np_situation_image_2,
                                  np_situation_image_2_reread), "test_image_representation_situations FAILED."
            if i == j:
                assert np.array_equal(np_situation_image_1, np_situation_image_2), \
                    "test_image_representation_situations FAILED."
            else:
                assert not np.array_equal(np_situation_image_1, np_situation_image_2), \
                    "test_image_representation_situations FAILED."
    os.remove(os.path.join(TEST_DIRECTORY, "test_im_1.png"))
    os.remove(os.path.join(TEST_DIRECTORY, "test_im_2.png"))
    TEST_DATASET.initialize_world(current_situation, mission=current_mission)
    end = time.time()
    logger.info("test_image_representation_situations PASSED in {} seconds".format(end - start))


def test_encode_situation():
    start = time.time()
    current_situation = TEST_DATASET._world.get_current_situation()
    current_mission = TEST_DATASET._world.mission
    test_situation = Situation(grid_size=15, agent_position=Position(row=7, column=2), agent_direction=INT_TO_DIR[0],
                               target_object=PositionedObject(object=Object(size=2, color='red', shape='circle'),
                                                              position=Position(row=7, column=2),
                                                              vector=np.array([1, 0, 1])),
                               placed_objects=[PositionedObject(object=Object(size=2, color='red', shape='circle'),
                                                                position=Position(row=7, column=2),
                                                                vector=np.array([1, 0, 1])),
                                               PositionedObject(object=Object(size=4, color='green', shape='circle'),
                                                                position=Position(row=3, column=12),
                                                                vector=np.array([0, 1, 0]))], carrying=None)
    TEST_DATASET._world.clear_situation()
    TEST_DATASET.initialize_world(test_situation)
    expected_numpy_array = np.zeros([15, 15, TEST_DATASET._world.grid._num_attributes_object + 1 + 4], dtype='uint8')
    expected_numpy_array[7, 2, -5] = 1
    expected_numpy_array[7, 2, -4:] = np.array([1, 0, 0, 0])
    expected_numpy_array[7, 2, :-5] = TEST_DATASET._object_vocabulary.get_object_vector(shape='circle', color='red',
                                                                                        size=2)
    expected_numpy_array[3, 12, :-5] = TEST_DATASET._object_vocabulary.get_object_vector(shape='circle', color='green',
                                                                                         size=4)
    encoded_numpy_array = TEST_DATASET._world.grid.encode(agent_row=7, agent_column=2, agent_direction=0)
    assert np.array_equal(expected_numpy_array, encoded_numpy_array), "test_encode_situation FAILED."
    TEST_DATASET.initialize_world(current_situation, mission=current_mission)
    end = time.time()
    logger.info("test_encode_situation PASSED in {} seconds".format(end - start))


def run_all_tests():
    test_save_and_load_dataset()
    test_derivation_from_rules()
    test_derivation_from_string()
    test_demonstrate_target_commands_one()
    test_demonstrate_target_commands_two()
    test_demonstrate_target_commands_three()
    test_demonstrate_command_one()
    test_demonstrate_command_two()
    test_demonstrate_command_three()
    test_demonstrate_command_four()
    test_demonstrate_command_five()
    test_demonstrate_command_six()
    test_find_referred_target_one()
    test_find_referred_target_two()
    test_generate_possible_targets_one()
    test_generate_possible_targets_two()
    test_generate_situations_one()
    test_generate_situations_two()
    test_generate_situations_three()
    test_situation_representation_eq()
    test_example_representation_eq()
    test_example_representation()
    test_initialize_world()
    test_image_representation_situations()
    test_encode_situation()
    os.rmdir(TEST_DIRECTORY)

