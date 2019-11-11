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
    TEST_DATASET.get_data_pairs(max_examples=10000)
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


def test_demonstrate_target_commands_one():
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_intrans 'to' DP,ROOT -> VP"
    lexicon_str = "T:walk,NT:VV_intransitive -> walk,T:to,T:a,T:big,NT:JJ -> big,T:circle,NT:NN -> circle"
    derivation = Derivation.from_str(rules_str, lexicon_str, TEST_DATASET._grammar)
    actual_target_commands, _, _ = TEST_DATASET.demonstrate_command(derivation, TEST_SITUATION_1)
    target_commands, _ = TEST_DATASET.demonstrate_target_commands(derivation, TEST_SITUATION_1, actual_target_commands)
    assert ','.join(actual_target_commands) == ','.join(target_commands),  \
        "test_demonstrate_target_commands_one FAILED"
    end = time.time()
    print("test_demonstrate_target_commands_one PASSED in {} seconds".format(end - start))


def test_demonstrate_target_commands_two():
    # TODO: test this function.
    raise NotImplementedError()


def test_demonstrate_command_one():
    """Test pushing a light object (where one target command of 'push <dir>' results in movement of 1 grid)."""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_trans DP,ROOT -> VP"
    lexicon_str = "T:push,NT:VV_transitive -> push,T:a,T:small,NT:JJ -> small,T:circle,NT:NN -> circle"
    derivation = Derivation.from_str(rules_str, lexicon_str, TEST_DATASET._grammar)
    expected_target_commands = "walk east,walk east,walk south,walk south,walk south,"\
                               "push south,push south,push south,push south"
    actual_target_commands, _, _ = TEST_DATASET.demonstrate_command(derivation, initial_situation=TEST_SITUATION_1)
    assert expected_target_commands == ','.join(actual_target_commands), "test_demonstrate_command_one FAILED"
    end = time.time()
    print("test_demonstrate_command_one PASSED in {} seconds".format(end - start))


def test_demonstrate_command_two():
    """Test pushing a heavy object (where one target command of 'push <dir>' results in movement of 1 grid)."""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_trans DP,ROOT -> VP"
    lexicon_str = "T:push,NT:VV_transitive -> push,T:a,T:small,NT:JJ -> small,T:circle,NT:NN -> circle"
    derivation = Derivation.from_str(rules_str, lexicon_str, TEST_DATASET._grammar)
    expected_target_commands = "walk east,walk east,walk south,walk south,walk south," \
                               "push south,push south,push south,push south,push south,push south,push south,push south"
    actual_target_commands, _, _ = TEST_DATASET.demonstrate_command(derivation, initial_situation=TEST_SITUATION_2)
    assert expected_target_commands == ','.join(actual_target_commands), "test_demonstrate_command_two FAILED"
    end = time.time()
    print("test_demonstrate_command_two PASSED in {} seconds".format(end - start))


def test_demonstrate_command_three():
    """Test walk to a small circle, tests that the function demonstrate command is able to find the target small circle
    even if that circle isn't explicitly set as the target object in the situation (which it wouldn't be at test time).
    """
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_intrans 'to' DP,ROOT -> VP"
    lexicon_str = "T:walk,NT:VV_intransitive -> walk,T:to,T:a,T:small,NT:JJ -> small,T:circle,NT:NN -> circle"
    derivation = Derivation.from_str(rules_str, lexicon_str, TEST_DATASET._grammar)
    expected_target_commands = "walk east,walk east,walk south,walk south,walk south"
    actual_target_commands, _, _ = TEST_DATASET.demonstrate_command(derivation, initial_situation=TEST_SITUATION_3)
    assert expected_target_commands == ','.join(actual_target_commands), "test_demonstrate_command_three FAILED"
    end = time.time()
    print("test_demonstrate_command_three PASSED in {} seconds".format(end - start))


def test_demonstrate_command_four():
    """Test walk to a small circle, tests that the function demonstrate command is able to find the target big circle
    even if that circle isn't explicitly set as the target object in the situation (which it wouldn't be at test time).
    """
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,DP -> 'a' NP,VP -> VV_intrans 'to' DP,ROOT -> VP"
    lexicon_str = "T:walk,NT:VV_intransitive -> walk,T:to,T:a,T:big,NT:JJ -> big,T:circle,NT:NN -> circle"
    derivation = Derivation.from_str(rules_str, lexicon_str, TEST_DATASET._grammar)
    expected_target_commands = "walk west,walk north,walk north,walk north,walk north"
    actual_target_commands, _, _ = TEST_DATASET.demonstrate_command(derivation, initial_situation=TEST_SITUATION_3)
    assert expected_target_commands == ','.join(actual_target_commands), "test_demonstrate_command_four FAILED"
    end = time.time()
    print("test_demonstrate_command_four PASSED in {} seconds".format(end - start))


def test_demonstrate_command_five():
    """Test that when referring to a small red circle and two present in the world, it finds the correct one."""
    start = time.time()
    rules_str = "NP -> NN,NP -> JJ NP,NP -> JJ NP,DP -> 'a' NP,VP -> VV_intrans 'to' DP,ROOT -> VP"
    lexicon_str = "T:walk,NT:VV_intransitive -> walk,T:to,T:a,T:red,NT:JJ -> small:JJ -> red,T:small,T:circle,NT:"\
                  "NN -> circle"
    derivation = Derivation.from_str(rules_str, lexicon_str, TEST_DATASET._grammar)
    expected_target_commands = "walk east,walk east,walk south,walk south,walk south"
    actual_target_commands, _, _ = TEST_DATASET.demonstrate_command(derivation, initial_situation=TEST_SITUATION_4)
    assert expected_target_commands == ','.join(actual_target_commands), "test_demonstrate_command_five FAILED"
    end = time.time()
    print("test_demonstrate_command_five PASSED in {} seconds".format(end - start))


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
    print("test_demonstrate_command_six PASSED in {} seconds".format(end - start))


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
    print("test_find_referred_target_one PASSED in {} seconds".format(end - start))


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
    print("test_find_referred_target_two PASSED in {} seconds".format(end - start))


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
    print("test_generate_possible_targets_one PASSED in {} seconds".format(end - start))


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
    print("test_generate_possible_targets_two PASSED in {} seconds".format(end - start))


def test_generate_situations():
    """Test that for particular commands the right situations get matched."""
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
    object_positions = TEST_DATASET._world.object_positions("green circle",
                                                            object_size="small")
    assert object_positions.pop() == relevant_situation["target_position"], "test_generate_situations FAILED."
    # TODO: test for no smaller green circle
    # TODO: test at least one larger green circle
    end = time.time()
    print("test_generate_situations PASSED in {} seconds".format(end - start))
    raise NotImplementedError()


if __name__ == "__main__":
    test_save_and_load_dataset()
    test_derivation_from_rules()
    test_derivation_from_string()
    test_demonstrate_target_commands_one()
    # test_demonstrate_target_commands_two()
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
    # test_generate_situations()
    os.rmdir(TEST_DIRECTORY)

