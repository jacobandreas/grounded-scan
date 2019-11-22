from GroundedScan.world import Situation
from GroundedScan.world import EVENT
from GroundedScan.world import Object
from GroundedScan.world import Position
from GroundedScan.world import World
from GroundedScan.world import ObjectVocabulary
from GroundedScan.grammar import Grammar
from GroundedScan.grammar import Derivation
from GroundedScan.vocabulary import Vocabulary
from GroundedScan.helpers import topo_sort
from GroundedScan.helpers import save_counter
from GroundedScan.helpers import bar_plot

import json
from collections import Counter
from collections import defaultdict
from typing import List
from typing import Tuple
import numpy as np
import os
import imageio
import random
import itertools


class GroundedScan(object):
    """

    """

    def __init__(self, intransitive_verbs: List[str], transitive_verbs: List[str], adverbs: List[str], nouns: List[str],
                 color_adjectives: List[str], size_adjectives: List[str], grid_size: int, min_object_size: int,
                 max_object_size: int, save_directory=os.getcwd(), max_recursion=1):

        # Command vocabulary
        self._vocabulary = Vocabulary(verbs_intrans=intransitive_verbs, verbs_trans=transitive_verbs, adverbs=adverbs,
                                      nouns=nouns, color_adjectives=color_adjectives, size_adjectives=size_adjectives)
        self.max_recursion = max_recursion
        self.save_directory = save_directory

        # Object vocabulary
        self._object_vocabulary = ObjectVocabulary(shapes=nouns, colors=color_adjectives,
                                                   min_size=min_object_size, max_size=max_object_size)

        # Initialize the world
        self._world = World(grid_size=grid_size, colors=self._vocabulary.color_adjectives,
                            object_vocabulary=self._object_vocabulary,
                            shapes=self._vocabulary.nouns,
                            save_directory=self.save_directory)
        self._relative_directions = {"n", "e", "s", "w", "ne", "se", "sw", "nw"}
        self._straight_directions = {"n", "e", "s", "w"}
        self._combined_directions = {"ne", "se", "se", "nw"}

        # Generate the grammar
        self._grammar = Grammar(self._vocabulary, max_recursion=max_recursion)

        # Data set pairs and statistics.
        self._data_pairs = {"train": [], "test": []}
        self._template_identifiers = {"train": [], "test": []}
        self._examples_to_visualize = []
        self._data_statistics = {"train": self.get_empty_data_statistics(), "test": self.get_empty_data_statistics()}

    def get_examples_with_image(self, split="train", simple_situation_representation=False) -> dict:
        """Get data pairs with images in the form of np.ndarray's with RGB values"""
        for example in self._data_pairs[split]:
            command = self.parse_command_repr(example["command"])
            situation = Situation.from_representation(example["situation"])
            self._world.clear_situation()
            self.initialize_world(situation)
            if simple_situation_representation:
                situation_image = self._world.get_current_situation_grid_repr()
            else:
                situation_image = self._world.get_current_situation_image()
            target_commands = self.parse_command_repr(example["target_commands"])
            yield {"input_command": command, "situation_image": situation_image,
                   "situation_representation": example["situation"], "target_command": target_commands}

    @property
    def situation_image_dimension(self):
        return self._world.get_current_situation_image().shape[0]

    def num_examples(self, split="train"):
        return len(self._data_pairs[split])

    def fill_example(self, command: List[str], derivation: Derivation, situation: Situation, target_commands: List[str],
                     verb_in_command: str, target_predicate: dict, visualize: bool, split="train"):
        example = {
            "command": self.command_repr(command),
            "derivation": self.derivation_repr(derivation),
            "situation": situation.to_representation(),
            "target_commands": self.command_repr(target_commands),
            "verb_in_command": verb_in_command,
            "referred_target": ' '.join([target_predicate["size"], target_predicate["color"],
                                         target_predicate["noun"]])
        }
        self._data_pairs[split].append(example)
        self.update_data_statistics(example, split)
        if visualize:
            self._examples_to_visualize.append(example)
        return example

    @staticmethod
    def compare_examples(example_1: dict, example_2: dict) -> bool:
        """An example is regarded the same if the command, situation, target commands are the same."""
        if example_1["command"] != example_2["command"]:
            return False
        if example_1["situation"] != example_2["situation"]:
            return False
        if example_1["target_commands"] != example_2["target_commands"]:
            return False
        return True

    def parse_example(self, data_example: dict):
        command = self.parse_command_repr(data_example["command"])
        situation = Situation.from_representation(data_example["situation"])
        target_commands = self.parse_command_repr(data_example["target_commands"])
        derivation = self.parse_derivation_repr(data_example["derivation"])
        assert self.derivation_repr(derivation) == data_example["derivation"]
        actual_target_commands, target_demonstration, action = self.demonstrate_command(derivation, situation)
        assert self.command_repr(actual_target_commands) == self.command_repr(target_commands)
        return command, derivation, situation, actual_target_commands, target_demonstration, action

    @staticmethod
    def get_empty_situation():
        return {
            "distance_to_target": None,
            "direction_to_target": None,
            "target_shape": None,
            "target_color": None,
            "target_size": None,
            "target_position": None,
            "agent_position": None
        }

    def get_empty_data_statistics(self):
        empty_dict = {
            "distance_to_target": Counter(),
            "direction_to_target": Counter(),
            "target_shape": Counter(),
            "target_color": Counter(),
            "target_size": Counter(),
            "target_position": Counter(),
            "agent_position": Counter(),
            "verbs_in_command": defaultdict(int),
            "referred_targets": defaultdict(lambda: defaultdict(int)),
            "placed_targets": defaultdict(int),
            "situations": {
                "shape": {"objects_in_world": defaultdict(int), "num_objects_placed": Counter()},
                "color,shape": {"objects_in_world": defaultdict(int), "num_objects_placed": Counter()},
                "size,shape": {"objects_in_world": defaultdict(int), "num_objects_placed": Counter()},
                "size,color,shape": {"objects_in_world": defaultdict(int), "num_objects_placed": Counter()},
                "all": {"objects_in_world": defaultdict(int), "num_objects_placed": Counter()},
            }
        }
        for target_object in self._object_vocabulary.all_objects:
            target_object_str = ' '.join([str(target_object[0]), target_object[1], target_object[2]])
            for key in empty_dict["situations"].keys():
                empty_dict["situations"][key][target_object_str] = 0
            empty_dict["placed_targets"][target_object_str] = 0
        return empty_dict

    @staticmethod
    def get_example_specification():
        """Represents how one data example can be fully specified for saving to a file and loading from a file."""
        return ["command", "command_rules", "command_lexicon", "target_commands", "agent_position",
                "target_position", "distance_to_target", "direction_to_target", "target_shape",
                "target_color", "target_size", "situation"]

    def update_data_statistics(self, data_example, split="train"):
        """Keeps track of certain statistics regarding the data pairs generated."""

        # Update the statistics regarding the situation.
        self._data_statistics[split]["distance_to_target"][int(data_example["situation"]["distance_to_target"])] += 1
        self._data_statistics[split]["direction_to_target"][data_example["situation"]["direction_to_target"]] += 1
        target_size = data_example["situation"]["target_object"]["object"]["size"]
        target_color = data_example["situation"]["target_object"]["object"]["color"]
        target_shape = data_example["situation"]["target_object"]["object"]["shape"]
        self._data_statistics[split]["target_shape"][target_shape] += 1
        self._data_statistics[split]["target_color"][target_color] += 1
        self._data_statistics[split]["target_size"][target_size] += 1
        self._data_statistics[split]["target_position"][
            (data_example["situation"]["target_object"]["position"]["column"],
             data_example["situation"]["target_object"]["position"]["row"])] += 1
        self._data_statistics[split]["agent_position"][(data_example["situation"]["agent_position"]["column"],
                                                 data_example["situation"]["agent_position"]["row"])] += 1
        placed_target = ' '.join([str(target_size), target_color, target_shape])
        self._data_statistics[split]["placed_targets"][placed_target] += 1

        # Update the statistics regarding the command.
        self._data_statistics[split]["verbs_in_command"][data_example["verb_in_command"]] += 1
        self._data_statistics[split]["referred_targets"][data_example["referred_target"]][placed_target] += 1

        referred_target = data_example["referred_target"].split()
        if len(referred_target) == 3:
            referred_categories = "size,color,shape"
        elif len(referred_target) == 1:
            referred_categories = "shape"
        else:
            if referred_target[0] in self._object_vocabulary.object_colors:
                referred_categories = "color,shape"
            else:
                referred_categories = "size,shape"
        num_placed_objects = len(data_example['situation']['placed_objects'].keys())
        self._data_statistics[split]["situations"][referred_categories]["num_objects_placed"][num_placed_objects] += 1
        self._data_statistics[split]["situations"]["all"]["num_objects_placed"][num_placed_objects] += 1
        for placed_object in data_example['situation']['placed_objects'].values():
            placed_object = ' '.join([placed_object['object']['size'], placed_object['object']['color'],
                                      placed_object['object']['shape']])
            self._data_statistics[split]["situations"][referred_categories]["objects_in_world"][placed_object] += 1
            self._data_statistics[split]["situations"]["all"]["objects_in_world"][placed_object] += 1

    def save_position_counts(self, position_counts, file) -> {}:
        """
        Prints a grid with at each position a count of something occurring at that position in the dataset
        (e.g. the agent or the target object.)
        """
        file.write("Columns\n")
        for row in range(self._world.grid_size):
            row_print = "Row {}".format(row)
            file.write(row_print)
            file.write((8 - len(row_print)) * ' ')
            for column in range(self._world.grid_size):
                if (str(column), str(row)) in position_counts:
                    count = position_counts[(str(column), str(row))]
                else:
                    count = 0
                count_print = "({}, {}): {}".format(column, row, count)
                fill_spaces = 20 - len(count_print)
                file.write(count_print + fill_spaces * ' ')
            file.write("\n")
            file.write("\n")

    def count_equivalent_examples(self, split_1="train", split_2="test"):
        """TODO: only compare examples from which we know they have the same command to optimize performance."""
        print("WARNING: about to compare {} examples.".format(len(self._data_pairs[split_1]) * len(self._data_pairs[split_2])))
        equivalent_examples = 0
        for i, example_1 in enumerate(self._data_pairs[split_1]):
            template_identifier_1 = self._template_identifiers[split_1][i]
            for j, example_2 in enumerate(self._data_pairs[split_2]):
                template_identifier_2 = self._template_identifiers[split_2][j]
                if i != j and template_identifier_2 == template_identifier_1:
                    if self.compare_examples(example_1, example_2):
                        equivalent_examples += 1
        return equivalent_examples

    def compare_split_statistics(self, split_1, split_2):
        raise NotImplementedError()

    def save_dataset_statistics(self, split="train") -> {}:
        """
        Summarizes the statistics and prints them.
        """
        with open(os.path.join(self.save_directory, split + "_dataset_stats.txt"), 'w') as infile:
            # General statistics
            number_of_examples = len(self._data_pairs[split])
            infile.write("Number of examples: {}\n".format(number_of_examples))
            # Situation statistics.
            mean_distance_to_target = 0
            for distance_to_target, count in self._data_statistics[split]["distance_to_target"].items():
                mean_distance_to_target += count * distance_to_target
            mean_distance_to_target /= sum(self._data_statistics[split]["distance_to_target"].values())
            infile.write("Mean walking distance to target: {}\n".format(mean_distance_to_target))
            infile.write("Agent positions:\n")
            self.save_position_counts(self._data_statistics[split]["agent_position"], infile)
            infile.write("Target positions:\n")
            self.save_position_counts(self._data_statistics[split]["target_position"], infile)
            referred_targets = self._data_statistics[split]["referred_targets"]
            infile.write("\nReferred Targets: \n")
            for key, values in referred_targets.items():
                save_counter("  " + key, values, infile)
            placed_targets = self._data_statistics[split]["placed_targets"]
            infile.write("\n")
            save_counter("placed_targets", placed_targets, infile)
            situation_stats = self._data_statistics[split]["situations"]
            infile.write("\nObjects placed in the world for particular referenced objects: \n")
            for key, values in situation_stats.items():
                save_counter("  " + key, values["num_objects_placed"], infile)
                save_counter("  " + key, values["objects_in_world"], infile)

        for key, values in self._data_statistics[split]["situations"].items():
            if len(values["objects_in_world"]):
                bar_plot(values["objects_in_world"], key, os.path.join(self.save_directory, split + "_" + key + ".png"))

        for key in self.get_empty_situation().keys():
            occurrence_counter = self._data_statistics[split][key]
            if key != "agent_position" and key != "target_position" and key != "distance_to_target":
                bar_plot(occurrence_counter, key, os.path.join(self.save_directory, split + "_" + key + ".png"))

        # Command statistics.
        verbs_in_command = self._data_statistics[split]["verbs_in_command"]
        bar_plot(verbs_in_command, "verbs_in_command", os.path.join(self.save_directory,
                                                                    split + "_" + "verbs_in_command.png"))

    def save_dataset(self, file_name: str) -> str:
        """
        Saves the current generated data to a file in a particular format that is readable by load_examples_from_file.
        :param file_name: file name to save the dataset in. Will get saved in self.save_directory
        :return: path to saved file.
        """
        assert len(self._data_pairs) > 0, "No data to save, call .get_data_pairs()"
        output_path = os.path.join(self.save_directory, file_name)
        with open(output_path, 'w') as outfile:
            json.dump({
                "examples": {key: values for key, values in self._data_pairs.items()},
                "intransitive_verbs": self._vocabulary.verbs_intrans,
                "transitive_verbs": self._vocabulary.verbs_trans,
                "nouns": self._vocabulary.nouns,
                "adverbs": self._vocabulary.adverbs,
                "color_adjectives": self._vocabulary.color_adjectives,
                "size_adjectives": self._vocabulary.size_adjectives,
                "min_object_size": self._object_vocabulary.smallest_size,
                "max_object_size": self._object_vocabulary.largest_size,
                "max_recursion": self.max_recursion,
                "grid_size": self._world.grid_size
            }, outfile, indent=4)
        return output_path

    @classmethod
    def load_dataset_from_file(cls, file_path: str, save_directory: str):
        with open(os.path.join(os.getcwd(), file_path), 'r') as infile:
            all_data = json.load(infile)
            dataset = cls(all_data["intransitive_verbs"], all_data["transitive_verbs"], all_data["adverbs"],
                          all_data["nouns"], all_data["color_adjectives"], all_data["size_adjectives"],
                          all_data["grid_size"], all_data["min_object_size"], all_data["max_object_size"],
                          save_directory, all_data["max_recursion"])
            for split, examples in all_data["examples"].items():
                for example in examples:
                    dataset._data_pairs[split].append(example)
                    dataset.update_data_statistics(example, split)
            return dataset

    def generate_all_commands(self) -> {}:
        self._grammar.generate_all_commands()

    def sample_command(self) -> Tuple[Derivation, list]:
        coherent = False
        while not coherent:
            command = self._grammar.sample()
            arguments = []
            meaning = command.meaning(arguments)
            if not self._grammar.is_coherent(meaning):
                continue
            else:
                return command, arguments

    def demonstrate_target_commands(self, command: str, initial_situation: Situation,
                                    target_commands: List[str]) -> Tuple[List[str], List[Situation]]:
        # command = ' '.join(derivation.words())
        # arguments = []
        # derivation.meaning(arguments)
        current_situation = self._world.get_current_situation()
        current_mission = self._world.mission

        # Initialize the world based on the initial situation and the command.
        self.initialize_world(initial_situation, mission=command)

        for target_command in target_commands:
            self._world.execute_command(target_command)

        target_commands, target_demonstration = self._world.get_current_observations()
        self._world.clear_situation()

        # Re-initialize the world as before the command
        self.initialize_world(current_situation, mission=current_mission)
        return target_commands, target_demonstration

    def demonstrate_command(self, derivation: Derivation, initial_situation: Situation) -> Tuple[List[str],
                                                                                                 List[Situation], str]:
        """
        Demonstrate a command derivation and situation pair. Done by extracting the events from the logical form
        of the command derivation, extracting the arguments of each event. The argument of the event gets located in the
        situation of the world and the path to that target gets calculated. Based on whether the verb in the command is
        transitive or not, the agent interacts with the object.
        :param derivation:
        :param initial_situation:
        :returns
        """
        command = ' '.join(derivation.words())
        arguments = []
        logical_form = derivation.meaning(arguments)
        current_situation = self._world.get_current_situation()
        current_mission = self._world.mission

        # Initialize the world based on the initial situation and the command.
        self.initialize_world(initial_situation, mission=command)

        # Extract all present events in the current command and order them by constraints.
        events = [variable for variable in logical_form.variables if variable.sem_type == EVENT]
        seq_constraints = [term.arguments for term in logical_form.terms if term.function == "seq"]
        ordered_events = topo_sort(events, seq_constraints)

        # Loop over the events to get the demonstrations.
        action = None
        for event in ordered_events:

            # Get the logical form of the current event
            sub_logical_form = logical_form.select([event], exclude={"seq"})
            event_lf = sub_logical_form.select([event], exclude={"patient"})
            args = [term.arguments[1] for term in sub_logical_form.terms if term.function == "patient"]

            # Find the action verb if it exists.
            is_transitive = False
            if event_lf.head.sem_type == EVENT:
                for term in event_lf.terms:
                    if term.specs.action:
                        action = term.specs.action
                        is_transitive = term.specs.is_transitive

            # Find the manner adverb if it exists.  TODO: there can only be a manner if there is a verb?
            manner = [term.specs.manner for term in event_lf.terms if term.specs.manner]
            manner = manner.pop() if manner else None
            assert len(args) <= 1, "Only one target object supported, but two arguments parsed in a derivation."
            if len(args) > 0:
                # Find the logical form of the argument of the verb and find its location
                arg_logical_form = sub_logical_form.select([args[0]])
                object_str, object_predicate = arg_logical_form.to_predicate()

                # If no location is passed, find the target object there
                if not initial_situation.target_object:
                    # TODO: check if this works
                    if self._world.has_object(object_str):
                        object_locations = self._world.object_positions(object_str,
                                                                        object_size=object_predicate["size"])
                    else:
                        object_locations = {}
                # Else we have saved the target location when we generated the situation
                else:
                    object_locations = [initial_situation.target_object.position]

                if len(object_locations) > 1:
                    print("WARNING: {} possible target locations.".format(len(object_locations)))
                if not object_locations:
                    continue
                goal = random.sample(object_locations, 1).pop()
                if not is_transitive:
                    primitive_command = action
                else:
                    primitive_command = "walk"

                self._world.go_to_position(goal, manner, primitive_command=primitive_command)

                # Interact with the object for transitive verbs.
                if is_transitive:
                    self._world.push_object_to_wall()

        target_commands, target_demonstration = self._world.get_current_observations()
        self._world.clear_situation()

        # Re-initialize the world as before the command
        self.initialize_world(current_situation, mission=current_mission)
        return target_commands, target_demonstration, action

    def initialize_world(self, situation: Situation, mission="") -> {}:
        """
        Initializes the world with the passed situation.
        :param situation: class describing the current situation in the world, fully determined by a grid size,
        agent position, agent direction, list of placed objects, an optional target object and optional carrying object.
        :param mission: a string defining a command (e.g. "Walk to a green circle.")
        """
        objects = []
        for positioned_object in situation.placed_objects:
            objects.append((positioned_object.object, positioned_object.position))
        self._world.initialize(objects, agent_position=situation.agent_pos, agent_direction=situation.agent_direction,
                               target_object=situation.target_object, carrying=situation.carrying)
        if mission:
            self._world.set_mission(mission)

    def visualize_prediction(self, predictions_file: str) -> List[Tuple[str, str]]:
        assert os.path.exists(predictions_file), "Trying to open a non-existing predictions file."
        with open(predictions_file, 'r') as infile:
            data = json.load(infile)
            save_dirs = []
            for predicted_example in data:
                command = predicted_example["input"]
                prediction = predicted_example["prediction"]
                target = predicted_example["target"]
                situation_repr = predicted_example["situation"]
                situation = Situation.from_representation(situation_repr[0])
                predicted_commands, predicted_demonstration = self.demonstrate_target_commands(
                    command, situation, target_commands=prediction)
                target_commands, target_demonstration = self.demonstrate_target_commands(
                    command, situation, target_commands=target)
                str_command = ' '.join(command)
                save_dir_prediction = self.visualize_command(situation, str_command, predicted_demonstration,
                                                             prediction, add_to_save_dir="_predicted")
                save_dir_target = self.visualize_command(situation, str_command, target_demonstration, target_commands,
                                                         add_to_save_dir="_actual")
                save_dirs.append((save_dir_prediction, save_dir_target))
        return save_dirs

    def visualize_data_example(self, data_example: dict) -> str:
        command, derivation, situation, actual_target_commands, target_demonstration, _ = self.parse_example(
            data_example)
        save_dir = self.visualize_command(situation, ' '.join(command), target_demonstration,
                                          actual_target_commands)
        return save_dir

    def visualize_data_examples(self) -> List[str]:
        if len(self._examples_to_visualize) == 0:
            print("No examples to visualize.")
        save_dirs = []
        for data_example in self._examples_to_visualize:
            save_dir = self.visualize_data_example(data_example)
            save_dirs.append(save_dir)
        return save_dirs

    def visualize_command(self, initial_situation: Situation, command: str, demonstration: List[Situation],
                          target_commands: List[str], add_to_save_dir="") -> str:
        """

        :param initial_situation: (list of objects with their location, grid size, agent position)
        :param command: command in natural language
        :param demonstration: action sequence
        :param target_commands: TODO
        :return: path_to_visualization
        """
        # Save current situation
        current_situation = self._world.get_current_situation()
        current_mission = self._world.mission

        # Initialize directory with current command as its name.
        mission = ' '.join(["Command:", command, "\nTarget:"] + target_commands)
        mission_folder = command.replace(' ', '_')
        mission_folder += add_to_save_dir
        full_dir = os.path.join(self.save_directory, mission_folder)
        if not os.path.exists(full_dir):
            os.mkdir(full_dir)
            file_count = 0
        else:
            files_list = os.listdir(full_dir)
            file_count = len(files_list)
        mission_folder = os.path.join(mission_folder, "situation_{}".format(file_count))
        final_dir = os.path.join(full_dir, "situation_{}".format(file_count))
        if not os.path.exists(final_dir):
            os.mkdir(final_dir)

        # Visualize command
        self.initialize_world(initial_situation, mission=mission)
        save_location = self._world.save_situation(os.path.join(mission_folder, 'initial.png'))
        filenames = [save_location]

        for i, situation in enumerate(demonstration):
            self.initialize_world(situation, mission=mission)
            save_location = self._world.save_situation(os.path.join(mission_folder, 'situation_' + str(i) + '.png'))
            filenames.append(save_location)

        # Make a gif of the action sequence.
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        movie_dir = os.path.join(self.save_directory, mission_folder)
        imageio.mimsave(os.path.join(movie_dir, 'movie.gif'), images, fps=5)

        # Restore situation
        self.initialize_world(current_situation, mission=current_mission)

        return movie_dir

    def generate_possible_targets(self, referred_size: str, referred_color: str, referred_shape: str):
        """
        Generate a list of possible target objects based on some target referred to in a command, e.g.
        for small red circle any sized circle but the largest can be a potential target.
        """
        if referred_size:
            if referred_size == "small":
                target_sizes = self._object_vocabulary.object_sizes[:-1]
            elif referred_size == "big":
                target_sizes = self._object_vocabulary.object_sizes[1:]
            else:
                raise ValueError("Unknown size adjective in command.")
        else:
            target_sizes = self._object_vocabulary.object_sizes
        # If no color specified, use all colors.
        if not referred_color:
            target_colors = self._object_vocabulary.object_colors
        else:
            target_colors = [referred_color]

        # Return all possible combinations of sizes and colors
        return list(itertools.product(target_sizes, target_colors, [referred_shape]))

    def all_objects_except_shape(self, shape: str) -> List[tuple]:
        all_sizes = self._object_vocabulary.object_sizes
        all_colors = self._object_vocabulary.object_colors
        all_shapes = self._object_vocabulary.object_shapes
        all_shapes.remove(shape)
        return list(itertools.product(all_sizes, all_colors, all_shapes))

    def get_larger_sizes(self, size: int) -> List[int]:
        return list(range(size + 1, self._object_vocabulary.largest_size + 1))

    def get_smaller_sizes(self, size: int) -> List[int]:
        return list(range(self._object_vocabulary.smallest_size, size))

    def generate_distinct_objects(self, referred_size: str, referred_color: str, referred_shape: str,
                                  actual_size: int) -> Tuple[list, list]:
        """
        Generate a list of objects that are distinct from some referred target. E.g. if the referred target is a
        small circle, and the actual color of the target object is red, there cannot also be a blue circle of the same
        size, since then there will be 2 possible targets.
        Currently makes sure at least 2 sized objects of each group is placed whenever a size is referred to in the
        referred_size. E.g. if the command is 'walk to a big circle', make sure there are at least 2 sized circles.
        This doesn't get done for the color, e.g. if the comment is 'walk to a green circle', there are not
        necessarily also other colored circles in obligatory_objects.
        """
        objects = []
        # Initialize list that will be filled with objects that need to be present in the situation for it to make sense
        # E.g. if the referred object is 'small circle' there needs to be at least 1 larger circle.
        obligatory_objects = []
        # E.g. distinct from 'circle' -> no other circles; generate one random object of each other shape.
        if not referred_size and not referred_color:
            all_shapes = self._object_vocabulary.object_shapes
            all_shapes.remove(referred_shape)

            for shape in all_shapes:
                objects.append((self._object_vocabulary.sample_size(), self._object_vocabulary.sample_color(), shape))
            return objects, obligatory_objects
        # E.g. distinct from 'red circle' -> no other red circles of any size; generate one randomly size object for
        # each color, shape combination that is not a 'red circle'.
        elif not referred_size:
            for shape in self._object_vocabulary.object_shapes:
                for color in self._object_vocabulary.object_colors:
                    if not (shape == referred_shape and color == referred_color):
                        objects.append((self._object_vocabulary.sample_size(), color, shape))
            return objects, obligatory_objects
        else:
            if referred_size == "small":
                all_other_sizes = self.get_larger_sizes(actual_size)
            elif referred_size == "big":
                all_other_sizes = self.get_smaller_sizes(actual_size)
            else:
                raise ValueError("Unknown referred size in command")
            all_other_shapes = self._object_vocabulary.object_shapes
            all_other_shapes.remove(referred_shape)
            # E.g. distinct from 'small circle' -> no circles of size <= as target in any color; generate two
            # random sizes for each color-shape pair except for the shape that is referred generate one larger objects
            # (if referred size is small, else a smaller object)
            if not referred_color:
                for shape in self._object_vocabulary.object_shapes:
                    for color in self._object_vocabulary.object_colors:
                        if not shape == referred_shape:
                            for _ in range(2):
                                objects.append((self._object_vocabulary.sample_size(), color, shape))
                        else:
                            obligatory_objects.append((random.choice(all_other_sizes), color, shape))
                return objects, obligatory_objects
            # E.g. distinct from 'small red circle' -> no red circles of size <= as target; generate for each
            # color-shape pair two random sizes, and when the pair is the referred pair, one larger size.
            else:
                for shape in self._object_vocabulary.object_shapes:
                    for color in self._object_vocabulary.object_colors:
                        if not (shape == referred_shape and color == referred_color):
                            for _ in range(2):
                                objects.append((self._object_vocabulary.sample_size(), color, shape))
                        else:
                            obligatory_objects.append((random.choice(all_other_sizes), color, shape))
                return objects, obligatory_objects

    def generate_situations(self, num_resampling=1):
        """
        Generate all semantically distinct situations with an agent and a target object.
        Number of situations: TODO
        :return:
        """
        # All possible target objects
        all_targets = itertools.product(self._object_vocabulary.object_sizes, self._object_vocabulary.object_colors,
                                        self._object_vocabulary.object_shapes)

        # Loop over all semantically different situation specifications
        situation_specifications = {}
        for target_size, target_color, target_shape in all_targets:
            if target_shape not in situation_specifications.keys():
                situation_specifications[target_shape] = {}
            if target_color not in situation_specifications[target_shape].keys():
                situation_specifications[target_shape][target_color] = {}
            if target_size not in situation_specifications[target_shape][target_color].keys():
                situation_specifications[target_shape][target_color][target_size] = []

            for direction_str in self._relative_directions:

                # For straight directions (e.g. North, East, South and West) loop over 1 to grid size number of steps.
                if direction_str in self._straight_directions:
                    for num_steps_to_target in range(1, self._world.grid_size):
                        if 1 < num_steps_to_target < self._world.grid_size - 1:
                            num_to_resample = num_resampling
                        else:
                            num_to_resample = 1
                        for _ in range(num_to_resample):
                            empty_situation = self.get_empty_situation()
                            target_position = Position(column=self._world.grid_size + 1, row=self._world.grid_size + 1)
                            while not self._world.within_grid(target_position):
                                condition = {"n": 0, "e": 0, "s": 0, "w": 0}
                                condition[direction_str] = num_steps_to_target
                                agent_position = self._world.sample_position_conditioned(*condition.values())
                                target_position = self._world.get_position_at(agent_position, direction_str,
                                                                              num_steps_to_target)
                            assert self._world.within_grid(target_position) and self._world.within_grid(agent_position)
                            empty_situation["agent_position"] = agent_position
                            empty_situation["target_position"] = target_position
                            empty_situation["distance_to_target"] = num_steps_to_target
                            empty_situation["direction_to_target"] = direction_str
                            empty_situation["target_shape"] = target_shape
                            empty_situation["target_color"] = target_color
                            empty_situation["target_size"] = target_size
                            situation_specifications[target_shape][target_color][target_size].append(empty_situation)

                # For combined dirs (e.g. North-East, South-West, etc.) loop over 1 to 2 * grid size number of steps
                elif direction_str in self._combined_directions:
                    for number_of_steps_in_direction in range(2, 2 * (self._world.grid_size - 1) + 1):
                        if 1 < number_of_steps_in_direction < 2 * (self._world.grid_size - 1):
                            num_to_resample = num_resampling
                        else:
                            num_to_resample = 1
                        for _ in range(num_to_resample):
                            empty_situation = self.get_empty_situation()

                            # Randomly divide the number of steps over each direction of the combination
                            random_divide = random.randint(max(1, number_of_steps_in_direction - self._world.grid_size + 1),
                                                           min(number_of_steps_in_direction - 1, self._world.grid_size - 1))
                            steps_in_first_direction = random_divide
                            steps_in_second_direction = number_of_steps_in_direction - random_divide
                            assert (steps_in_second_direction + steps_in_first_direction) == number_of_steps_in_direction
                            assert (steps_in_first_direction and steps_in_second_direction) <= self._world.grid_size - 1
                            directions = list(direction_str)
                            target_position = Position(column=self._world.grid_size + 1, row=self._world.grid_size + 1)
                            while not self._world.within_grid(target_position):
                                condition = {"n": 0, "e": 0, "s": 0, "w": 0}
                                condition[directions[0]] = steps_in_first_direction
                                condition[directions[1]] = steps_in_second_direction
                                agent_position = self._world.sample_position_conditioned(*condition.values())
                                intermediate_target_position = self._world.get_position_at(agent_position, directions[0],
                                                                                           steps_in_first_direction)
                                target_position = self._world.get_position_at(intermediate_target_position,
                                                                              directions[1], steps_in_second_direction)
                            assert self._world.within_grid(target_position) and self._world.within_grid(agent_position)
                            empty_situation["agent_position"] = agent_position
                            empty_situation["target_position"] = target_position
                            empty_situation["distance_to_target"] = number_of_steps_in_direction
                            empty_situation["direction_to_target"] = direction_str
                            empty_situation["target_shape"] = target_shape
                            empty_situation["target_color"] = target_color
                            empty_situation["target_size"] = target_size
                            situation_specifications[target_shape][target_color][target_size].append(empty_situation)
        return situation_specifications

    def initialize_world_from_spec(self, situation_spec, referred_size: str, referred_color: str, referred_shape: str,
                                   actual_size: int, sample_percentage=0.5):
        self._world.clear_situation()
        self._world.place_agent_at(situation_spec["agent_position"])
        target_shape = situation_spec["target_shape"]
        target_color = situation_spec["target_color"]
        target_size = situation_spec["target_size"]
        self._world.place_object(Object(size=target_size, color=target_color, shape=target_shape),
                                 position=situation_spec["target_position"], target=True)
        distinct_objects, obligatory_objects = self.generate_distinct_objects(referred_size=referred_size,
                                                                              referred_color=referred_color,
                                                                              referred_shape=referred_shape,
                                                                              actual_size=actual_size)
        num_to_sample = int(len(distinct_objects) * sample_percentage)
        objects_to_place = obligatory_objects
        objects_to_place.extend(random.sample(distinct_objects, k=num_to_sample))
        for size, color, shape in objects_to_place:
            other_position = self._world.sample_position()
            self._world.place_object(Object(size=size, color=color, shape=shape), position=other_position)

    @staticmethod
    def command_repr(command: List[str]) -> str:
        return ','.join(command)

    @staticmethod
    def parse_command_repr(command_repr: str) -> List[str]:
        return command_repr.split(',')

    @staticmethod
    def derivation_repr(derivation: Derivation) -> str:
        return str(derivation)

    def parse_derivation_repr(self, derivation_repr: str) -> Derivation:
        command_rules, command_lexicon = derivation_repr.split(';')
        return Derivation.from_str(command_rules, command_lexicon, self._grammar)

    @staticmethod
    def position_repr(position: Position):
        return ','.join([str(position.column), str(position.row)])

    @staticmethod
    def parse_position_repr(position_repr: str) -> Position:
        column, row = position_repr.split(',')
        return Position(column=int(column), row=int(row))

    def get_data_pairs(self, max_examples=None, num_resampling=1, other_objects_sample_percentage=0.5,
                       visualize_per_template=0, train_percentage=0.8) -> {}:
        """
        Generate a set of situations and generate all possible commands based on the current grammar and lexicon,
        match commands to situations based on relevance (if a command refers to a target object, it needs to be
        present in the situation) and save these pairs in a the list of data examples.
        """
        # Save current situation of the world for later restoration.
        current_situation = self._world.get_current_situation()
        current_mission = self._world.mission

        # Generate all situations and commands
        situation_specifications = self.generate_situations(num_resampling=num_resampling)
        self.generate_all_commands()
        example_count = 0
        for template_num, template_derivations in self._grammar.all_derivations.items():
            visualized_per_template = 0
            for derivation in template_derivations:
                arguments = []
                derivation.meaning(arguments)
                assert len(arguments) == 1, "Only one target object currently supported."
                target_str, target_predicate = arguments.pop().to_predicate()
                possible_target_objects = self.generate_possible_targets(referred_size=target_predicate["size"],
                                                                         referred_color=target_predicate["color"],
                                                                         referred_shape=target_predicate["noun"])
                for target_size, target_color, target_shape in possible_target_objects:
                    relevant_situations = situation_specifications[target_shape][target_color][target_size]
                    num_relevant_situations = len(relevant_situations)
                    idx_to_visualize = random.sample([i for i in range(num_relevant_situations)], k=1).pop()
                    idx_for_train = random.sample([i for i in range(num_relevant_situations)], k=int(
                        num_relevant_situations * train_percentage))
                    idx_for_train = set(idx_for_train)
                    for i, relevant_situation in enumerate(relevant_situations):
                        visualize = False
                        if (example_count + 1) % 10000 == 0:
                            print("Number of examples: {}".format(example_count + 1))
                        if max_examples:
                            if example_count >= max_examples:
                                return

                        self.initialize_world_from_spec(relevant_situation, referred_size=target_predicate["size"],
                                                        referred_color=target_predicate["color"],
                                                        referred_shape=target_predicate["noun"],
                                                        actual_size=target_size,
                                                        sample_percentage=other_objects_sample_percentage
                                                        )
                        situation = self._world.get_current_situation()
                        assert situation.direction_to_target == relevant_situation["direction_to_target"]
                        assert situation.distance_to_target == relevant_situation["distance_to_target"]
                        target_commands, target_situations, target_action = self.demonstrate_command(
                            derivation, initial_situation=situation)
                        if i == idx_to_visualize:
                            visualize = True
                        if visualized_per_template >= visualize_per_template:
                            visualize = False
                        if i in idx_for_train:
                            split = "train"
                        else:
                            split = "test"
                        self.fill_example(command=derivation.words(), derivation=derivation,
                                          situation=situation, target_commands=target_commands,
                                          verb_in_command=target_action, target_predicate=target_predicate,
                                          visualize=visualize, split=split)
                        self._template_identifiers[split].append(template_num)
                        example_count += 1
                        if visualize:
                            visualized_per_template += 1

                        self._world.clear_situation()

        # restore situation
        self.initialize_world(current_situation, mission=current_mission)
        return

    def random_color(self) -> str:
        return np.random.choice(self._vocabulary.color_adjectives)

    def random_size(self) -> str:
        return np.random.choice(self._vocabulary.size_adjectives)

    def random_shape(self) -> str:
        return np.random.choice(self._vocabulary.nouns)

    def random_adverb(self) -> str:
        return np.random.choice(self._vocabulary.adverbs)

    def random_adjective(self) -> str:
        return self._vocabulary.adjectives[np.random.randint(len(self._vocabulary.adjectives))][0]

    def assign_split(self, command, demo):
        # Split information
        # TODO: this doesn't seem like it should be in the Grammar class.
        # Sample a random object
        random_object = lambda: np.random.binomial(1, 0.5, size=8)

        # Move this outside of default constructor like with Vocabulary.sample()
        hold_out_object = random_object()
        hold_out_adverb = self.random_adverb()
        hold_out_adjective = self.random_adjective()
        hold_out_application = (self.random_adjective(), random_object())
        hold_out_composition = (self.random_adjective(), self.random_adjective())
        hold_out_recursion = (self.random_adjective(), np.random.randint(self.max_recursion))
        arguments = []
        logical_form = command.meaning(arguments)

        if any(term.function == hold_out_adverb for term in logical_form.terms):
            return "adverb"

        if any(term.function == hold_out_adjective for term in logical_form.terms):
            return "adjective"

        if any(
            term_1.function == hold_out_composition[0]
            and term_2.function == hold_out_composition[1]
            and term_1.arguments == term_2.arguments
            for term_1 in logical_form.terms
            for term_2 in logical_form.terms
        ):
            return "composition"

        if any(
            term.function == hold_out_recursion[0]
            and term.meta is not None
            and term.meta["recursion"] == hold_out_recursion[1]
            for term in logical_form.terms
        ):
            return "recursion"

        for command, situation, _ in demo:
            if command is None or command.event is None:
                continue
            event_logical_form = logical_form.select([command.event])
            arguments = [
                term.arguments[1] for term in event_logical_form.terms
                if term.function == "patient" and term.arguments[0] == command.event
            ]
            if len(arguments) == 0:
                continue
            arg_var, = arguments
            arg_object = situation.grid[situation.agent_pos]

            if (arg_object == hold_out_object).all():
                return "object"

            if (
                any(
                    term.function == hold_out_application[0]
                    for term in logical_form.terms
                    if term.arguments[0] == arg_var
                )
                and (arg_object == hold_out_application[1]).all()
            ):
                return "application"

        return "main"
