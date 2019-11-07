from world import Situation
from world import EVENT
from world import Object
from world import Position
from world import PositionedObject
from world import World
from world import EAST
from world import ObjectVocabulary
from grammar import Grammar
from grammar import Derivation
from vocabulary import Vocabulary
from helpers import topo_sort
from helpers import print_counter

import json
import statistics
from collections import Counter
from collections import defaultdict
from typing import List
from typing import Tuple
from typing import Set
from typing import Dict
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
        self._data_pairs = []
        self._data_statistics = self.get_empty_data_statistics()

    def fill_example(self, command: List[str], derivation: Derivation, situation: Situation, target_commands: List[str],
                     verb_in_command: str, target_predicate: dict):
        example = {
            "command": self.command_repr(command),
            "derivation": self.derivation_repr(derivation),
            "situation": situation.to_representation(),
            "target_commands": self.command_repr(target_commands),
            "verb_in_command": verb_in_command,
            "referred_target": ' '.join([target_predicate["size"], target_predicate["color"],
                                         target_predicate["noun"]])
        }
        self._data_pairs.append(example)
        self.update_data_statistics(example)
        return example

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

    @staticmethod
    def get_empty_data_statistics():
        return {
            "distance_to_target": [],
            "direction_to_target": [],
            "target_shape": [],
            "target_color": [],
            "target_size": [],
            "target_position": [],
            "agent_position": [],
            "verbs_in_command": defaultdict(int),
            "referred_targets": defaultdict(int),
            "placed_targets": defaultdict(int)
        }

    @staticmethod
    def get_example_specification():
        """Represents how one data example can be fully specified for saving to a file and loading from a file."""
        return ["command", "command_rules", "command_lexicon", "target_commands", "agent_position",
                "target_position", "distance_to_target", "direction_to_target", "target_shape",
                "target_color", "target_size", "situation"]

    def update_data_statistics(self, data_example):
        """Keeps track of certain statistics regarding the data pairs generated."""

        # Update the statistics regarding the situation.
        self._data_statistics["distance_to_target"].append(int(data_example["situation"]["distance_to_target"]))
        self._data_statistics["direction_to_target"].append(data_example["situation"]["direction_to_target"])
        target_size = data_example["situation"]["target_object"]["object"]["size"]
        target_color = data_example["situation"]["target_object"]["object"]["color"]
        target_shape = data_example["situation"]["target_object"]["object"]["shape"]
        self._data_statistics["target_shape"].append(target_shape)
        self._data_statistics["target_color"].append(target_color)
        self._data_statistics["target_size"].append(target_size)
        self._data_statistics["target_position"].append(
            (data_example["situation"]["target_object"]["position"]["column"],
             data_example["situation"]["target_object"]["position"]["row"]))
        self._data_statistics["agent_position"].append((data_example["situation"]["agent_position"]["column"],
                                                        data_example["situation"]["agent_position"]["row"]))
        placed_target = ' '.join([str(target_size), target_color, target_shape])
        self._data_statistics["placed_targets"][placed_target] += 1

        # Update the statistics regarding the command.
        self._data_statistics["verbs_in_command"][data_example["verb_in_command"]] += 1
        self._data_statistics["referred_targets"][data_example["referred_target"]] += 1

    def print_position_counts(self, position_counts) -> {}:
        """
        Prints a grid with at each position a count of something occurring at that position in the dataset
        (e.g. the agent or the target object.)
        """
        print("Columns")
        for row in range(self._world.grid_size):
            row_print = "Row {}".format(row)
            print(row_print, end='')
            print((8 - len(row_print)) * ' ')
            for column in range(self._world.grid_size):
                if (str(column), str(row)) in position_counts:
                    count = position_counts[(str(column), str(row))]
                else:
                    count = 0
                count_print = "({}, {}): {}".format(column, row, count)
                fill_spaces = 20 - len(count_print)
                print(count_print + fill_spaces * ' ', end='')
            print()
            print()

    def print_dataset_statistics(self) -> {}:
        """
        Summarizes the statistics and prints them.
        """
        # General statistics
        number_of_examples = len(self._data_pairs)
        print("Number of examples: {}".format(number_of_examples))

        # Situation statistics.
        mean_distance_to_target = statistics.mean(self._data_statistics["distance_to_target"])
        print("Mean walking distance tot target: {}".format(mean_distance_to_target))
        for key in self.get_empty_situation().keys():
            occurrence_counter = Counter(self._data_statistics[key])
            if key != "agent_position" and key != "target_position" and key != "distance_to_target":
                print(key + ": ")
                for current_stat, occurrence_count in occurrence_counter.items():
                    print("   {}: {}".format(current_stat, occurrence_count))
            elif key != "distance_to_target":
                print(key + ": ")
                self.print_position_counts(occurrence_counter)

        # Command statistics.
        verbs_in_command = self._data_statistics["verbs_in_command"]
        print_counter("verbs_in_command", verbs_in_command)
        referred_targets = self._data_statistics["referred_targets"]
        print_counter("referred_targets", referred_targets)
        placed_targets = self._data_statistics["placed_targets"]
        print_counter("placed_targets", placed_targets)

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
                "examples": self._data_pairs,
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
            dataset = GroundedScan(all_data["intransitive_verbs"], all_data["transitive_verbs"], all_data["adverbs"],
                                   all_data["nouns"], all_data["color_adjectives"], all_data["size_adjectives"],
                                   all_data["grid_size"], all_data["min_object_size"], all_data["max_object_size"],
                                   save_directory, all_data["max_recursion"])
            for example in all_data["examples"]:
                dataset._data_pairs.append(example)
                dataset.update_data_statistics(example)
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

    def demonstrate_target_commands(self, derivation: Derivation, initial_situation: Situation,
                                    target_commands: List[str]) -> Tuple[List[str], List[Situation]]:
        command = ' '.join(derivation.words())
        arguments = []
        derivation.meaning(arguments)
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
        Option to pass an initial situation or sample a relevant situation based on the target objects.
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
            assert len(args) <= 1
            if len(args) > 0:
                # Find the logical form of the argument of the verb and find its location
                arg_logical_form = sub_logical_form.select([args[0]])
                object_str, object_predicate = arg_logical_form.to_predicate()

                # If no location is passed, find the target object there
                if not initial_situation.target_object:
                    if self._world.has_object(object_str):
                        object_locations = self._world.object_positions(object_str,
                                                                        object_size=object_predicate["size"])
                    else:
                        object_locations = {}
                # Else we have saved the target location when we generated the situation
                else:
                    object_locations = [[initial_situation.target_object.position]]

                if len(object_locations) > 1:
                    print("WARNING: {} possible target locations.".format(len(object_locations)))
                if not object_locations:
                    continue
                goal = random.sample(object_locations, 1).pop()
                sampled_goal = random.sample(goal, 1).pop()
                if not is_transitive:
                    primitive_command = action
                else:
                    primitive_command = "walk"

                self._world.go_to_position(sampled_goal, manner, primitive_command=primitive_command)

                # Interact with the object for transitive verbs.
                if is_transitive:
                    self._world.push_object_to_wall()

        target_commands, target_demonstration = self._world.get_current_observations()
        self._world.clear_situation()

        # Re-initialize the world as before the command
        self.initialize_world(current_situation, mission=current_mission)
        return target_commands, target_demonstration, action

    def initialize_world(self, situation: Situation, mission=""):
        """

        :param situation:
        :param mission:
        :return:
        """
        objects = []
        for positioned_object in situation.placed_objects:
            objects.append((positioned_object.object, positioned_object.position))
        self._world.initialize(objects, agent_position=situation.agent_pos, agent_direction=situation.agent_direction,
                               target_object=situation.target_object, carrying=situation.carrying)
        if mission:
            self._world.set_mission(mission)

    def visualize_data_examples(self, num_to_visualize=1) -> List[str]:
        assert len(self._data_pairs) > num_to_visualize, "Not enough examples in dataset to visualize."
        save_dirs = []
        for data_example in self._data_pairs[0:num_to_visualize]:
            command = self.parse_command_repr(data_example["command"])
            situation = Situation
            situation.from_representation(data_example["situation"])
            target_commands = self.parse_command_repr(data_example["target_commands"])
            derivation = self.parse_derivation_repr(data_example["derivation"])
            assert self.derivation_repr(derivation) == data_example["derivation"]
            actual_target_commands, target_demonstration, action = self.demonstrate_command(derivation, situation)
            assert self.command_repr(actual_target_commands) == self.command_repr(target_commands)
            save_dir = self.visualize_command(situation, ' '.join(command), target_demonstration,
                                              actual_target_commands)
            save_dirs.append(save_dir)
        return save_dirs

    def visualize_command(self, initial_situation: Situation, command: str, demonstration: List[Situation],
                          target_commands: List[str]) -> str:
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
                                  actual_size: int) -> list:
        """
        Generate a list of objects that are distinct from some referred target. E.g. if the referred target is a
        small circle, and the actual color of the target object is red, there cannot also be a blue circle of the same
        size, since then there will be 2 possible targets.
        """
        # E.g. distinct from 'circle' -> no other circles.
        if not referred_size and not referred_color:
            return self.all_objects_except_shape(referred_shape)
        # E.g. distinct from 'red circle' -> no other red circles of any size.
        elif not referred_size:
            all_sizes = self._object_vocabulary.object_sizes
            other_colors = self._object_vocabulary.object_colors
            other_shapes = self._object_vocabulary.object_shapes
            other_shapes.remove(referred_shape)
            other_colors.remove(referred_color)
            other_objects = list(itertools.product(all_sizes, other_colors, [referred_shape])) + \
                list(itertools.product(all_sizes, self._object_vocabulary.object_colors, other_shapes))
            return other_objects
        else:
            colors = self._object_vocabulary.object_colors
            all_sizes = self._object_vocabulary.object_sizes
            if referred_size == "small":
                all_other_sizes = self.get_larger_sizes(actual_size)
            elif referred_size == "big":
                all_other_sizes = self.get_smaller_sizes(actual_size)
            else:
                raise ValueError("Unknown referred size in command")
            all_other_shapes = self._object_vocabulary.object_shapes
            all_other_shapes.remove(referred_shape)
            # E.g. distinct from 'small circle' -> no circles of size <= as target in any color.
            if not referred_color:
                all_other_objects = list(itertools.product(all_other_sizes, colors, [referred_shape])) + \
                    list(itertools.product(all_sizes, colors, all_other_shapes))
                return all_other_objects
            # E.g. distinct from 'small red circle' -> no red circles of size <= as target.
            else:
                all_other_colors = self._object_vocabulary.object_colors
                all_other_colors.remove(referred_color)
                all_other_objects = list(itertools.product(all_other_sizes, [referred_color], [referred_shape])) + \
                    list(itertools.product(all_sizes, all_other_colors, [referred_shape])) + \
                    list(itertools.product(all_sizes, self._object_vocabulary.object_colors, all_other_shapes))
                return all_other_objects

    def generate_situations(self):
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
                                   actual_size: int):
        self._world.clear_situation()
        self._world.place_agent_at(situation_spec["agent_position"])
        target_shape = situation_spec["target_shape"]
        target_color = situation_spec["target_color"]
        target_size = situation_spec["target_size"]
        self._world.place_object(Object(size=target_size, color=target_color, shape=target_shape),
                                 position=situation_spec["target_position"], target=True)
        distinct_objects = self.generate_distinct_objects(referred_size=referred_size, referred_color=referred_color,
                                                          referred_shape=referred_shape, actual_size=actual_size)
        for size, color, shape in distinct_objects:
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

    def get_data_pairs(self, max_examples=None) -> {}:
        """
        Generate a set of situations and generate all possible commands based on the current grammar and lexicon,
        match commands to situations based on relevance (if a command refers to a target object, it needs to be
        present in the situation) and save these pairs in a the list of data examples.
        """
        # Save current situation of the world for later restoration.
        current_situation = self._world.get_current_situation()
        current_mission = self._world.mission

        # Generate all situations and commands
        situation_specifications = self.generate_situations()
        self.generate_all_commands()
        example_count = 0
        for derivation in self._grammar.all_derivations:
            arguments = []
            derivation.meaning(arguments)
            assert len(arguments) == 1, "Only one target object currently supported."
            target_str, target_predicate = arguments.pop().to_predicate()
            possible_target_objects = self.generate_possible_targets(referred_size=target_predicate["size"],
                                                                     referred_color=target_predicate["color"],
                                                                     referred_shape=target_predicate["noun"])
            for target_size, target_color, target_shape in possible_target_objects:

                relevant_situations = situation_specifications[target_shape][target_color][target_size]
                for relevant_situation in relevant_situations:
                    if (example_count + 1) % 10000 == 0:
                        print("Number of examples: {}".format(example_count + 1))
                    if max_examples:
                        if example_count >= max_examples:
                            return
                    self.initialize_world_from_spec(relevant_situation, referred_size=target_predicate["size"],
                                                    referred_color=target_predicate["color"],
                                                    referred_shape=target_predicate["noun"], actual_size=target_size)
                    situation = self._world.get_current_situation()
                    assert situation.direction_to_target == relevant_situation["direction_to_target"]
                    assert situation.distance_to_target == relevant_situation["distance_to_target"]
                    target_commands, target_situations, target_action = self.demonstrate_command(
                        derivation, initial_situation=situation)
                    self.fill_example(command=derivation.words(), derivation=derivation, situation=situation,
                                      target_commands=target_commands, verb_in_command=target_action,
                                      target_predicate=target_predicate)
                    example_count += 1
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
