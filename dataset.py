from world import LogicalForm
from world import Situation
from world import EVENT
from world import Object
from world import Position
from world import PositionedObject
from world import World
from world import EAST
from vocabulary import Vocabulary
from helpers import topo_sort

from typing import List
from typing import Set
import numpy as np
import os
import imageio
import random


class GroundedScan(object):

    def __init__(self, intransitive_verbs: List[str], transitive_verbs: List[str], adverbs: List[str], nouns: List[str],
                 color_adjectives: List[str], size_adjectives: List[str], grid_size: int,
                 save_directory=os.getcwd(), max_recursion=1):

        # The vocabulary
        self.vocabulary = Vocabulary(verbs_intrans=intransitive_verbs, verbs_trans=transitive_verbs, adverbs=adverbs,
                                     nouns=nouns, color_adjectives=color_adjectives, size_adjectives=size_adjectives)

        # The grammar used to generate the commands
        # TODO (dependent on split?)
        self.max_recursion = max_recursion

        self.save_directory = save_directory

        # Initialize the world
        # TODO: multiple grid sizes?
        self.world = World(grid_size=grid_size, color_adjectives=self.vocabulary.color_adjectives,
                           size_adjectives=self.vocabulary.size_adjectives, shape_nouns=self.vocabulary.nouns,
                           save_directory=self.save_directory)

    def demonstrate_command(self, command: str, logical_form: LogicalForm, initial_situation: Situation):
        current_situation = self.world.get_current_situation()
        current_mission = self.world.mission

        # Initialize the world based on the initial situation and the command.
        self.initialize_world(initial_situation, mission=command)

        # Extract all present events in the current command and order them by constraints.
        events = [variable for variable in logical_form.variables if variable.sem_type == EVENT]
        seq_constraints = [term.arguments for term in logical_form.terms if term.function == "seq"]
        ordered_events = topo_sort(events, seq_constraints)

        # Initialize the resulting demonstration
        demonstration = [initial_situation]

        # Loop over the events to get the demonstrations
        for event in ordered_events:

            # Get the logical form of the current event
            sub_logical_form = logical_form.select([event], exclude={"seq"})
            event_lf = sub_logical_form.select([event], exclude={"patient"})
            args = [term.arguments[1] for term in sub_logical_form.terms if term.function == "patient"]

            # Find the action verb if it exists.
            action = None
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
            if len(args) == 0:
                # No step
                demonstration.append(self.world.get_current_situation())
            else:
                # Find the logical form of the argument of the verb and find its location
                arg_logical_form = sub_logical_form.select([args[0]])
                object_locations = arg_logical_form.to_predicate(self.world.object_lookup_table)
                if not object_locations:
                    continue
                object_name, goal = random.sample(object_locations.items(), 1).pop()
                sampled_goal = random.sample(goal, 1).pop()
                self.world.go_to_position(sampled_goal, demonstration, manner)

                # Interact with the object for transitive verbs.
                if is_transitive:
                    self.world.pick_up_object(demonstration)
                    if self.world.carrying:
                        sampled_goal = self.sample_position(set(), set())
                        self.world.go_to_position(sampled_goal, demonstration, manner)
                        self.world.drop_object(demonstration)

        # Re-initialize the world as before the command
        self.initialize_world(current_situation, mission=current_mission)
        return demonstration

    def generate_split(self, split: str):
        """
        Generate a particular split of grounded SCAN.
        :param split: which split
        :return:
        """
        raise NotImplementedError()

    def initialize_world(self, situation: Situation, mission=""):
        objects = []
        for positioned_object in situation.placed_objects:
            objects.append((positioned_object.object, positioned_object.position))
        self.world.initialize(objects, agent_position=situation.agent_pos, agent_direction=situation.agent_direction,
                              carrying=situation.carrying)
        if mission:
            self.world.set_mission(mission)

    def visualize_command(self, initial_situation, command, demonstration) -> str:
        """

        :param situation: (list of objects with their location, grid size, agent position)
        :param command: command in natural language
        :param demonstration: action sequence
        :return: gif
        """
        # Save current situation
        current_situation = self.world.get_current_situation()
        current_mission = self.world.mission

        # Initialize directory with current command as its name.
        mission_folder = command.replace(' ', '_')
        full_dir = os.path.join(self.save_directory, mission_folder)
        if not os.path.exists(full_dir):
            os.mkdir(full_dir)

        # Visualize command
        self.initialize_world(initial_situation, mission=command)
        save_location = self.world.save_situation(os.path.join(mission_folder, 'initial.png'))
        filenames = [save_location]

        for i, situation in enumerate(demonstration):
            self.initialize_world(situation, mission=command)
            save_location = self.world.save_situation(os.path.join(mission_folder, 'situation_' + str(i) + '.png'))
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

    def save_data_set(self):
        """
        Need to be saved to a file such that that file can be used to reload this class object again
        :return:
        """
        raise NotImplementedError()

    @classmethod
    def load_from_file(cls, file_name: str):
        """
        TODO: load a dataset from a file
        :param file_name:
        :return:
        """
        raise NotImplementedError()

    def sample_object(self) -> Object:
        color = self.random_color()
        size = self.random_size()
        shape = self.random_shape()
        return Object(size=size, color=color, shape=shape)

    def sample_position(self, occupied_rows: Set[int], occupied_cols: Set[int]) -> Position:
        available_rows = [row for row in range(self.world.grid_size) if row not in occupied_rows]
        available_cols = [col for col in range(self.world.grid_size) if col not in occupied_cols]
        sampled_row = random.sample(available_rows, 1).pop()
        sampled_col = random.sample(available_cols, 1).pop()
        occupied_rows.add(sampled_row)
        occupied_cols.add(sampled_col)
        return Position(row=sampled_row, column=sampled_col)

    def sample_situation(self, num_objects) -> Situation:
        assert num_objects < self.world.grid_size**2 - 1
        objects = []
        occupied_rows = set()
        occupied_cols = set()
        agent_position = self.sample_position(occupied_rows, occupied_cols)
        agent_direction = EAST
        for i in range(num_objects):
            object = self.sample_object()
            position = self.sample_position(occupied_rows, occupied_cols)
            object_vector = self.world.object_vocabulary.object_vectors[object.shape][object.color][object.size]
            objects.append(PositionedObject(object=object, position=position, vector=object_vector))
        return Situation(agent_direction=agent_direction, agent_position=agent_position, placed_objects=objects,
                         grid_size=self.world.grid_size)

    def random_color(self) -> str:
        return np.random.choice(self.vocabulary.color_adjectives)

    def random_size(self) -> str:
        return np.random.choice(self.vocabulary.size_adjectives)

    def random_shape(self) -> str:
        return np.random.choice(self.vocabulary.nouns)

    def random_adverb(self) -> str:
        return np.random.choice(self.vocabulary.adverbs)

    def random_adjective(self) -> str:
        return self.vocabulary.adjectives[np.random.randint(len(self.vocabulary.adjectives))][0]

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
        logical_form = command.meaning()

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
