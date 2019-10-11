import numpy as np
from typing import Tuple
from gym_minigrid.minigrid import DIR_TO_VEC
from gridworld import GridWorld


# TODO faster
def topo_sort(items, constraints):
    if not constraints:
        return items
    items = list(items)
    constraints = list(constraints)
    out = []
    while len(items) > 0:
        roots = [
            i for i in items
            if not any(c[1] == i for c in constraints)
        ]
        assert len(roots) > 0, (items, constraints)
        to_pop = roots[0]
        items.remove(to_pop)
        constraints = [c for c in constraints if c[0] != to_pop]
        out.append(to_pop)
    return out


def random_weights(size: int):
    return 2 * (np.random.random(size) - 0.5)


def accept_weights(size: int):
    return np.ones(size)


def plan_step(position: Tuple[int, int], move_direction: int):
    """

    :param position: current position of form (x-axis, y-axis) (i.e. column, row)
    :param move_direction: East is 0, south is 1, west is 2, north is 3.
    :return: next position of form (x-axis, y-axis) (i.e. column, row)
    """
    assert 0 <= move_direction < 4
    dir_vec = DIR_TO_VEC[move_direction]
    return position + dir_vec


def visualize_action_sequence(example: Tuple, visualization_directory: str) -> str:
    command, demonstration = example
    initial_situation = demonstration[0][1]
    grid_size = initial_situation.grid_size
    action_sequence = [action for _, _, action in demonstration]
    gym_world = GridWorld(command=' '.join(command.words()), save_directory=visualization_directory,
                          size=grid_size, agent_start_pos=(0, 0))
    gym_world.place_objects(initial_situation.objects)
    save_directory = gym_world.visualize_sequence(action_sequence)
    return save_directory
