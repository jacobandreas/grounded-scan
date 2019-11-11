import numpy as np
from typing import Tuple
from typing import List
from typing import Dict
from gym_minigrid.minigrid import DIR_TO_VEC
import matplotlib
import matplotlib.pyplot as plt


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


def random_weights(size: int) -> np.ndarray:
    return 2 * (np.random.random(size) - 0.5)


def accept_weights(size: int) -> np.ndarray:
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


def one_hot(size: int, idx: int) -> np.ndarray:
    one_hot_vector = np.zeros(size, dtype=int)
    one_hot_vector[idx] = 1
    return one_hot_vector


def generate_possible_object_names(color: str, shape: str) -> List[str]:
    # TODO: does this still make sense when size is not small or large
    names = [shape, ' '.join([color, shape])]
    return names


def save_counter(description, counter, file):
    file.write(description + ": \n")
    for key, occurrence_count in counter.items():
        file.write("   {}: {}\n".format(key, occurrence_count))


def bar_plot(values: dict, title: str, save_path: str):
    # TODO: x axis higher, also plot bars for objects not present! (maybe just change defaults in get_empty_stats
    sorted_values = list(values.items())
    sorted_values = [(y, x) for x, y in sorted_values]
    sorted_values.sort()
    values_per_label = [value[0] for value in sorted_values]
    labels = [value[1] for value in sorted_values]
    assert len(labels) == len(values_per_label)
    y_pos = np.arange(len(labels))

    plt.bar(y_pos, values_per_label, align='center', alpha=0.5)
    plt.gcf().subplots_adjust(bottom=0.2, )
    plt.xticks(y_pos, labels, rotation=90, fontsize="xx-small")
    plt.ylabel('Occurrence')
    plt.title(title)

    plt.savefig(save_path)
    plt.close()
