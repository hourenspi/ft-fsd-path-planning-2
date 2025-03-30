#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: Combines the results of the search along the left and right traces
Project: fsd_path_planning
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from fsd_path_planning.sorting_cones.trace_sorter.line_segment_intersection import (
    lines_segments_intersect_indicator,
)
from fsd_path_planning.types import FloatArray, IntArray
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import (
    angle_difference,
    angle_from_2d_vector,
    my_njit,
)

# richiamato da core_trace_sorter.py
# in def sort_left_right
def calc_final_configs_for_left_and_right(
    left_scores: Optional[FloatArray], # calcolati da calc_configurations_with_score_for_one_side
    # costi, configurazioni, history
    left_configs: Optional[IntArray],
    right_scores: Optional[FloatArray],
    right_configs: Optional[IntArray],
    cones: FloatArray, # vettore flat
    car_pos: FloatArray,
    car_dir: FloatArray,
) -> tuple[IntArray, IntArray]: # ritorna right_config, left_config
    left_score_is_none = left_scores is None #boolean
    left_config_is_none = left_configs is None
    assert left_score_is_none == left_config_is_none

    #flags
    right_score_is_none = right_scores is None
    right_config_is_none = right_configs is None
    assert right_score_is_none == right_config_is_none

    # somma dei costi 
    n_non_none = sum(x is not None for x in (left_scores, right_scores))

    # if both sides are None, we have no valid configuration
    empty_config = np.zeros(0, dtype=int) # array vuoto di int
    empty_result = (empty_config, empty_config)

    if n_non_none == 0:
        return empty_result

    if n_non_none == 1:
        # only one side has a valid configuration
        # prende
        return calc_final_configs_when_only_one_side_has_configs(
            left_configs,
            right_configs,
        )

    # both sides have valid configurations
    # we need to pick the best one for each side

    return calc_final_configs_when_both_available(
        left_scores,
        left_configs,
        right_scores,
        right_configs,
        cones,
        car_pos,
        car_dir,
    )

# richiamato da calc_final_configs_for_left_and_right
# (combine_traces.py)
def calc_final_configs_when_only_one_side_has_configs(
    left_configs: Optional[IntArray],
    right_configs: Optional[IntArray],
) -> tuple[IntArray, IntArray]: # array left_config, right_config
    empty_config = np.zeros(0, dtype=int)

    left_config_is_none = left_configs is None
    right_config_is_none = right_configs is None

    assert left_config_is_none != right_config_is_none

    if left_configs is None:
        left_config = empty_config
        # prende la prima perchè array sorted
        # prendo quella col costo minore
        right_config = right_configs[0]
        right_config = right_config[right_config != -1]
        # configurazione non valide settate a -1
    elif right_configs is None:
        right_config = empty_config
        left_config = left_configs[0]
        left_config = left_config[left_config != -1]
    else:
        raise ValueError("Should not happen")

    return left_config, right_config
    # ritorna array vuoto, e uno con configurazioni valide

# richiamato da calc_final_configs_for_left_and_right
def calc_final_configs_when_both_available(
    left_scores: FloatArray,
    left_configs: IntArray,
    right_scores: FloatArray,
    right_configs: IntArray,
    cones: FloatArray,
    car_position: FloatArray,
    car_direction: FloatArray,
) -> tuple[IntArray, IntArray]:
    # we need to pick the best one for each side
    # ovvero la prima configurazione, quella con costo minore
    left_config = left_configs[0]
    left_config = left_config[left_config != -1]

    right_config = right_configs[0]
    right_config = right_config[right_config != -1]

    # PRENDO CONFIGURAZIONI MODIFICATE IN BASE ALL'INDICE COMUNE E LA DISTANZA DAI CONI PRECEDENTI
    left_config, right_config = handle_same_cone_in_both_configs(
        cones, left_config, right_config
    )

    # ritorna configurazioni
    return (left_config, right_config)

# richiamato da calc_final_configs_when_both_available
# (combine_traces.py)
def handle_same_cone_in_both_configs(
    cones: FloatArray,
    left_config: IntArray,
    right_config: IntArray,
) -> tuple[Optional[IntArray], Optional[IntArray]]:
    (
        same_cone_intersection,
        left_intersection_idxs,
        right_intersection_idxs,
    ) = np.intersect1d(left_config, right_config, return_indices=True)
    # Find the intersection of two arrays.
    # ritorna: 1. Sorted 1D array of common and unique elements.
    # 2. The indices of the first occurrences of the common values in ar1. Only provided if return_indices is True
    # 3. The indices of the first occurrences of the common values in ar2. Only provided if return_indices is True.
    if len(same_cone_intersection) == 0:
        return left_config, right_config

    left_intersection_index = min(
        left_intersection_idxs
    )  # first index of common cone in left config
    right_intersection_index = min(
        right_intersection_idxs
    )  # first index of common cone in right config

    # if both sides have the same FIRST common cone, then we try to find the
    # side to which the cone probably belongs
    (
        left_stop_idx,
        right_stop_idx,
    ) = calc_new_length_for_configs_for_same_cone_intersection(
        cones,
        left_config,
        right_config,
        left_intersection_index,
        right_intersection_index,
    )

    # arriva fino all'indice di stop ritornato dalla funzione richiamata
    left_config = left_config[:left_stop_idx]
    right_config = right_config[:right_stop_idx]

    return left_config, right_config
# TODO: RIVEDI QUANDO NON LAVORI
# richiamato da handle_same_cone_in_both_configs
# (combine_traces.py)
def calc_new_length_for_configs_for_same_cone_intersection(
    cones: FloatArray,
    left_config: IntArray,
    right_config: IntArray,
    left_intersection_index: int,
    right_intersection_index: int,
) -> tuple[int, int]: # ritorna indici a cui fermarsi
    cones_xy = cones[:, :2]
    if left_intersection_index > 0 and right_intersection_index > 0:
        prev_left = left_config[left_intersection_index - 1]
        prev_right = right_config[right_intersection_index - 1]
        intersection_cone = left_config[left_intersection_index] # valore del cono che è in entrambe le configurazioni

        # calcola distanza dai coni precedenti per entrambe le configurazioni,
        # e lo associa alla configurazione con la distanza minore
        dist_intersection_to_prev_left = np.linalg.norm(
            cones_xy[intersection_cone] - cones_xy[prev_left]
        )
        dist_intersection_to_prev_right = np.linalg.norm(
            cones_xy[intersection_cone] - cones_xy[prev_right]
        )

        low_distance = 3.0
        left_dist_is_very_low = dist_intersection_to_prev_left < low_distance
        right_dist_is_very_low = dist_intersection_to_prev_right < low_distance
        any_distance_very_low = left_dist_is_very_low or right_dist_is_very_low
        both_distances_very_low = left_dist_is_very_low and right_dist_is_very_low

        if any_distance_very_low and not both_distances_very_low:
            if left_dist_is_very_low: # se sinistro p vicino, prendo tutti quelli a sinistra, e stoppo al conf. a destra all'indice passato
                left_stop_idx = len(left_config)
                right_stop_idx = right_intersection_index
            else: # stessa cosa se è a destra
                left_stop_idx = left_intersection_index
                right_stop_idx = len(right_config)
        else:
            left_stop_idx = None
            right_stop_idx = None
    else:
        left_stop_idx = None
        right_stop_idx = None

    if (
        left_stop_idx is None
        and right_stop_idx is None
        and left_config[left_intersection_index]
        == right_config[right_intersection_index]
        and left_intersection_index
        in range(1, len(left_config) - 1)  # not first or last
        and right_intersection_index in range(1, len(right_config) - 1)
    ):
        # intersection happens in the middle of the config
        angle_left = calc_angle_change_at_position(
            cones[:, :2], left_config, left_intersection_index
        )
        angle_right = calc_angle_change_at_position(
            cones[:, :2], right_config, right_intersection_index
        )

        sign_angle_left = np.sign(angle_left)
        sign_angle_right = np.sign(angle_right)

        absolute_angle_diff = abs(abs(angle_left) - abs(angle_right))

        left_has_three = len(left_config) == 3
        right_has_three = len(right_config) == 3

        n_cones_diff = abs(len(left_config) - len(right_config))

        if sign_angle_left == sign_angle_right:
            if sign_angle_left == 1:
                # this is a left corner, prefer the left
                left_stop_idx = len(left_config)
                right_stop_idx = right_intersection_index
            else:
                # this is a right corner, prefer the right
                left_stop_idx = left_intersection_index
                right_stop_idx = len(right_config)
        elif n_cones_diff > 2:
            # if the difference in number of cones is greater than 2, we assume that the
            # longer config is the correct one
            if len(left_config) > len(right_config):
                left_stop_idx = len(left_config)
                right_stop_idx = right_intersection_index
            else:
                left_stop_idx = left_intersection_index
                right_stop_idx = len(right_config)
        elif absolute_angle_diff > np.deg2rad(5):
            if abs(angle_left) > abs(angle_right):
                left_stop_idx = len(left_config)
                right_stop_idx = right_intersection_index
            else:
                left_stop_idx = left_intersection_index
                right_stop_idx = len(right_config)
        else:
            left_stop_idx = left_intersection_index
            right_stop_idx = right_intersection_index
    elif left_stop_idx is None and right_stop_idx is None:
        # if the intersection is the last cone in the config, we assume that this is
        # an error because the configuration could not continue, so we only remove it
        # from that side
        left_intersection_is_at_end = left_intersection_index == len(left_config) - 1
        right_intersection_is_at_end = right_intersection_index == len(right_config) - 1

        if left_intersection_is_at_end and right_intersection_is_at_end:
            left_stop_idx = len(left_config) - 1
            right_stop_idx = len(right_config) - 1

        elif left_intersection_is_at_end:
            right_stop_idx = len(right_config)
            left_stop_idx = left_intersection_index

        elif right_intersection_is_at_end:
            left_stop_idx = len(left_config)
            right_stop_idx = right_intersection_index
        else:
            left_stop_idx = left_intersection_index
            right_stop_idx = right_intersection_index

    return left_stop_idx, right_stop_idx


def calc_angle_change_at_position(
    cones: FloatArray,
    config: IntArray,
    position_in_config: int,
) -> float:
    previous_cone, intersection_cone, next_cone = cones[
        config[position_in_config - 1 : position_in_config + 2], :2
    ]

    intersection_to_next = next_cone - intersection_cone
    intersection_to_prev = previous_cone - intersection_cone

    angle_intersection_to_next = angle_from_2d_vector(intersection_to_next)
    angle_intersection_to_prev = angle_from_2d_vector(intersection_to_prev)

    angle = angle_difference(angle_intersection_to_next, angle_intersection_to_prev)

    return angle
