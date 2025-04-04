#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: This File calculates all the possible paths
Project: fsd_path_planning
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from fsd_path_planning.sorting_cones.trace_sorter.common import NoPathError
from fsd_path_planning.sorting_cones.trace_sorter.line_segment_intersection import (
    cast,
    lines_segments_intersect_indicator,
)
from fsd_path_planning.types import BoolArray, FloatArray, GenericArray, IntArray
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import (
    my_in1d,
    my_njit,
    points_inside_ellipse,
    vec_angle_between,
)

# my_njit = lambda x: x  # XXX: just for debugging

# rivedere meglio algoritmo implementativo
@my_njit
def adjacency_matrix_to_borders_and_targets(
    adjacency_matrix: IntArray,
) -> Tuple[IntArray, IntArray]:
    # converte la matrice di adiacenza in 2 array flat 
    """
    Convert an adjacency matrix to two flat arrays, one representing the neighbors of
    each node and one which indicates the starting index of each node in the neighbors
    array
    [
        [0 1 0]
        [1 1 1]
        [1 0 0]
    ]
    is converted to
    neighbors -> [1, 0, 1, 2, 0]
    borders -> [0, 1, 4, 5]
    Args:
        adjacency_matrix: The adjacency matrix to convert
    Returns:
        The neighbors and the starting position of each node in the neighbors array
    """
    source, neighbors_flat = np.where(adjacency_matrix)

    borders: IntArray = np.zeros((adjacency_matrix.shape[0], 2), dtype=np.int32) - 1
    borders[0] = [0, 0]

    for index in range(len(neighbors_flat)):
        source_idx = source[index]

        if borders[source_idx][0] == -1:
            borders[source_idx][0] = index

        borders[source_idx][1] = index + 1

    # for all the nodes that have no neighbors set their start and end to the
    # previous node's end
    for i, value in enumerate(borders):
        if value[0] == -1:
            previous = borders[i - 1][1]
            borders[i] = [previous, previous]

    final_borders = np.concatenate((borders[:, 0], borders[-1:, -1]))

    return neighbors_flat, final_borders


@my_njit
def double_stack_len(stack: GenericArray) -> GenericArray:
    """
    Double the capacity of a stack (used in the path search)
    Args:
        stack: The stack whose size will be doubled
    Returns:
        A copy of the stack with double the capacity
    """
    _len = stack.shape[0]
    new_shape = (_len * 2, *stack.shape[1:])
    new_buffer = np.full(new_shape, -1, dtype=stack.dtype)

    new_buffer[:_len] = stack
    return new_buffer


@my_njit
def resize_stack_if_needed(stack: GenericArray, stack_pointer: int) -> GenericArray:
    """
    Given a stack and its current pointer resize the stack if adding one more element
    would result in an index error
    Args:
        stack: The stack to potentially resize
        stack_pointer: The current position (filled size) of the stack
    Returns:
        np.ndarray: The stack, if resized then a copy is returned
    """
    if stack_pointer >= stack.shape[0]:
        stack = double_stack_len(stack)

    return stack


# for numba
FLOAT = float if TYPE_CHECKING else np.float32

# chiamata da _impl_find_all_end_configurations
# restituisce un array di booleans
@my_njit
def neighbor_bool_mask_can_be_added_to_attempt(
    trace: FloatArray,
    cone_type: ConeTypes,
    current_attempt: IntArray,
    position_in_stack: int,
    neighbors: IntArray,
    threshold_directional_angle: float,
    threshold_absolute_angle: float,
    car_position: FloatArray,
    car_direction: FloatArray,
    car_size: float,
) -> BoolArray:
    # TODO: this function is too long, split it up
    # print(locals())
    # normalizza la direzione della macchina
    car_direction_normalized = car_direction / np.linalg.norm(car_direction)

    # neighbor can be added if not in current 
    #~ : complementare
    # my_in1d (math_utils.py)
    # can_be_added = array di booleani
    can_be_added = ~my_in1d(neighbors, current_attempt[: position_in_stack + 1])

    # prende dalla path i punti che corrispondono ai valori vicini (ai neighbors)
    neighbors_points = trace[neighbors]
    if position_in_stack >= 1:
        mask_in_ellipse = calculate_mask_within_ellipse(
            trace, current_attempt, position_in_stack, neighbors_points
        )

        # definisce maschera
        can_be_added = can_be_added & mask_in_ellipse

    if position_in_stack == 0:
        # the second cone in the attempt should be on the expected side
        # of the car (left cone on the left side of the car, right cone on the right)
        mask_second_cone_right_side = mask_second_in_attempt_is_on_right_vehicle_side(
            cone_type, car_position, car_direction_normalized, neighbors_points
        )

        can_be_added = can_be_added & mask_second_cone_right_side

    # range: crea array di numeri che va da 0 a len(can_be_added)
    for i in range(len(can_be_added)):
        if not can_be_added[i]:
            continue # used to end the current iteration in a for loop

        # aggiungi neighbor a candidato 
        candidate_neighbor = neighbors[i]

        # find if there is a cone that is between the last cone in the attempt
        # and the candidate neighbor, if so we do not want to pursue this path, because
        # it will skip one cone
        check_if_neighbor_lies_between_last_in_attempt_and_candidate(
            trace,
            current_attempt,
            position_in_stack,
            neighbors,
            can_be_added,
            i,
            candidate_neighbor,
        )

        # prende posizione del candidato all'interno della traccia
        candidate_neighbor_pos = trace[neighbors[i]]

        # calculate angle between second to last to last vector in attempt
        # and the vector between the last node and the candidate neighbor
        # add to current attempt only if the angle between the current last
        # vector and the potential new last vector is less than a specific
        # threshold. there are two thresholds, one is the maximum angle in a specific direction
        # for blue cones that is counter-clockwise and for yellow cones that is clockwise
        # the second threshold is an absolute angle between the two vectors.
        # XXX: There might be a bug where the can_be_added[i] is set to false and then
        # back to true

        # position_in_stack rappresenta la posizione attuale nella sequenza di coni costruita.
        # se è almeno 1, significa che c'è almeno un segmento precedente con cui confrontare l'angolo.

        if can_be_added[i] and position_in_stack >= 1:
            second_to_last_in_attempt = trace[current_attempt[position_in_stack - 1]]
            last_in_attempt = trace[current_attempt[position_in_stack]]
            # Calcolo del vettore tra i due ultimi coni
            second_to_last_to_last = last_in_attempt - second_to_last_in_attempt

            # Calcolo del vettore tra ultimo cono e candidato
            last_to_candidate = candidate_neighbor_pos - last_in_attempt

            # calcolo degli angoli: angle_1 tra ultimi 2, angle_2 tra ultimo cono e candidato
            angle_1 = cast(
                FLOAT,
                np.arctan2(second_to_last_to_last[1], second_to_last_to_last[0]),
            )
            angle_2 = cast(
                FLOAT,
                np.arctan2(last_to_candidate[1], last_to_candidate[0]),
            )
            # order is important here [n.a.]
            # differenza tra i due angoli: aiuta a capire se l'aggiunta del cono fa cambiare bruscamente direzione
            difference = angle_difference(angle_2, angle_1)
            len_last_to_candidate = np.linalg.norm(last_to_candidate)

            # controllo condizioni su angolo
            if np.abs(difference) > threshold_absolute_angle:
                can_be_added[i] = False
            elif cone_type == ConeTypes.LEFT:
                # SX: Viene accettato se l'angolo è inferiore alla soglia o se è molto vicino (< 4 metri).
                can_be_added[i] = (
                    difference < threshold_directional_angle
                    or len_last_to_candidate < 4.0
                )
            elif cone_type == ConeTypes.RIGHT:
                # Simile, ma accetta solo angoli superiori a -threshold_directional_angle.
                can_be_added[i] = (
                    difference > -threshold_directional_angle
                    or len_last_to_candidate < 4.0
                )
            else:
                raise AssertionError("Unreachable code")

            # check if candidate causes change in direction in attempt
            if position_in_stack >= 2:
                third_to_last = trace[current_attempt[position_in_stack - 2]]
                third_to_last_to_second_to_last = (
                    second_to_last_in_attempt - third_to_last
                )
                angle_3 = cast(
                    FLOAT,
                    np.arctan2(
                        third_to_last_to_second_to_last[1],
                        third_to_last_to_second_to_last[0],
                    ),
                )
                difference_2 = angle_difference(angle_1, angle_3)

                if (
                    np.sign(difference) != np.sign(difference_2)
                    and np.abs(difference - difference_2) > 1.3
                ):
                    can_be_added[i] = False

                # Se ci sono almeno tre coni nella sequenza, confrontiamo anche l'angolo del terzo cono rispetto al secondo.
                # Se il nuovo angolo cambia segno rispetto al precedente (da positivo a negativo o viceversa) 
                # e la differenza è > 1.3 radianti (~75°), il candidato non può essere aggiunto.

        # Controllo sulla direzione iniziale: 
        # Se siamo al primo cono della sequenza:
        # Controlliamo che la direzione del nuovo segmento non si discosti troppo dalla direzione iniziale della macchina (car_direction).
        # Se l'angolo tra car_direction e diff è maggiore di 90°, il cono non può essere aggiunto.

        if can_be_added[i] and position_in_stack == 1:
            start = trace[current_attempt[0]]
            diff = candidate_neighbor_pos - start
            direction_offset = vec_angle_between(car_direction, diff)
            can_be_added[i] &= direction_offset < np.pi / 2

        # controllo collisione con macchina
        if can_be_added[i] and position_in_stack >= 0:
            # make sure that no intersection with car occurs
            last_in_attempt = trace[current_attempt[position_in_stack]]
            car_start = car_position - car_direction_normalized * car_size / 2
            car_end = car_position + car_direction_normalized * car_size

            can_be_added[i] &= not lines_segments_intersect_indicator(
                last_in_attempt, candidate_neighbor_pos, car_start, car_end
            )

    return can_be_added # array booleani

# chiamato da neighbor_bool_mask_can_be_added_to_attempt
# controlla se c'è un cono tra il cono corrente e il vicino candidato
# nel caso ci fosse, la path current-candidate non viene salvata come ottimale
@my_njit
def check_if_neighbor_lies_between_last_in_attempt_and_candidate(
    trace: FloatArray, # insieme di coni
    current_attempt: IntArray,
    position_in_stack: int,
    neighbors: IntArray,
    can_be_added: BoolArray,
    i: int,
    candidate_neighbor: int,
):
    for neighbor in neighbors:
        if neighbor == neighbors[i]:
            continue

        # vettore tra il cono corrente e il neighbor che non corrisponde al neighbors[i]
        neighbor_to_last_in_attempt = (
            trace[current_attempt[position_in_stack]] - trace[neighbor]
        )

        # vettore tra il neighbor e il candidate neighbor originale
        neighbor_to_candidate = trace[candidate_neighbor] - trace[neighbor]

        # calcolo distanza euclidea
        dist_to_candidate = np.linalg.norm(neighbor_to_candidate)
        dist_to_last_in_attempt = np.linalg.norm(neighbor_to_last_in_attempt)

        # if the angle between the two vectors is more than 150 degrees
        # then the neighbor cone is between the last cone in the attempt
        # and the candidate neighbor. we can't add the candidate neighbor
        # to the attempt
        if (
            # control threshold
            dist_to_candidate < 6.0
            and dist_to_last_in_attempt < 6.0
            and vec_angle_between(neighbor_to_last_in_attempt, neighbor_to_candidate)
            > np.deg2rad(150)
        ):
            can_be_added[i] = False
            break

# chiamata da neighbor_can_be_added_to_attemps
# ritorna array di booleani
@my_njit
def mask_second_in_attempt_is_on_right_vehicle_side(
    cone_type: ConeTypes,
    car_position: FloatArray,
    car_direction_normalized: FloatArray,
    neighbors_points: FloatArray,
) -> BoolArray:

    # calcola posizione della macchina rispetto ai punti vicini al punto corrente
    car_to_neighbors = neighbors_points - car_position
    # direzione dell'auto
    angle_car_dir = np.arctan2(car_direction_normalized[1], car_direction_normalized[0])
    # angolo tra macchina e vicini
    angle_car_to_neighbors = np.arctan2(car_to_neighbors[:, 1], car_to_neighbors[:, 0])

    # differenza di angolo tra la direzione normalizzata ai coni, e la direzione dell'auto 
    angle_diff = angle_difference(angle_car_to_neighbors, angle_car_dir)

    expected_sign = 1 if cone_type == ConeTypes.LEFT else -1
    # tira fuori il segno
    mask_expected_side = np.sign(angle_diff) == expected_sign

    # 
    mask_other_side_tolerance = np.abs(angle_diff) < np.deg2rad(5)

    mask = mask_expected_side | mask_other_side_tolerance
    return mask

# chiamato da neighbor_bool_mask_can_be_added_to_attempt
#ritorna un array booleano che indica se ogni punto in neighbors_points è dentro l'ellisse.
@my_njit
def calculate_mask_within_ellipse(
    trace: FloatArray,
    current_attempt: IntArray,
    position_in_stack: int,
    neighbors_points: FloatArray,
) -> BoolArray:
    # last_in_attempt: è l'ultimo punto del percorso corrente
    last_in_attempt = trace[current_attempt[position_in_stack]]
    second_to_last_in_attempt = trace[current_attempt[position_in_stack - 1]]

    # si calcola il vettore che va dal penultimo all'ultimo punto,
    # usato come ASSE MAGGIORE dell'ellisse
    second_to_last_to_last = last_in_attempt - second_to_last_in_attempt

    # points_inside_ellipse (math_utils.py)
    # controlla se i tre coni sono all'interno di un ellisse definita da due punti 
    # di riferimento definiti in trace
    mask_in_ellipse = points_inside_ellipse(
        neighbors_points,
        last_in_attempt, # Centro nell'ultimo punto del percorso
        major_direction=second_to_last_to_last, #asse maggiore in direzione del vettore
        major_radius=6,
        minor_radius=3,
    )

    return mask_in_ellipse


@my_njit
def angle_difference(angle1: FloatArray, angle2: FloatArray) -> FloatArray:
    """
    Calculate the difference between two angles. The range of the difference is [-pi, pi].
    The order of the angles *is* important.

    Args:
        angle1: First angle.
        angle2: Second angle.

    Returns:
        The difference between the two angles.
    """
    return cast(FloatArray, (angle1 - angle2 + 3 * np.pi) % (2 * np.pi) - np.pi)  # type: ignore

# ritorna un array 2d di indice che definiscono tutti i path validi.
# cerca tutti i percorsi validi a partire da un nodo iniziale,
# seguendo vincoli di angoli e posizione
# fino a target_length
@my_njit
def _impl_find_all_end_configurations(
    trace: FloatArray,
    cone_type: ConeTypes,
    start_idx: int,
    adjacency_neighbors: IntArray,
    adjacency_borders: IntArray,
    target_length: int,
    threshold_directional_angle: float,
    threshold_absolute_angle: float,
    first_k_indices_must_be: IntArray,
    car_position: FloatArray,
    car_direction: FloatArray,
    car_size: float,
    store_all_end_configurations: bool,
) -> tuple[IntArray, Optional[tuple[IntArray, BoolArray]]]:
    """
    Finds all the possible paths up to length target length. If a path
    # se una path
    Args:
        start_idx: The index of the starting node
        # trovate con adjacency_matrix_to_border_and_targets
        adjacency_neighbors: The indices of the sink for each edge
        adjacency_borders: The start position of the indices of each node
        target_length: The length of the path that the search is searching for
    Returns:
        A 2d array of configurations (indices) that define 
        all the valid paths
    """

    # creazione array di int riempito con -1, di dim [10xtarget_length]
    end_configurations: IntArray = np.full((10, target_length), -1, dtype=np.int32)

    # creazione di un pointer per navigare
    end_configurations_pointer = 0

    # stack tiene conto di [nodo, posizione] per esplorare il grafo
    # creazione di stack di dimensione [10x2] e pointer all'elemento finale per navigare lo stack
    stack: IntArray = np.zeros((10, 2), dtype=np.int32)
    stack_end_pointer = 0

    # array di interi inizializzato a -1 di dimensione target_length
    current_attempt: IntArray = np.zeros(target_length, dtype=np.int32) - 
    
    # se first_k_indices_must_be è non vuoto
    # (insieme di indici che devono essere forzati come primi elementi di tentativo iniziale)
    if len(first_k_indices_must_be) > 0:
        pos = len(first_k_indices_must_be) - 1
        current_attempt[:pos] = first_k_indices_must_be[:-1]
        # :-1 sta per: prendi tutto tranne l'ultimo
        # imposta lo stack al suo utlime valore
        # primo elemento di stack sarà l'ultimo el di fkimb a posizione fkimb-1
        stack[0] = [first_k_indices_must_be[-1], pos]
    else:
        # altrimenti parte da start index
        stack[0] = [start_idx, 0]

    # se vuoi salvare tutte le configurazioni (boolean)
    if store_all_end_configurations:
        all_configurations_counter = 0
        all_configurations = end_configurations.copy()

        # crei nuovo array di boolean di ddimensione righe di end_configurations
        configuration_is_end = np.zeros(end_configurations.shape[0], dtype=np.bool_)

    while stack_end_pointer >= 0:
        # pop the index and the position from the stack
        # estrae il nodo dallo stack

        # a ritroso
        next_idx, position_in_stack = stack[stack_end_pointer]
        stack_end_pointer -= 1

        # add the node to the current path
        # salva il nodo nel eprcorso corrente
        current_attempt[position_in_stack] = next_idx

        # azzera le posizioni successive
        current_attempt[position_in_stack + 1 :] = -1

        # get the neighbors of the last node in the attempt
        # Estrae i vicini del nodo next_idx.
        neighbors = adjacency_neighbors[
            adjacency_borders[next_idx] : adjacency_borders[next_idx + 1]
        ]

        # definizione maschera booleana x ogni cono
        can_be_added = neighbor_bool_mask_can_be_added_to_attempt(
            trace,
            cone_type,
            current_attempt,
            position_in_stack,
            neighbors,
            threshold_directional_angle,
            threshold_absolute_angle,
            car_position,
            car_direction,
            car_size,
        )

        has_valid_neighbors = position_in_stack < target_length - 1 and np.any(
            can_be_added
        )
        # check that we haven't hit target length and that we have neighbors to add
        if has_valid_neighbors:
            for i in range(len(can_be_added)): # [0-can_be_added-1]
                if not can_be_added[i]:
                    continue

                # scorre pointer
                stack_end_pointer += 1

                # aggiunta cono vicino e incremento posizione
                stack = resize_stack_if_needed(stack, stack_end_pointer)
                #     Given a stack and its current pointer resize the stack if adding one more element would result in an index error
                stack[stack_end_pointer] = [
                    neighbors[i],
                    position_in_stack + 1,
                ]

        # leaf
        else:
            end_configurations = resize_stack_if_needed(
                end_configurations, end_configurations_pointer
            )

            end_configurations[end_configurations_pointer:] = current_attempt.copy()

            end_configurations_pointer += 1

        if store_all_end_configurations:
            all_configurations = resize_stack_if_needed(
                all_configurations, all_configurations_counter
            )
            configuration_is_end = resize_stack_if_needed(
                configuration_is_end, all_configurations_counter
            )
            all_configurations[all_configurations_counter] = current_attempt
            configuration_is_end[all_configurations_counter] = not has_valid_neighbors
            all_configurations_counter += 1

    # creazione array
    return_value_end_configurations: IntArray = end_configurations[
        :end_configurations_pointer
    ]

    mask_end_configurations_with_more_that_two_nodes = (
        return_value_end_configurations != -1
    ).sum(axis=1) > 2
    return_value_end_configurations = return_value_end_configurations[
        mask_end_configurations_with_more_that_two_nodes
    ]

    if store_all_end_configurations:
        all_configurations = all_configurations[:all_configurations_counter]
        configuration_is_end = configuration_is_end[:all_configurations_counter]

    config_history = None
    if store_all_end_configurations:
        config_history = (all_configurations, configuration_is_end)

    return return_value_end_configurations, config_history #config_history solo se store_all_end_configurations è TRUE


def find_all_end_configurations(
    points: FloatArray,
    cone_type: ConeTypes,
    start_idx: int,
    adjacency_matrix: IntArray,
    target_length: int,
    threshold_directional_angle: float,
    threshold_absolute_angle: float,
    first_k_indices_must_be: IntArray,
    car_position: FloatArray,
    car_direction: FloatArray,
    car_size: float,
    store_all_end_configurations: bool,
) -> tuple[IntArray, Optional[tuple[IntArray, BoolArray]]]:
    """
    Finds all the possible paths that include all the reachable nodes from the starting
    Args:
        start_idx: The index of the starting node
        adjacency_matrix: The adjacency matrix indicating which nodes are
        connected to which
        target_length: The length of the path that the search is searching for
    Raises:
        NoPathError: If no path has been found
    Returns:
        A 2d array of configurations (indices) that define a path
    """
    # print(locals())

    # adjacency_matrix_to_borders_and_targets
    # (end_configurations.py)
    # ritorna i vicini per ogni starting point, e la posizione iniziale di ogni nodo nell'array dei punti più vicini
    neighbors_flat, borders = adjacency_matrix_to_borders_and_targets(adjacency_matrix)

    # array dei punti passato come parametro
    # viene tolto il colore
    points_xy = points[:, :2]

    # creazione di end_configurations e dell'indicatore
    (
        end_configurations,
        all_configurations_and_is_end_configuration_indicator,
    ) = _impl_find_all_end_configurations(
        points_xy,
        cone_type,
        start_idx,
        neighbors_flat,
        borders,
        target_length,
        threshold_directional_angle,
        threshold_absolute_angle,
        first_k_indices_must_be,
        car_position,
        car_direction,
        car_size,
        store_all_end_configurations,
    )
    # impl_find_all_end_configurations
    # (end_configurations.py)
    # ritorna tutti i possibili path fino alla lunghezza target_length
    # passata come argomento


    # filtra end_configurations mantenendo solo quelle i cui primi k valori corrispondono esattamente a first_k_indices_must_be
    if len(first_k_indices_must_be) > 0 and len(end_configurations) > 0:
        mask_keep = (
            end_configurations[:, : len(first_k_indices_must_be)]
            == first_k_indices_must_be
        ).all(axis=1)
        end_configurations = end_configurations[mask_keep]

    # configurazione mantenuta solo se ci sono almeno 3 coni:
    mask_length_is_atleast_3 = (end_configurations != -1).sum(axis=1) >= 3
    end_configurations = end_configurations[mask_length_is_atleast_3]

    # remove last cone from config if it is of unknown or orange type
    last_cone_in_each_config_idx = (
        np.argmax(end_configurations == -1, axis=1) - 1
    ) % end_configurations.shape[1]

    # Trova il primo -1 in ogni configurazione (indice del primo spazio vuoto).
    # Sottrae 1 per ottenere l'indice dell'ultimo cono valido in ogni configurazione.

    last_cone_in_each_config = end_configurations[
        np.arange(end_configurations.shape[0]), last_cone_in_each_config_idx
    ]

    # rimuove coni non validi: 
    mask_last_cone_is_not_of_type = points[last_cone_in_each_config, 2] != cone_type

    last_cone_in_each_config_idx_masked = last_cone_in_each_config_idx[
        mask_last_cone_is_not_of_type
    ]

    end_configurations[
        mask_last_cone_is_not_of_type, last_cone_in_each_config_idx_masked
    ] = -1

    # nuovo filtraggio 3 coni: 
    # keep only configs with at least 3 cones
    mask_length_is_atleast_3 = (end_configurations != -1).sum(axis=1) >= 3
    end_configurations = end_configurations[mask_length_is_atleast_3]

    # remove identical configs
    end_configurations = np.unique(end_configurations, axis=0)
    # remove subsets
    are_equal_mask = end_configurations[:, None] == end_configurations
    are_minus_1_mask = end_configurations == -1
    are_equal_mask = are_equal_mask | are_minus_1_mask

    is_duplicate = are_equal_mask.all(axis=-1).sum(axis=0) > 1

    end_configurations = end_configurations[~is_duplicate]

    if len(end_configurations) == 0:
        raise NoPathError("Could not create a valid trace using the provided points")

    # ritorna solo le configurazioni corrette e l'indicatore 
    return end_configurations, all_configurations_and_is_end_configuration_indicator
