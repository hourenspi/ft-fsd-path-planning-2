#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: This module provides functionality for sorting a trace of cones into a
plausible track
Project: fsd_path_planning
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

# from fsd_path_planning.cone_matching.functional_cone_matching import \
#     combine_and_sort_virtual_with_real
from fsd_path_planning.sorting_cones.trace_sorter.combine_traces import (
    calc_final_configs_for_left_and_right,
)
from fsd_path_planning.sorting_cones.trace_sorter.common import NoPathError
from fsd_path_planning.sorting_cones.trace_sorter.find_configs_and_scores import (
    calc_scores_and_end_configurations,
)
from fsd_path_planning.types import FloatArray, IntArray
from fsd_path_planning.utils.cone_types import ConeTypes, invert_cone_type
from fsd_path_planning.utils.math_utils import (
    angle_from_2d_vector,
    my_cdist_sq_euclidean,
    points_inside_ellipse,
    rotate,
    vec_angle_between,
)
from fsd_path_planning.utils.utils import Timer

# rivedi bene funzione 
# restituisce un float array
def flatten_cones_by_type_array(cones_by_type: list[FloatArray]) -> FloatArray:
    """Ravel the cones_by_type_array"""

    if (
        isinstance(cones_by_type, np.ndarray)
        and cones_by_type.ndim == 2
        and cones_by_type.shape[1] == 3
    ):
        return cones_by_type

    n_all_cones = sum(map(len, cones_by_type))

    # (x, y, color)
    out = np.empty((n_all_cones, 3)) # inizializza matrice vuota con n_cones_righe e 3 colonne
    n_start = 0 # parte da 0
    # cones by type avrà sempre 5 elementi e l'indice di ogni elementi
    # corrisponde a tipo di coni di quella struttura dati (o=unknown, 1>=right, ecc)
    for cone_type in ConeTypes: # per ogni tipo di cono in ConeTypes
        n_cones = len(cones_by_type[cone_type]) # len ritorna numero di righe 
        # matrice: righe = numero di coni di quel tipo
        
        # dalla riga 0 alla numero coni, prende le prime due colonne
        # e gli associa il tipo dei coni
        # fa reshape: cambia al shape dell'array
        # passo -1 come dimensione sconosciuta
        # shape 0 contiene numero di coni

        # prende l'array unknown, tutti gli elementi e lo trasforma in una matrice a due dim 
        # la cui seconda dimensione ha due colonne, e ? righe 

        # Prende gli (x, y) dei coni di quel tipo e li assegna alle prime due colonne dell'array
        # Assegna il valore cone_type alla terza colonna dell'array per etichettare i coni con il loro tipo.
        out[n_start : n_start + n_cones, :2] = cones_by_type[cone_type].reshape(-1, 2)
        out[n_start : n_start + n_cones, 2] = cone_type
        n_start += n_cones

    return out # ritorna matrice

# confronto tra i due set di coni
def cone_arrays_are_similar(
    cones: FloatArray | None,
    other_cones: FloatArray | None,
    threshold: float,
) -> bool:
    if cones is None or other_cones is None:
        return False

    assert cones.shape[1] == other_cones.shape[1], (cones.shape, other_cones.shape)
    if cones.shape != other_cones.shape:
        return False

    distances = my_cdist_sq_euclidean(cones[:, :2], other_cones[:, :2])

    closest_for_each = distances.min(axis=1)
    distances_all_close = np.all(closest_for_each < (threshold * threshold))

    if cones.shape[1] == 2:
        # in this case we have no color information, we only use distance as indicator
        return distances_all_close

    if not distances_all_close:
        # distances not close no need to check color
        return False

    # last dimension is color
    idx_closest = distances.argmin(axis=1)

    color_match = cones[:, 2] == other_cones[idx_closest, 2]
    return distances_all_close and color_match.all()


# def cones_by_type_are_similar(
#     cones_by_type: list[FloatArray],
#     other_cones_by_type: list[FloatArray],
#     threshold: float,
# ) -> bool:
#     return all(
#         cone_arrays_are_similar(cones, other_cones, threshold)
#         for cones, other_cones in zip(cones_by_type, other_cones_by_type)
#     )


@dataclass
class ConeSortingCacheEntry:
    """
    Dataclass for the cache entry
    """

    input_cones: FloatArray  # x, y, color
    left_starting_cones: FloatArray
    right_starting_cones: FloatArray
    left_result: tuple[Any, ...]
    right_result: tuple[Any, ...]


class TraceSorter:
    """
    Wraps the trace sorting functionality into a class
    """

    def __init__(
        self,
        max_n_neighbors: int,
        max_dist: float,
        max_dist_to_first: float,
        max_length: int,
        threshold_directional_angle: float,
        threshold_absolute_angle: float,
        experimental_caching: bool = False,
    ):
        """
        Constructor for TraceSorter class
        Args:
            max_n_neighbors: The maximum allowed number of neighbors for each node
            during sorting
            max_dist: The maximum allowed distance for two nodes to be
            considered neighbors
            max_dist_to_first: The maximum allowed distance in order for a node
            to be considered a viable first node
        """
        self.max_n_neighbors = max_n_neighbors
        self.max_dist = max_dist
        self.max_dist_to_first = max_dist_to_first
        self.max_length = max_length
        self.threshold_directional_angle = threshold_directional_angle
        self.threshold_absolute_angle = threshold_absolute_angle

        self.cached_results: ConeSortingCacheEntry | None = None
        self.experimental_caching = experimental_caching

    def sort_left_right(
        self,
        cones_by_type: list[FloatArray],
        car_pos: FloatArray,
        car_dir: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        timer_no_print = True
        cones_flat = flatten_cones_by_type_array(cones_by_type)

        # mask_cones_close = (
        #     my_cdist_sq_euclidean(car_pos[None], cones_flat[:, :2])[0] < 25**2
        # )

        # print(mask_cones_close.mean())

        # cones_flat = cones_flat[mask_cones_close]

        with Timer("left config search", timer_no_print):
            ( #costs, configs, history
                left_scores,
                left_configs,
                left_first_cones,
            ) = left_result = self.calc_configurations_with_score_for_one_side(
                cones_flat,
                ConeTypes.LEFT,
                car_pos,
                car_dir,
            )

        with Timer("right config search", timer_no_print):
            (
                right_scores, #costi
                right_configs, # configurazioni
                right_first_cones, #history
            ) = right_result = self.calc_configurations_with_score_for_one_side(
                cones_flat,
                ConeTypes.RIGHT,
                car_pos,
                car_dir,
            )

        self.cached_results = ConeSortingCacheEntry(
            input_cones=cones_flat,
            left_starting_cones=left_first_cones,
            right_starting_cones=right_first_cones,
            left_result=left_result,
            right_result=right_result,
        )

        # step di combinazione dei path sx e dx
        # è in combine_traces

        # trovo le configurazioni disponibili, col minor costo, e senza intersezioni tra le due
        (left_config, right_config) = calc_final_configs_for_left_and_right(
            left_scores,
            left_configs,
            right_scores,
            right_configs,
            cones_flat,
            car_pos,
            car_dir,
        )
        left_config = left_config[left_config != -1]
        right_config = right_config[right_config != -1]


        # remove any placeholder positions if they are present
        left_config = left_config[left_config != -1]
        right_config = right_config[right_config != -1]

        # riduco a matriciona
        left_sorted = cones_flat[left_config]
        right_sorted = cones_flat[right_config]

        """   
    def sort_left_right(
        self,
        cones_by_type: list[FloatArray],
        car_pos: FloatArray,
        car_dir: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        timer_no_print = True
        cones_flat = flatten_cones_by_type_array(cones_by_type)
        """

        # ritorno: tutte le righe (tutte le configurazioni) e le prime due colonne (ovvero, x ed y, alsciando fuori il colore)
        return left_sorted[:, :2], right_sorted[:, :2]

    def input_is_very_similar_to_previous_input(
        self,
        cones_by_type: FloatArray,
        starting_cones: FloatArray,
        threshold: float,
        cone_type: ConeTypes,
    ) -> bool:
        assert cone_type in (ConeTypes.LEFT, ConeTypes.RIGHT)

        # automatically fail if caching is not enabled
        if not self.experimental_caching:
            return False

        if self.cached_results is None:
            return False

        previous_starting_cones = (
            self.cached_results.left_starting_cones
            if cone_type == ConeTypes.LEFT
            else self.cached_results.right_starting_cones
        )

        # first we check if small array is similar
        # cone_arrays_are_similar (core_trace_sorter) 
        # ritorna booleano per il confronto tra i due set di coni
        previous_cones_similar = cone_arrays_are_similar(
            starting_cones, previous_starting_cones, threshold
        )

        # if it not then we need to redo the sorting
        if not previous_cones_similar:
            return False

        # if the small array is similar, we check if the large array is similar
        all_cones_are_similar = cone_arrays_are_similar(
            cones_by_type, self.cached_results.input_cones, threshold
        )
        # if the large array is not similar, we need to redo the sorting
        return all_cones_are_similar


    # chiamato pr coni di sx e poi per i coni di dx
    # 
    # 
    def calc_configurations_with_score_for_one_side(
        self,
        cones: FloatArray,
        cone_type: ConeTypes,
        car_pos: FloatArray,
        car_dir: FloatArray,
    ) -> Tuple[Optional[FloatArray], Optional[IntArray], Optional[FloatArray]]:
        """
        Args:
            cones: The trace to be sorted.
            cone_type: The type of cone to be sorted.
            car_pos: The position of the car.
            car_dir: The direction towards which the car is facing.
        Returns:
            np.ndarray: The sorted trace, `len(return_value) <= len(trace)`
        """
        assert cone_type in (ConeTypes.LEFT, ConeTypes.RIGHT)

        no_result = None, None, None

        if len(cones) < 3:
            return no_result

        # prende le stesse variabili in ordine diverso
        # ritorna l'indeice dei due coni davanti e dietro
        # select first k starting cones richiama la funzione
        # select_starting_cones, che ritorna l'indice del cono di partenza
        first_k = self.select_first_k_starting_cones(
            car_pos,
            car_dir,
            cones,
            cone_type,
        )
        if first_k is not None:
            start_idx = first_k[0]
            if len(first_k) > 1:
                first_k_indices_must_be = first_k.copy()
            else:
                first_k_indices_must_be = None
        else:
            start_idx = None
            first_k_indices_must_be = None

        if start_idx is None and first_k_indices_must_be is None:
            return no_result

        assert first_k is not None

        starting_cones = cones[first_k]

        if self.input_is_very_similar_to_previous_input(
            cones, starting_cones, threshold=0.1, cone_type=cone_type
        ):
            # print("Using cached results")
            cr = self.cached_results
            return cr.left_result if cone_type == ConeTypes.LEFT else cr.right_result

        # print("Calculating new configuration")

        n_neighbors = min(self.max_n_neighbors, len(cones) - 1)
        try:
            result = calc_scores_and_end_configurations(
                cones,
                cone_type,
                n_neighbors,
                start_idx,
                self.threshold_directional_angle,
                self.threshold_absolute_angle,
                car_pos,
                car_dir,
                self.max_dist,
                self.max_length,
                first_k_indices_must_be,
            )

        # if no configurations can be found, then return nothing
        except NoPathError:
            return no_result

        return_value = (*result[:2], starting_cones)

        return return_value

    def invert_cone_type(self, cone_type: ConeTypes) -> ConeTypes:
        """
        Inverts the cone type
        Args:
            cone_type: The cone type to be inverted
        Returns:
            ConeTypes: The inverted cone type
        """
        if cone_type == ConeTypes.LEFT:
            return ConeTypes.RIGHT
        if cone_type == ConeTypes.RIGHT:
            return ConeTypes.LEFT

        raise ValueError(f"Cone type {cone_type} cannot be inverted.")

    # richiamato da calc_configurations_with_score_for_one_side
    def select_starting_cone(
        self,
        car_position: FloatArray,
        car_direction: FloatArray,
        cones: FloatArray,
        cone_type: ConeTypes,
        index_to_skip: Optional[np.ndarray] = None,
    ) -> Optional[int]:
        """
        Return the index of the starting cone
            int: The index of the stating cone
        """

        # mask_cone prende la x e la y di tutti i coni
        # rimuove il tipo di cono
        # all'interno di questo stesso file
        trace_distances, mask_is_valid = self.mask_cone_can_be_first_in_config(
            car_position, car_direction, cones, cone_type
        )
        if index_to_skip is not None:
            mask_is_valid[index_to_skip] = False

        trace_distances_copy = trace_distances.copy()

        # ~ -> unary operator
        # invert or complement operation

        # negazione bitwise -> inverte maschera booleana
        # mette a infinito tutte le distanze che non ci interessano,
        # quelle non valide secondo la maschera mask_is_valid 
        trace_distances_copy[~mask_is_valid] = np.inf

        if np.any(mask_is_valid) > 0:
            # se mask_is_valid count > 0
            # ordinamento secondo distanza
            sorted_idxs = np.argsort(trace_distances_copy)
            # indice start = None (?? perchè)
            start_idx = None

            # ciclo for sul vettore ordinato
            for idx in sorted_idxs:
                if index_to_skip is None or idx not in index_to_skip:
                    start_idx = idx
                    break
            if trace_distances_copy[start_idx] > self.max_dist_to_first:
                start_idx = None
        else:
            start_idx = None

    # ritorna indice iniziale di cono che deve stare a tutti i requisiti
        return start_idx

    # richiamata da select_starting_cones (core_trace_sorter)
    def mask_cone_can_be_first_in_config(
        self, car_position, car_direction, cones, cone_type
    ):
    # car_direction: vettore 2D
    # cones: array nx3 [x, y, tipo]

        cones_xy = cones[:, :2]  # remove cone type

        # cambia sis. rif: dal sistema di riferimento globale a quello della mappa
        # matrice va da nx3 a nx2
        # mediante la funzione di rotate (su math_utils.py)
        # vogliamo mettere tutto a 0 gradi, perciò durabte il cambiamento anche i
        # cones_xy - car_position: sposta i coni rispetto alla pos. dell'auto
        # angle: angolo rispetto all'asse x
        # rotate: ruota tutti i coni per allineare la posizione dell'auto con l'asse x.
        cones_relative = rotate(
            cones_xy - car_position, -angle_from_2d_vector(car_direction)
        )
        # angle from 2d vector: si trova in math_utils.py
        # calcola l'angolo di ogni vettore in vecs.

        # calcolo angoli di ogni cono rispetto all'auto
        cone_relative_angles = angle_from_2d_vector(cones_relative)

        # calcolo distanze coni - auto
        trace_distances = np.linalg.norm(cones_relative, axis=-1)

        # ellisse per selezionare solo coni vicini
        # su math_utils.py
        mask_is_in_ellipse = points_inside_ellipse(
            cones_xy,
            car_position,
            car_direction,
            self.max_dist_to_first * 1.5,
            self.max_dist_to_first / 1.5,
        )
        # Determiniamo se i coni sono a sinistra o destra rispetto all'auto
        # coni sinistri -> angoli positivi
        angle_signs = np.sign(cone_relative_angles)
        valid_angle_sign = 1 if cone_type == ConeTypes.LEFT else -1
        mask_valid_side = angle_signs == valid_angle_sign

        # escludiamo coni troppo lontani o del tipo errato
        mask_is_valid_angle = np.abs(cone_relative_angles) < np.pi - np.pi / 5
        mask_is_valid_angle_min = np.abs(cone_relative_angles) > np.pi / 10
        mask_is_right_color = cones[:, 2] == cone_type

        # creazione della maschera: 
        # Un cono è valido se sul lato giusto, angolo valido, tipo giusto.
        mask_side = (
            mask_valid_side * mask_is_valid_angle * mask_is_valid_angle_min
        ) + mask_is_right_color

        mask_is_not_opposite_cone_type = cones[:, 2] != invert_cone_type(cone_type)
        mask_is_valid = mask_is_in_ellipse * mask_side * mask_is_not_opposite_cone_type
    # trace_distances: vettore con distanze di ogni cono rispetto all'auto
    # mask_is_valid: boolean che indica quali coni possono essere primo punto valido
        return trace_distances, mask_is_valid

    def select_first_k_starting_cones(
        self,
        car_position: FloatArray, 
        car_direction: FloatArray,
        cones: FloatArray,
        cone_type: ConeTypes,
    ) -> Optional[np.ndarray]:
        # ritorna l'indice dei coni iniziali. 
        """
        Return the index of the starting cones. Pick the cone that is closest in front
        of the car and the cone that is closest behind the car.
        """

        # select_starting_cone: ritorna l'indice del cono di partenza
        # (core_trace_sorter)
        index_1 = self.select_starting_cone(
            car_position,
            car_direction,
            cones,
            cone_type,
        )

        if index_1 is None:
            return None
        # posizione dei coni [x, y] rispetto alla macchina
        cones_to_car = cones[:, :2] - car_position

        # angolo coni - auto tra due vettori
        # (math_utils.py)
        # ritorna un vettore in cui ogni elemento contiene l'angolo tra i vettore v1[i] e v2[i]
        angle_to_car = vec_angle_between(cones_to_car, car_direction)

        # non va bene se l'angolo cono - macchina è minore di 90 gradi
        # faccio indexs_to_skip per rpendere i coni dietro
        # in particolare il primo dietro
        mask_should_not_be_selected = np.abs(angle_to_car) < np.pi / 2
        idxs_to_skip = np.where(mask_should_not_be_selected)[0]
        if index_1 not in idxs_to_skip:
            idxs_to_skip = np.concatenate([idxs_to_skip, np.array([index_1])])

        # get the cone behind the car
        index_2 = self.select_starting_cone(
            car_position,
            car_direction,
            cones,
            cone_type,
            index_to_skip=idxs_to_skip,
        )

    # se non trova il primo cono posteriore restituisce solo il primo
        if index_2 is None:
            return np.array([index_1], dtype=np.int_)

        # vettore distanza va da ?
        cone_dir_1 = cones[index_1, :2] - cones[index_2, :2]
        cone_dir_2 = cones[index_2, :2] - cones[index_1, :2]

        # angle_1 : coerente con direzione dell'auto essendo cono anteriore
        angle_1 = vec_angle_between(cone_dir_1, car_direction)
        angle_2 = vec_angle_between(cone_dir_2, car_direction)

    # controllo che la distanza sia accettabile prima di restituirli
    # cond. comuni: angle_1 < angle_2
        if angle_1 > angle_2:
            index_1, index_2 = index_2, index_1

        # norm = distanza euclidea tra i due coni
        dist = np.linalg.norm(cone_dir_1)
        if dist > self.max_dist * 1.1 or dist < 1.4:
            return np.array([index_1], dtype=np.int_)

        # array con index2, index1 di tipo int
        two_cones = np.array([index_2, index_1], dtype=np.int_)
     
        return two_cones
