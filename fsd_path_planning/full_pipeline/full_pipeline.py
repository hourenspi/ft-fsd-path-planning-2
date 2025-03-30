#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: A class that runs the whole path planning pipeline.

- Cone sorting
- Cone Matching
- Path Calculation

Project: fsd_path_planning
"""
from __future__ import annotations

from typing import Any, List, Optional, Union

import numpy as np

from fsd_path_planning.calculate_path.core_calculate_path import PathCalculationInput
from fsd_path_planning.cone_matching.core_cone_matching import ConeMatchingInput
from fsd_path_planning.config import (
    create_default_cone_matching_with_non_monotonic_matches,
    create_default_pathing,
    create_default_sorting,
)
from fsd_path_planning.relocalization.acceleration.acceleration_relocalization import (
    AccelerationRelocalizer,
)
from fsd_path_planning.relocalization.relocalization_base_class import Relocalizer
from fsd_path_planning.relocalization.relocalization_information import (
    RelocalizationInformation,
)
from fsd_path_planning.relocalization.skidpad.skidpad_path_data import BASE_SKIDPAD_PATH
from fsd_path_planning.relocalization.skidpad.skidpad_relocalizer import (
    SkidpadRelocalizer,
)
from fsd_path_planning.sorting_cones.core_cone_sorting import ConeSortingInput
from fsd_path_planning.types import FloatArray, IntArray
from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import (
    angle_from_2d_vector,
    unit_2d_vector_from_angle,
)
from fsd_path_planning.utils.mission_types import MissionTypes
from fsd_path_planning.utils.utils import Timer

# mappa missioni - localizer specifici 
MissionToRelocalizer: dict[MissionTypes, Relocalizer] = {
    MissionTypes.acceleration: AccelerationRelocalizer,
    MissionTypes.ebs_test: AccelerationRelocalizer,
    MissionTypes.skidpad: SkidpadRelocalizer,
}


class PathPlanner:
    def __init__(
        self, mission: MissionTypes, experimental_performance_improvements: bool = False
    ) -> None:
        self.mission = mission

        # creazione oggetto di classe Relocalizer, con vettori posizione e direzione
        self.relocalizer: Relocalizer | None = None
        # mission to relocalizer prende oggetto localizer dalla mappa di tutti in base alla missione
        # e lo assegna a relocalizer_class
        relocalizer_class = MissionToRelocalizer.get(mission)

        # se non è nullo, associa quel relocalizer all'oggetto corrente 
        if relocalizer_class is not None:
            self.relocalizer = relocalizer_class()

        # chiama funzione che crea oggetto conesorting su oggetto corrente pathplanner
        self.cone_sorting = create_default_sorting(
            mission, experimental_performance_improvements
        )

        self.cone_matching = create_default_cone_matching_with_non_monotonic_matches(mission)
        self.pathing = create_default_pathing(mission)
        self.global_path: Optional[FloatArray] = None

        self.experimental_performance_improvements = (experimental_performance_improvements)

    # funzione che converte vettore direzione in array
    def _convert_direction_to_array(self, direction: Any) -> FloatArray:
        direction = np.squeeze(np.array(direction))
        if direction.shape == (2,):
            return direction

        if direction.shape in [(1,), ()]:
            return unit_2d_vector_from_angle(direction)

        raise ValueError("direction must be a float or a 2 element array")

    # assegnamento del path all'oggetto corrente
    def set_global_path(self, global_path):
        self.global_path = global_path


    def calculate_path_in_global_frame(
        self,
        cones: List[FloatArray], #lista dell'insieme di punti dei coni
        vehicle_position: FloatArray,
        vehicle_direction: Union[FloatArray, float], # potrei ricevere in input o un floatarray o un float
        return_intermediate_results: bool = False,
    ) -> Union[ # in output potrebbe ritornare un float array o un array di tuple con campi di questi tipi:
        FloatArray,
        tuple[
            FloatArray,
            FloatArray,
            FloatArray,
            FloatArray,
            FloatArray,
            IntArray,
            IntArray,
        ],
    ]:
        """
        Runs the whole path planning pipeline.

        Args:
            cones: List of cones in global frame. Position of Nx2 arrays based on
            `ConeTypes`.
            vehicle_position: Vehicle position in global frame. 2 element array (x,y).
            vehicle_direction: Vehicle direction in global frame. 2 element array
            (dir_x,dir_y).
            return_intermediate_results: If True, returns intermediate results (sorting)
            and matching).

        Returns:
            A Nx4 array of waypoints in global frame. Each waypoint is a 4 element array
            (spline_parameter, path_x, path_y, curvature).
        """
        # traduzione del vettore direzione in array
        vehicle_direction = self._convert_direction_to_array(vehicle_direction)

        noprint = True

        # relocalizzazione non è nostra facoltà poichè reactive algorithm
        # ci arriva posizione dei coni rispetto a veicolo e posizione veicolo data da slam

        # relocalizer non nullo: inizializzazione già avvenuta, non è il primo giro(?)
        # tenta di correggere la posiione del veicolo rispetto a mappa nota:
        if self.relocalizer is not None:
            # si cerca di relocalizzare, classe Timer definita all'interno di utils.py
            with Timer("Relocalization", noprint=noprint):
                self.relocalizer.attempt_relocalization_calculation(
                    cones, vehicle_position, vehicle_direction
                )
            # se è stato relocalizzato con successo:
            if self.relocalizer.is_relocalized:
                # converte la posizione e direzione del veicolo nel sistema di riferimento della mappa nota
                vehicle_yaw = angle_from_2d_vector(vehicle_direction)
                (
                    vehicle_position,
                    vehicle_yaw,
                ) = self.relocalizer.transform_to_known_map_frame(
                    vehicle_position, vehicle_yaw
                )
                vehicle_direction = unit_2d_vector_from_angle(vehicle_yaw)
                # ottiene percorso globale definitio dalla mappa nota
                self.global_path = self.relocalizer.get_known_global_path()

                # print(vehicle_position, vehicle_yaw)

            sorted_left, sorted_right = np.zeros((2, 0, 2))
            left_cones_with_virtual, right_cones_with_virtual = np.zeros((2, 0, 2))
            left_to_right_match, right_to_left_match = np.zeros((2, 0), dtype=int)

    # algo per primo giro parte da qui:

        else:
            # run cone sorting
            # all'interno della classe ConeSorting utilizzato un TraceSorter, in cone_trace_sorter, 
            # prende input dal config 
            with Timer("Cone sorting", noprint=noprint):
                # ConeSortingInput classe itnerna a core_cone_sorting.py
                # 
                cone_sorting_input = ConeSortingInput(
                    cones, vehicle_position, vehicle_direction
                )
                # input: coni, posizione e direzione dallo slam
                self.cone_sorting.set_new_input(cone_sorting_input)
                # run_cone_sorting:
                # chiama transition_input_state che prende le variabili slam position e direction dallo state
                # mette gli unknown a zero
                # chiama sort_left_right
                # 
                """
                Calculate the sorted cones.
                Returns:
                    The sorted cones. The first array contains the sorted blue (left) cones and
                    the second array contains the sorted yellow (right) cones.
                """
                sorted_left, sorted_right = self.cone_sorting.run_cone_sorting()

            # run cone matching
            with Timer("Cone matching", noprint=noprint):
                """
                np.zeros((0, 2)): Creates an empty NumPy array with shape (0, 2), which means it has zero rows and two columns.
                [... for _ in ConeTypes]: Iterates over ConeTypes, creating a new empty (0, 2) array for each item.
                matched_cones_input: The resulting list stores these empty arrays, with the number of elements in the list matching the length of ConeTypes.
                """
                matched_cones_input = [np.zeros((0, 2)) for _ in ConeTypes]
                matched_cones_input[ConeTypes.LEFT] = sorted_left
                matched_cones_input[ConeTypes.RIGHT] = sorted_right

                cone_matching_input = ConeMatchingInput( # core_cone_matching
                    matched_cones_input, vehicle_position, vehicle_direction
                )
                # chiamata a set_new_input metodo di core_cone_matching
                # Save inputs from other software nodes in variable.
                self.cone_matching.set_new_input(cone_matching_input)
                (
                    left_cones_with_virtual,
                    right_cones_with_virtual,
                    left_to_right_match,
                    right_to_left_match,
                ) = self.cone_matching.run_cone_matching() # calcola i coni matchati

        # run path calculation
        with Timer("Path calculation", noprint=noprint):
            path_calculation_input = PathCalculationInput(
                # dataclass holding calculation variables
                # (core_calculate_path.py)
                left_cones_with_virtual,
                right_cones_with_virtual,
                left_to_right_match,
                right_to_left_match,
                vehicle_position,
                vehicle_direction,
                self.global_path,
            )
            # update the state of the calculation
            self.pathing.set_new_input(path_calculation_input)
            final_path, _ = self.pathing.run_path_calculation() # path_rotated

        # non ci interessa relocalization (non ce freca)
        if self.relocalization_info is not None and self.relocalizer.is_relocalized:
            final_path = final_path.copy()
            # convert path points back to global frame
            path_xy = final_path[:, 1:3]
            fake_yaw = np.zeros(len(path_xy))

            # print("prev", path_xy)

            path_xy, _ = self.relocalizer.transform_to_original_frame(path_xy, fake_yaw)

            # trans rights
            # print("trans", path_xy)

            # assert 0

            final_path = final_path.copy()

            final_path[:, 1:3] = path_xy

        if return_intermediate_results:
            return (
                final_path,
                sorted_left,
                sorted_right,
                left_cones_with_virtual,
                right_cones_with_virtual,
                left_to_right_match,
                right_to_left_match,
            )

        return final_path

    @property
    def relocalization_info(self) -> RelocalizationInformation | None:
        if self.relocalizer is None:
            return None

        if not self.relocalizer.is_relocalized:
            return None

        return RelocalizationInformation.from_transform_function(
            self.relocalizer.transform_to_known_map_frame
        )
