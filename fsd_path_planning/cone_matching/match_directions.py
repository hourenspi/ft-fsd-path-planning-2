import numpy as np

from fsd_path_planning.utils.cone_types import ConeTypes
from fsd_path_planning.utils.math_utils import my_njit, rotate

# richiamata da calculate_match_search_direction_for_one_if_not_in_cache
# (nearby_cone_serach.py) -> SORTING

# richiamata anche da calculate_match_search_direction
@my_njit
def calculate_search_direction_for_one(cones, idxs, cone_type):
    """
    Calculates the search direction for one cone
    """
    # si assicura che gli indici relativi ai coni siano 2
    assert len(idxs) == 2

    track_direction = cones[idxs[1]] - cones[idxs[0]]

    rotation_angle = np.pi / 2 if cone_type == ConeTypes.RIGHT else -np.pi / 2

    # ruota e normalizza
    search_direction = rotate(track_direction, rotation_angle)

    return search_direction / np.linalg.norm(search_direction)

# richiamato da calculate_matches_for_side
# (functional_cone_matching.py)
@my_njit
def calculate_match_search_direction(
    cones,
    cone_type: ConeTypes,
):
    number_of_cones = len(cones)
    assert number_of_cones > 1

    cones_xy = cones[:, :2]

    out = np.zeros((number_of_cones, 2))
    out[0] = calculate_search_direction_for_one(cones_xy, np.array([0, 1]), cone_type) # array[0,1]: indici    Calculates the search direction for one cone
    out[-1] = calculate_search_direction_for_one(
        cones_xy, np.array([-2, -1]), cone_type
    )

    for i in range(1, number_of_cones - 1):
        out[i] = calculate_search_direction_for_one(
            cones_xy, np.array([i - 1, i + 1]), cone_type
        )

    return out
