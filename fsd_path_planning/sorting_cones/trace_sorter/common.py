#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: This File provides several functions used in several other files in the
sorting algorithm
Project: fsd_path_planning
"""
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from fsd_path_planning.types import FloatArray, IntArray
from fsd_path_planning.utils.math_utils import my_njit

if not TYPE_CHECKING:

    @my_njit
    def cast(  # pylint: disable=function-redefined
        type_: Any,
        value_: Any,  # pylint: disable=unused-argument
    ) -> Any:
        "Dummy numba jit function"
        return value_


class NoPathError(RuntimeError):
    """
    A special exception thrown when no path can be found (i.e. no configuration)
    """

# richiamata da calc_angle_to_next
# (cost_function.py)
def get_configurations_diff(points: FloatArray, configurations: IntArray) -> FloatArray:
    """
    Gets the difference from each point to its next for each order defined by configurations
    Args:
        points: The points for which the differences should be calculated
        configurations: (n,m), all the configurations that define the orders
    Returns:
        np.array: The difference from one point to the next for each configuration
    """
    result: FloatArray
    result = points[configurations[..., :-1]] # prende tutti tranne l'ultimo
    result -= points[configurations[..., 1:]] # prende tutti tranne il primo
    # lungo l'ultima dimensione
    return result
    # Allora result avrà shape (n, m-1, d)
    # d = dimensione spaziale dei punti (= x, y)
    # n = dimensione int array che corrisponde al path corrente
    # m = dimensione tupla op int-bool


@my_njit
def breadth_first_order(adjacency_matrix: IntArray, start_idx: int) -> IntArray:
    
    # BFS = ricerca in ampiezza = algoritmo ricerca per grafi che partendo da un vertice
    # permette di cercare il cammino fino ad un altro nodo tramite meccanismo di queue.
    """
    Returns the nodes reachable from `start_idx` in BFS order
    Args:
        adjacency_matrix: The adjacency matrix describing the graph
        start_idx: The index of the starting node
    Returns:
        np.array: An array containing the nodes reachable from the starting node in BFS order
    """

    # numero di righe : crea array con stesso numero di righe di a_m, con tutti 0
    visited = np.zeros(adjacency_matrix.shape[0], dtype=np.uint8)

    # queue = array di stessa dim e mette tutto a -1 
    queue = np.full(adjacency_matrix.shape[0], fill_value=-1)

    # primo elemento della coda: start index -> setto a 1 in visired per dire che ho visitato
    queue[0] = start_idx
    visited[start_idx] = 1

    queue_pointer = 0
    queue_end_pointer = 0

    while queue_pointer <= queue_end_pointer:
        node = queue[queue_pointer]

        # argwhere = quasi come traspose, ma trova gli indici degli elementi dell'array
        # che non sono 0, raggruppa per elemento

        # estrae gli indici dei nodi adiacenti al nodo node da una matrice di adiacenza 
        next_nodes = np.argwhere(adjacency_matrix[node])[:, 0]
        # per ogni nodo specificato, ritorna gli indici dei coni che ci sono connessi:
        # prende riga matrice adiacenza -> indici diversi da 0 -> prende prima colonna
        for i in next_nodes:
            if not visited[i]:
                # se nodo non è stato ancora visitato, incremento queue_end_pointer
                # sposto il puntatore della coda ad i
                # e setto visited[i] come visitato
                queue_end_pointer += 1
                queue[queue_end_pointer] = i
                visited[i] = 1

        # incremento il contatore visitato
        queue_pointer += 1

    return cast(IntArray, queue[:queue_pointer])
    # visita tutti i rami possibili per trpvare tutti i nodi raggiungibili. e ritorna i loro indici in ordine
    # come int array
