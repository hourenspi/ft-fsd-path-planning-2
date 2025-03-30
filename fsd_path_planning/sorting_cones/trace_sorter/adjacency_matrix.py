#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Description: This File calculates the Adjacency Matrix
Project: fsd_path_planning
"""

from typing import Tuple, cast

import numpy as np

from fsd_path_planning.sorting_cones.trace_sorter.common import breadth_first_order
from fsd_path_planning.types import FloatArray, IntArray
from fsd_path_planning.utils.cone_types import ConeTypes, invert_cone_type
from fsd_path_planning.utils.math_utils import calc_pairwise_distances

LAST_MATRIX_CALC_HASH = None
LAST_MATRIX_CALC_DISTANCE_MATRIX = None


# se è uguale a quella di prima, prende la variabile globale, altrimenti
# ricalcola matrice distanze mediante calc_pairwise_distances
def calculate_distance_matrix(cones_xy: FloatArray) -> FloatArray:
    # salva l'hash dell'ultimo array in input per trovare i cambiamenti
    global LAST_MATRIX_CALC_HASH
    # salva l'ultima matrice distanze calcolata
    global LAST_MATRIX_CALC_DISTANCE_MATRIX
    input_hash = hash(cones_xy.tobytes()) # trasforma coni in hash

    # se è diverso dall'ultimo calcolato
    if input_hash != LAST_MATRIX_CALC_HASH:
        LAST_MATRIX_CALC_HASH = input_hash

        # calc_pairwise_distances
        # (math_utils.py)
        # dato un set di punti, crea la matrice distzna per ogni punto ad ogni punto
        LAST_MATRIX_CALC_DISTANCE_MATRIX = calc_pairwise_distances(
            cones_xy, dist_to_self=np.inf
        )

    return LAST_MATRIX_CALC_DISTANCE_MATRIX.copy()


LAST_IDXS_CALCULATED = None
LAST_IDXS_HASH = None

# trova gli indici dei punti più vicini utilizzando la distanza euclidea
# ne trova un numero pari a n_neighbors
def find_k_closest_in_point_cloud(pairwise_distances: FloatArray, k: int) -> IntArray:
    """
    Finds the indices of the k closest points for each point in a point cloud from its
    pairwise distances.

    Args:
        pairwise_distances: A square matrix containing the distance from each
        point to every other point
        k: The number closest points (indices) to return of each point
    Returns:
        np.array: An (n,k) array containing the indices of the `k` closest points.
    """
    global LAST_IDXS_CALCULATED
    global LAST_IDXS_HASH


    # controllo con hash: se è già stato calcolato ed è doverso dal precendente (var. globale), lo ricalcola
    # altrimenti gli passa quello già salvato
    input_hash = hash((pairwise_distances.tobytes(), k))
    if input_hash != LAST_IDXS_HASH:
        LAST_IDXS_HASH = input_hash
        # li ordina con arg_sort:
        # ordina gli indici degli elementi in ogni riga in ordine crescente
        LAST_IDXS_CALCULATED = np.argsort(pairwise_distances, axis=1)[:, :k] # : - tutte le righe; :k - le prime k colonne
        # axis = specifica l'asse in base al quale computare i valori
        # prende le colonne e le ordina, e prende le prima k colonne
        # ARG SORT ORINA PER INDICE!
    return LAST_IDXS_CALCULATED.copy()


def create_adjacency_matrix(
    cones: FloatArray,
    n_neighbors: int,
    start_idx: int,
    max_dist: float,
    cone_type: ConeTypes,
) -> Tuple[IntArray, IntArray]:
    """
    Creates the adjacency matrix that defines the possible points each point can be connected with
    Args:
        cones: The trace containing all the points
        n_neighbors: The maximum number of neighbors each node can have
        start_idx: The index from which the trace starts
        max_dist: The maximum distance two points can have in order for them to
        be considered possible neighbors
    Returns:
        Tuple[np.array, np.array]: Three values are returned. First a square boolean
        matrix indicating at each position if two nodes are connected. The second 1d
        matrix contains the reachable nodes from `start_idx`.
    """
    # shape=size, poichè c'è 0 prendiamo il numero di righe dell'array 'cones'
    n_points = cones.shape[0]

    cones_xy = cones[:, :2]
    cones_color = cones[:, 2]

    # chiama calculate_distance_matrix
    # (adjacency_matrix.py)
    # setta le variabili globali
    pairwise_distances: FloatArray = calculate_distance_matrix(cones_xy)

    # array di booleano dove inserire se il cono è del tipo opposto
    mask_is_other_cone_type = cones_color == invert_cone_type(cone_type)

    # mediante l'indice e l'array mask_is_other_cone_type
    # setta le distanze dei punti dell'altro tipo come infinito
    pairwise_distances[mask_is_other_cone_type, :] = np.inf
    pairwise_distances[:, mask_is_other_cone_type] = np.inf

    # do not connect points that are very close to each other
    # pairwise_distances[pairwise_distances < 1.5] = np.inf

    # chiama find_k_closest_in_point_cloud
    # (adjacency_matrix.py)
    # per non connettere punti troppo vicini: ritorna array degli indici dei k punti più vicini
    k_closest_each = find_k_closest_in_point_cloud(pairwise_distances, n_neighbors)
    
    # sources 
    sources = np.repeat(np.arange(n_points), n_neighbors) # crea array con elementi da 0 a n_points -1, ognuno ripetuto n_neighbors volte
    targets = k_closest_each.flatten() # prende gli inidic più vicini e li flatten, ovvero rende ad una dimensione
    # targets è un intarray praticamente

    # creazione int array adjacency_matrix
    # zeros: creazione matrice xpointsxnpoints riempita di zeri 
    # uint8 = 8 bit
    adjacency_matrix: IntArray = np.zeros((n_points, n_points), dtype=np.uint8)

    # creazione della mappa di adiacenza per tenere conto, per ogni nodo, dei k punti più vicini:
    # ci sarà un 1 nella matrice ad indicarli.
    adjacency_matrix[
        sources, targets
    ] = 1  # for each node set its closest n_neighbor to 1

    # creazione matrice di true e false:
    # se la distanza > max_dist^2, la connessione non è valida -> settato a 0 nella matrice
    adjacency_matrix[
        pairwise_distances > (max_dist * max_dist)
    ] = 0  # but if distance is too high set to 0 again

    # remove all edges that don't have a revere i.e. convert to undirected graph
    # rimuove tutte le frecce che non hanno corrispondenza doppia, ovvero converte in un grafo senza direzione/collegamento 
    # c'è la trasposta -> se il collegamento non è bidirezionale, viene eliminato
    adjacency_matrix = np.logical_and(adjacency_matrix, adjacency_matrix.T)

    # breathd_first_order: (common.py)
    # ritorna i nodi raggiungibili dall'indice iniziale, in ordine 'BestFirstSearch'
    # prende come parametri la matrice di adiacenza e l'indice iniziale.
    # ritorna un array con gli indici dei nodi raggiungibili dal nodo iniziale.
    reachable_nodes = breadth_first_order(adjacency_matrix, start_idx)

    # completely disconnect nodes that are not reachable from start node
    # assume that all nodes will be disconnected
    nodes_to_disconnect = np.ones(n_points, dtype=bool)


    # but for the reachable nodes don't do anything
    # setta a false i nodi raggiungibili in modo che non siano disconnessi
    nodes_to_disconnect[reachable_nodes] = False

    # if we are sorting left (blue) cones, then we want to disconnect
    # all right (yellow) cones
    # nodes_to_disconnect[cones_color == invert_cone_type(cone_type)] = True

    # disconnect the remaining nodes in both directions
    # adjacency_matrix[:, nodes_to_disconnect] = 0
    # adjacency_matrix[nodes_to_disconnect, :] = 0

    # ritorna matrice di adiacenza e nodi raggiungibili
    return adjacency_matrix, reachable_nodes
