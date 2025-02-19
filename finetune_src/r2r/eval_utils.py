''' Utils for io, language, connectivity graphs etc '''

import os
import sys
import re

import MatterSim
import string
import json
import jsonlines
import time
import math
import h5py
import dtw
from collections import Counter, defaultdict
import numpy as np
import networkx as nx

from numpy.linalg import norm



class FloydGraph:
    def __init__(self):
        self._dis = defaultdict(lambda :defaultdict(lambda: 95959595))
        self._point = defaultdict(lambda :defaultdict(lambda: ""))
        self._visited = set()

    def distance(self, x, y):
        if x == y:
            return 0
        else:
            return self._dis[x][y]

    def add_edge(self, x, y, dis):
        if dis < self._dis[x][y]:
            self._dis[x][y] = dis
            self._dis[y][x] = dis
            self._point[x][y] = ""
            self._point[y][x] = ""

    def update(self, k):
        for x in self._dis:
            for y in self._dis:
                if x != y:
                    if self._dis[x][k] + self._dis[k][y] < self._dis[x][y]:
                        self._dis[x][y] = self._dis[x][k] + self._dis[k][y]
                        self._dis[y][x] = self._dis[x][y]
                        self._point[x][y] = k
                        self._point[y][x] = k
        self._visited.add(k)

    def visited(self, k):
        return (k in self._visited)

    def path(self, x, y):
        """
        :param x: start
        :param y: end
        :return: the path from x to y [v1, v2, ..., v_n, y]
        """
        if x == y:
            return []
        if self._point[x][y] == "":     # Direct edge
            return [y]
        else:
            k = self._point[x][y]
            # print(x, y, k)
            # for x1 in (x, k, y):
            #     for x2 in (x, k, y):
            #         print(x1, x2, "%.4f" % self._dis[x1][x2])
            return self.path(x, k) + self.path(k, y)


def cal_dtw(shortest_distances, prediction, reference, success=None, threshold=3.0):
    dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
    dtw_matrix[0][0] = 0
    for i in range(1, len(prediction)+1):
        for j in range(1, len(reference)+1):
            best_previous_cost = min(
                dtw_matrix[i-1][j], dtw_matrix[i][j-1], dtw_matrix[i-1][j-1])
            cost = shortest_distances[prediction[i-1]][reference[j-1]]
            dtw_matrix[i][j] = cost + best_previous_cost

    dtw = dtw_matrix[len(prediction)][len(reference)]
    ndtw = np.exp(-dtw/(threshold * len(reference)))
    if success is None:
        success = float(shortest_distances[prediction[-1]][reference[-1]] < threshold)
    sdtw = success * ndtw

    return {
        'DTW': dtw,
        'nDTW': ndtw,
        'SDTW': sdtw
    }

def cal_dtw_window(prediction, reference, success=None, threshold=3.0, env=None):
    assert env is not None
    current_scan = prediction[0]['scan']
    shortest_distances = env.shortest_distances[current_scan]
    alignments = alignments_from_paths(prediction, reference)
    prediction = [p["node"] for p in prediction]
    reference = [p["node"] for p in reference]
    window = window_align_func(0, 0, len(prediction), len(reference), alignments=alignments)
    dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
    dtw_matrix[0][0] = 0
    for i in range(1, len(prediction)+1):
        for j in range(1, len(reference)+1):
            if not window[i - 1][j - 1]:
                continue
            best_previous_cost = min(
                dtw_matrix[i-1][j], dtw_matrix[i][j-1], dtw_matrix[i-1][j-1])
            cost = shortest_distances[prediction[i-1]][reference[j-1]]
            dtw_matrix[i][j] = cost + best_previous_cost

    dtw = dtw_matrix[len(prediction)][len(reference)]
    ndtw = np.exp(-dtw/(threshold * len(reference)))

    success = 0.0
    alignments = [(0, 0)] + alignments
    alignments.append((len(prediction) - 1, len(reference) - 1))
    episodes_num = len(alignments) // 2
    for i, end_idx in enumerate(alignments):
        if i % 2 == 1:
            prediction_end = end_idx[0]
            reference_end = end_idx[1]
            this_success = float(shortest_distances[prediction[prediction_end]][reference[reference_end]] < threshold)
            success += (this_success * 1 / episodes_num)
    # if success is None:
    #     success = float(shortest_distances[prediction[-1]][reference[-1]] < threshold)
    sdtw = success * ndtw

    return {
        'DTW': dtw,
        'nDTW': ndtw,
        'SDTW': sdtw
    }

def cal_dtw_by_dtw_py_per_ep(agent_path, gt_path, dist_func, threshold=3.0):
    alignments = alignments_from_paths(agent_path, gt_path)
    ap = [p["position"] for p in agent_path]
    gtp = [p["position"] for p in gt_path]

    ndtw_list = list()

    alignments = [(0, 0)] + alignments
    alignments.append((len(agent_path), len(gt_path))) # NOTE: here should be len(agent_path)-1 and len(gt_path)-1
    for i in range(len(alignments)):
        if i % 2 == 1:
            continue
        a_start, gt_start = alignments[i]
        a_end, gt_end = alignments[i + 1]
        for j in range(a_start, a_end - 1):
            assert agent_path[j]['episode_id'] == agent_path[j+1]['episode_id']
        if a_end + 1 < len(agent_path):
            assert agent_path[a_end]['episode_id'] != agent_path[a_end+1]['episode_id']
        for j in range(gt_start, gt_end - 1):
            assert gt_path[j]['episode_id'] == gt_path[j+1]['episode_id']
        if gt_end + 1 < len(gt_path):
            assert gt_path[gt_end]['episode_id'] != gt_path[gt_end+1]['episode_id']
        ap_episode = ap[a_start:a_end+1]
        gtp_episode = gtp[gt_start:gt_end+1]
        dtw_dist = dtw.dtw(ap_episode, gtp_episode, dist_method=dist_func, step_pattern="symmetric1").distance
        ndtw_list.append(np.exp(-dtw_dist / (len(gtp_episode) * threshold)))
    return ndtw_list

def cal_cls(shortest_distances, prediction, reference, threshold=3.0):
    def length(nodes):
      return np.sum([
          shortest_distances[a][b]
          for a, b in zip(nodes[:-1], nodes[1:])
      ])

    coverage = np.mean([
        np.exp(-np.min([  # pylint: disable=g-complex-comprehension
            shortest_distances[u][v] for v in prediction
        ]) / threshold) for u in reference
    ])
    expected = coverage * length(reference)
    score = expected / (expected + np.abs(expected - length(prediction)))
    return coverage * score
    
def extract_ep_order(path):
    eps = [p["episode_id"] for p in path]
    eps_single = []
    for i in range(1, len(eps)):
        if eps[i-1] != eps[i]:
            eps_single.append(eps[i-1])
    eps_single.append(eps[-1])
    return eps_single

def alignments_from_paths(agent_path, gt_path):
    assert extract_ep_order(gt_path) == extract_ep_order(agent_path), (
        "agent and GT episode orders do not match."
    )

    alen = len(agent_path)
    gtlen = len(gt_path)

    agent_alignment_points = []
    for i in range(1, alen):
        if agent_path[i]["episode_id"] != agent_path[i - 1]["episode_id"]:
            agent_alignment_points.append(i - 1)  # stopping point
            agent_alignment_points.append(i)  # starting point

    gt_alignment_points = []
    for i in range(1, gtlen):
        if gt_path[i]["episode_id"] != gt_path[i - 1]["episode_id"]:
            gt_alignment_points.append(i - 1)  # stopping point
            gt_alignment_points.append(i)  # starting point

    assert len(agent_alignment_points) == len(gt_alignment_points), (
        "mismatch in number of alignment points."
    )
    return list(zip(agent_alignment_points, gt_alignment_points))

def window_align_func(iw, jw, query_size, reference_size, alignments):
    window = np.ones((query_size, reference_size), dtype=np.bool)

    # for each alignment (i, j), make col j False except (i,j)
    for (i, j) in alignments:
        window[:, j] = False
        window[:, j] = False
        window[i, j] = True

    return window

def compute_tour_ndtw(agent_path, gt_path, dist_func, threshold=3.0):
    # compute the constrained ndtw of each tour
    alignments = alignments_from_paths(agent_path, gt_path)
    ap = [p["position"] for p in agent_path]
    gtp = [p["position"] for p in gt_path]
    dtw_dist = dtw.dtw(
        ap,
        gtp,
        dist_method=dist_func,
        step_pattern="symmetric1",
        window_type=window_align_func,
        window_args={"alignments": alignments},
    ).distance
    return np.exp(-dtw_dist / (len(gtp) * threshold))