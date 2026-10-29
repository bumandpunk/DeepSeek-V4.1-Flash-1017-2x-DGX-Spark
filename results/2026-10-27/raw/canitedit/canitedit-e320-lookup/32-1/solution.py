import numpy as np
from typing import Dict, List

class MarkovChain:

    def create_transition_matrix(self, matrix):
        
        matrix = np.array(matrix)
        column_sums = np.sum(matrix, axis=0)
        normalized_matrix = matrix / column_sums
        return normalized_matrix.tolist()

    def translate_from_list(self, adj_list: Dict[int, List[int]]) -> List[List[float]]:
        nodes = sorted(adj_list.keys())
        node_index = {node: i for i, node in enumerate(nodes)}
        n = len(nodes)
        matrix = np.zeros((n, n))
        for node, neighbors in adj_list.items():
            i = node_index[node]
            for neighbor in neighbors:
                j = node_index[neighbor]
                matrix[i, j] = 1
                matrix[j, i] = 1
        column_sums = np.sum(matrix, axis=0)
        column_sums[column_sums == 0] = 1
        normalized_matrix = matrix / column_sums
        return normalized_matrix.tolist()