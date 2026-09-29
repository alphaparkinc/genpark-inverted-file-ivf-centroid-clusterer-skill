"""Inverted File (IVF) Centroid Clusterer.
100% Python Standard Library.
"""

import math
from collections import defaultdict

def dot_product(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

def norm(v):
    return math.sqrt(sum(x * x for x in v)) or 1e-9

def cosine_similarity(v1, v2):
    return dot_product(v1, v2) / (norm(v1) * norm(v2))

def euclidean_dist(v1, v2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

class IVFClusterer:
    """Inverted File (IVF) index with centroid-based Voronoi cell routing."""
    def __init__(self, n_centroids=3):
        self.n_centroids = n_centroids
        self.centroids = []
        self.inverted_lists = defaultdict(list)

    def fit_centroids(self, vectors):
        self.centroids = vectors[:self.n_centroids]

    def insert(self, node_id, vector):
        if not self.centroids:
            self.centroids = [vector]
        best_c = min(range(len(self.centroids)), key=lambda c: euclidean_dist(vector, self.centroids[c]))
        self.inverted_lists[best_c].append((node_id, vector))

    def query(self, query_vec, nprobe=1, top_k=2):
        if not self.centroids:
            return []
        sorted_centroids = sorted(range(len(self.centroids)), key=lambda c: euclidean_dist(query_vec, self.centroids[c]))
        candidates = []
        for c in sorted_centroids[:nprobe]:
            for node_id, v in self.inverted_lists[c]:
                sim = cosine_similarity(query_vec, v)
                candidates.append((sim, node_id))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [{"id": nid, "similarity": round(s, 4)} for s, nid in candidates[:top_k]]
