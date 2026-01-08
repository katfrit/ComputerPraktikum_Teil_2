"""
def norm(x):
    norm_squared = 0

    try:
        for component in x:
            try:
                norm_squared += component**2
            except TypeError:
                print("Expected float valued entries.")
                sys.exit(1)
    except AttributeError:
        print("Expected list as input.")
        sys.exit(1)

    return math.sqrt(norm_squared)
"""

import math
import random

def norm(X):
    return math.sqrt(sum(x**2 for x in X))

def distance(X, Y):
    return math.sqrt(sum((x - y)**2 for x, y in zip(X, Y)))

def farthest_point(X, set):
    return max(set, key=lambda Y: distance(X, Y))

def vector_sum(vectors):
    return [sum(components) for components in zip(*vectors)]

def sign(x):
    return 1 if x >= 0 else -1


class BallTree:
    def __init__(self, data, leaf_size=1):
        self.leaf_size = leaf_size
        self.data = data 
        self.left = None
        self.right = None

        self.center, self.radius = self._compute_center_radius()

        if len(data) > self.leaf_size:
            self._split()
    
    def _compute_center_radius(self):
        points = [entry[1] for entry in self.data]
        center = [coord / len(points) for coord in vector_sum(points)]
        return center, max(distance(point, center) for point in points)
    
    def _split(self):
        points = [entry[1] for entry in self.data]
        p1 = farthest_point(points[0], points)
        p2 = farthest_point(p1, points) 

        left_data = []
        right_data = []

        for d in self.data: 
            if distance(d[1], p1) <= distance(d[1], p2):
                left_data.append(d)
            else:
                right_data.append(d)
        
        self.left = BallTree(left_data, self.leaf_size)
        self.right = BallTree(right_data, self.leaf_size)

        self.data = None  # free memory

    def knn_update(self, x, k, heap=[]): 
        lower_bound = max(0.0, distance(x, self.center) - self.radius)

        if len(heap) == k and lower_bound > heap[0][0]:
            return 
        
        if self.left is None and self.right is None: 
            for d in self.data:
                dist = distance(x, d[1])
                if len(heap) < k-1:
                    heap.append((dist, d))
                elif len(heap) == k-1:
                    heap.append((dist, d)) 
                    heap.sort(reverse=True)
                elif dist < heap[0][0]:
                    heap[0] = (dist, d)
                    heap.sort(reverse=True)
            return 
        
        if distance(x, self.left.center) < distance(x, self.right.center):
            self.left.knn_update(x, k, heap)
            self.right.knn_update(x, k, heap)
        else:
            self.right.knn_update(x, k, heap)
            self.left.knn_update(x, k, heap)

    def knn_query(self, x, k):
        heap = []
        self.knn_update(x, k, heap)
        return [p for (_, p) in heap]
    
    def predict(self, x, k):
        knn = self.knn_query(x, k)
        return sign(sum(labels for labels, _ in knn))
    

class Classifier:
    def __init__(self, data, l, mode, K, leaf_size=1):
        self.folds = self._generate_folds(data, l, mode)
        self.folds_gapped = self._generate_folds_gapped() 
        self.l = l 
        self.K = K
        self.leaf_size = leaf_size

        self.trees = [BallTree(fold, leaf_size=self.leaf_size) for fold in self.folds_gapped]
        self.k_opt = self._find_k_opt()


    def _generate_folds(data, l, mode):
        n = len(data)
        folds = [[] for _ in range(l)]

        if mode == 0:
            random.shuffle(data) 
            for i, item in enumerate(data):
                folds[i % l].append(item)
        else:
            for idx, item in enumerate(data):
                folds[idx % l].append(item)

        return folds

    def _generate_folds_gapped(self):
        folds_gapped = []

        for i in range(len(self.folds)):
            gapped_fold = []
            for j in range(len(self.folds)):
                if j != i:
                    gapped_fold.extend(self.folds[j])
            folds_gapped.append(gapped_fold)

        return folds_gapped

    def _find_k_opt(self):
        residues = []
        for k in self.K:
            avg_residue = sum(sum(abs(y - self.trees[i].predict(x, k)) for y, x in self.folds[i]) / len(self.folds[i]) for i in range(self.l))
            residues.append((k, avg_residue))
        return min(residues, key=lambda item: item[1])[0]

    def predict(self, x):
        return sign(sum(self.trees[i].predict(x, self.k_opt) for i in range(self.l)))