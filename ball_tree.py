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

def norm(X):
    return math.sqrt(sum(x**2 for x in X))

def distance(X, Y):
    return math.sqrt(sum((x - y)**2 for x, y in zip(X, Y)))

def farthest_point(X, set):
    return max(set, key=lambda Y: distance(X, Y))

def vector_sum(vectors):
    return [sum(components) for components in zip(*vectors)]


class BallTree:
    def __init__(self, data, leaf_size=1):
        self.leaf_size = leaf_size
        self.points = data 
        self.left = None
        self.right = None

        self.center = self._computer_center(data)
        self.radius = self._compute_radius(data, self.center) 

        if len(data) > self.leaf_size:
            self._split()
    
    def _computer_center(self, data):
        return [coord / len(data) for coord in vector_sum(data)]
    
    def _compute_radius(self, data, center): 
        return max(distance(point, center) for point in data)
    
    def _split(self):
        p1 = farthest_point(self.points[0], self.points)
        p2 = farthest_point(p1, self.points) 

        left_points = []
        right_points = []

        for x in self.points: 
            if distance(x, p1) <= distance(x, p2):
                left_points.append(x)
            else:
                right_points.append(x)
        
        self.left = BallTree(left_points, self.leaf_size)
        self.right = BallTree(right_points, self.leaf_size)

        self.points = None  # free memory

    def knn_update(self, x, k, heap=[]): 
        lower_bound = max(0.0, distance(x, self.center) - self.radius)

        if len(heap) == k and lower_bound > heap[0][0]:
            print("Pruned")
            return 
        
        if self.left is None and self.right is None: 
            for p in self.points:
                dist = distance(x, p)
                if len(heap) < k-1:
                    heap.append((dist, p))
                elif len(heap) == k-1:
                    heap.append((dist, p)) 
                    heap.sort(reverse=True)
                elif dist < heap[0][0]:
                    heap[0] = (dist, p)
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