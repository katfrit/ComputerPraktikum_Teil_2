import math
import time

# BallTree data structure for efficient k-NN search.
# Builds a binary tree by recursively partitioning the data into hyperspherical nodes.
# Leaf nodes store data points, inner nodes store center and radius for pruning.
# Nearest-neighbor queries are performed using a best-first search with distance-based pruning.

class BallTree:
    """
    BallTree data structure for efficient k-NN search.

    Structure:
        __init__(data, leaf_size)
            Initializes the tree and precomputes point norms.

        _build_iterative(data)
            Builds the tree iteratively using ball partitioning.

        query(target, k)
            Performs a k-nearest neighbor search with pruning.

    Each node stores a center and radius; leaf nodes additionally
    store the associated data points.
    """

    def __init__(self, data, leaf_size=40):
        self.leaf_size = leaf_size
        self.nodes = []
        self.point_norms = {}
        self.query_time = 0.0
        t_start_build = time.perf_counter()

        # Calculating point-norms
        for label, coords in data:
            coords_tuple = tuple(coords)
            if coords_tuple not in self.point_norms:
                self.point_norms[coords_tuple] = sum(c * c for c in coords)

        self._build_iterative(data)

        # Precompute the center norms for faster distance computation
        for node in self.nodes:
            if node:
                node['center_norm_sq'] = sum(c * c for c in node['center'])

        self.build_time = time.perf_counter() - t_start_build


    def _build_iterative(self, data):
        if not data: return None
        self.nodes = []
        stack = [(data, 0)]
        self.nodes.append({})

        while stack:
            current_data, node_idx = stack.pop()
            points_coords = [p[1] for p in current_data]

            # determine center and radius
            center = [coord / len(points_coords) for coord in [sum(components) for components in zip(*points_coords)]]
            radius = max(math.sqrt(sum((x - y) ** 2 for x, y in zip(point, center))) for point in points_coords)

            if len(points_coords) <= self.leaf_size:
                # save actual data
                self.nodes[node_idx] = {'center': center, 'radius': radius, 'points': current_data, 'left': None,
                                        'right': None}
            else:
                p1 = max(points_coords, key=lambda Y: sum((x - y) ** 2 for x, y in zip(points_coords[0], Y)))
                p2 = max(points_coords, key=lambda Y: sum((x - y) ** 2 for x, y in zip(p1, Y)))

                left_data, right_data = [], []

                for data in current_data:
                    if math.sqrt(sum((x - y) ** 2 for x, y in zip(data[1], p1))) <= math.sqrt(
                            sum((x - y) ** 2 for x, y in zip(data[1], p2))):
                        left_data.append(data)
                    else:
                        right_data.append(data)

                l_idx, r_idx = len(self.nodes), len(self.nodes) + 1
                self.nodes.extend([{}, {}])
                self.nodes[node_idx] = {'center': center, 'radius': radius, 'points': None, 'left': l_idx,
                                        'right': r_idx}

                stack.append((right_data, r_idx))
                stack.append((left_data, l_idx))


    def query(self, target, k):
        t_start_q = time.perf_counter()
        if not self.nodes: return []

        target_norm_sq = sum(t * t for t in target)
        neighbors = []  # (dist_sq, label)

        # Initialize the queue with the root
        root = self.nodes[0]
        dot_root = sum(t * c for t, c in zip(target, root['center']))
        d_root = math.sqrt(max(0, target_norm_sq - 2 * dot_root + root['center_norm_sq']))

        # Queue format: (distance_to_center, index)
        queue = [(d_root, 0)]

        while queue:
            # Sorting for best-first search (smallest distance first)
            queue.sort(key=lambda x: x[0], reverse=True)
            d_to_center, idx = queue.pop()
            node = self.nodes[idx]

            # Early termination (global stop)
            if len(neighbors) == k:
                # d_min_so_far is the square root of the worst distance
                if d_to_center > node['radius'] + math.sqrt(neighbors[-1][0]):
                    break  

            if node['points'] is not None:
                for label, coord in node['points']:
                    c_tuple = tuple(coord)
                    # Binomial formula: ||x-y||^2 = ||x||^2 - 2<x,y> + ||y||^2
                    d_sq = target_norm_sq - 2 * sum(t * c for t, c in zip(target, coord)) + self.point_norms[c_tuple]

                    if len(neighbors) < k:
                        neighbors.append((d_sq, label))
                        neighbors.sort()
                    elif d_sq < neighbors[-1][0]:
                        neighbors[-1] = (d_sq, label)
                        neighbors.sort()
            else:
                # Inspect child nodes and add them to the queue only if they can potentially improve the result

                for child_idx in [node['left'], node['right']]:
                    child = self.nodes[child_idx]
                    dot_child = sum(t * c for t, c in zip(target, child['center']))
                    d_child = math.sqrt(max(0, target_norm_sq - 2 * dot_child + child['center_norm_sq']))

                    # Local pruning: add only if the ball is reachable

                    if len(neighbors) < k or d_child <= child['radius'] + math.sqrt(neighbors[-1][0]):
                        queue.append((d_child, child_idx))

        self.query_time += (time.perf_counter() - t_start_q)
        return [n[1] for n in neighbors]

