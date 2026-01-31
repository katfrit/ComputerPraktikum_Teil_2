import math
import time


class BallTree:
    def __init__(self, data, leaf_size=40):
        """Initialize BallTree with precomputed norms for data points.

        OPTIMIZATION: Precompute ||coord||^2 for all data points using binomial formula.
        This is mathematically equivalent but computationally faster:
        ||x - y||^2 = ||x||^2 - 2<x,y> + ||y||^2

        By storing ||y||^2 for all data points, we reduce query time significantly.
        """
        self.leaf_size = leaf_size
        self.nodes = []

        # CRITICAL: Precompute squared norms for ALL data points
        # This is the key - we pay upfront cost for faster queries
        self.query_time = 0.0
        t_start_build = time.perf_counter()

        self.point_norms = {}
        for label, coords in data:
            coords_tuple = tuple(coords)
            if coords_tuple not in self.point_norms:
                # Compute ||coords||^2 = sum(c^2)
                self.point_norms[coords_tuple] = sum(c * c for c in coords)
        self._build_iterative(data)
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
        """Find k nearest neighbors using binomial formula optimization.
        """
        t_start_q = time.perf_counter()
        if not self.nodes: return []

        # OPTIMIZATION Step 1: Precompute ||target||^2 once
        target_norm_sq = sum(t * t for t in target)

        neighbors = []  # saves knn according to (dist_sq, label)
        stack = [0]
        nodes = self.nodes

        while stack:
            idx = stack.pop()
            node = nodes[idx]

            # For center: compute norm on-the-fly (centers are not in dictionary)
            center_norm_sq = sum(c * c for c in node['center'])
            dot_product = sum(t * c for t, c in zip(target, node['center']))
            dist_to_center_sq = target_norm_sq - 2 * dot_product + center_norm_sq
            distance_to_center = math.sqrt(dist_to_center_sq)

            # check pruning condition
            if len(neighbors) == k and distance_to_center > node['radius'] + math.sqrt(neighbors[-1][0]):
                continue

            if node['points'] is not None:
                for label, coord in node['points']:
                    # OPTIMIZATION Step 2: Use precomputed ||coord||^2 from dictionary
                    coord_tuple = tuple(coord)
                    coord_norm_sq = self.point_norms.get(coord_tuple)

                    # Fallback if not in dictionary (should never happen for data points)
                    if coord_norm_sq is None:
                        coord_norm_sq = sum(c * c for c in coord)

                    # Compute dot product <target, coord>
                    dot_product = sum(t * c for t, c in zip(target, coord))

                    # Apply binomial formula: ||target - coord||^2
                    d_sq = target_norm_sq - 2 * dot_product + coord_norm_sq

                    # CRITICAL FIX: Changed from "< k - 1" to "< k"
                    if len(neighbors) < k:
                        neighbors.append((d_sq, label))
                        if len(neighbors) == k:
                            neighbors.sort()
                    elif d_sq < neighbors[-1][0]:
                        neighbors[-1] = (d_sq, label)
                        neighbors.sort()

            else:
                # search closer child first
                l_child = nodes[node['left']]
                r_child = nodes[node['right']]

                # For child centers: compute norm on-the-fly
                l_norm_sq = sum(c * c for c in l_child['center'])
                l_dot = sum(t * c for t, c in zip(target, l_child['center']))
                l_dist_sq = target_norm_sq - 2 * l_dot + l_norm_sq

                r_norm_sq = sum(c * c for c in r_child['center'])
                r_dot = sum(t * c for t, c in zip(target, r_child['center']))
                r_dist_sq = target_norm_sq - 2 * r_dot + r_norm_sq

                if l_dist_sq < r_dist_sq:
                    stack.append(node['right'])
                    stack.append(node['left'])
                else:
                    stack.append(node['left'])
                    stack.append(node['right'])

        self.query_time += (time.perf_counter() - t_start_q)
        return [n[1] for n in neighbors]
