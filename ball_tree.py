import math


class BallTree:
    def __init__(self, data):
        # Blattgröße wie in deinem effizienten Original
        self.leaf_size = max(40, min(200, len(data) // 100))
        self.nodes = []
        self.root_idx = self._build_iterative(data)

    def _build_iterative(self, data):
        if not data: return None
        self.nodes = []
        stack = [(data, 0)]
        self.nodes.append({})

        while stack:
            current_data, node_idx = stack.pop()
            points_coords = [p[1] for p in current_data]
            n_points = len(points_coords)
            dim = len(points_coords[0])

            center = [sum(p[i] for p in points_coords) / n_points for i in range(dim)]
            max_dist_sq = 0.0
            for p_coords in points_coords:
                d_sq = sum((p_coords[i] - center[i]) ** 2 for i in range(dim))
                if d_sq > max_dist_sq: max_dist_sq = d_sq

            radius_val = math.sqrt(max_dist_sq)

            if n_points <= self.leaf_size:
                self.nodes[node_idx] = {
                    'center': center, 'radius_sq': max_dist_sq, 'radius': radius_val,
                    'points': current_data, 'left': None, 'right': None
                }
            else:
                best_dim = 0
                max_spread = -1
                for d in range(dim):
                    vals = [p[d] for p in points_coords]
                    spread = max(vals) - min(vals)
                    if spread > max_spread:
                        max_spread = spread
                        best_dim = d

                current_data.sort(key=lambda x: x[1][best_dim])
                mid = n_points // 2
                l_idx, r_idx = len(self.nodes), len(self.nodes) + 1
                self.nodes.extend([{}, {}])
                self.nodes[node_idx] = {
                    'center': center, 'radius_sq': max_dist_sq, 'radius': radius_val,
                    'points': None, 'left': l_idx, 'right': r_idx
                }
                stack.append((current_data[mid:], r_idx))
                stack.append((current_data[:mid], l_idx))
        return 0

    def query(self, target, k):
        if not self.nodes: return []
        neighbors = []  # (dist_sq, label)
        stack = [0]
        nodes = self.nodes
        dim_range = range(len(target))

        while stack:
            idx = stack.pop()
            node = nodes[idx]
            center = node['center']
            d_sq_to_center = sum((target[i] - center[i]) ** 2 for i in dim_range)

            # Pruning
            if len(neighbors) == k:
                d_max_sq = neighbors[-1][0]
                d_max = math.sqrt(d_max_sq)

                # Binomische Formel: (r + d_max)^2 = r^2 + d_max^2 + 2 * r * d_max
                if d_sq_to_center > (node['radius_sq'] + d_max_sq + 2 * node['radius'] * d_max):
                    continue

            if node['points'] is not None:
                for label, coords in node['points']:
                    d_sq = sum((target[i] - coords[i]) ** 2 for i in dim_range)
                    if len(neighbors) < k:
                        neighbors.append((d_sq, label))
                        if len(neighbors) == k: neighbors.sort()
                    elif d_sq < neighbors[-1][0]:
                        neighbors[-1] = (d_sq, label)
                        neighbors.sort()
            else:
                # Optimierung: Zuerst in das Kind gehen, das dem Target näher ist
                l_child = nodes[node['left']]
                r_child = nodes[node['right']]

                # Wir schätzen, welches Zentrum näher liegt
                d_l = sum((target[i] - l_child['center'][i]) ** 2 for i in dim_range)
                d_r = sum((target[i] - r_child['center'][i]) ** 2 for i in dim_range)

                if d_l < d_r:
                    stack.append(node['right'])
                    stack.append(node['left'])
                else:
                    stack.append(node['left'])
                    stack.append(node['right'])

        return [n[1] for n in neighbors]
