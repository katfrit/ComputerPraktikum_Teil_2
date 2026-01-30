import math


class BallTree:
    def __init__(self, data, leaf_size=40):
        self.leaf_size = leaf_size
        self.nodes = []
        self._build_iterative(data)

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
        if not self.nodes: return []
        neighbors = []  # saves knn according to (dist_sq, label)
        stack = [0]
        nodes = self.nodes
        dim_range = range(len(target))

        while stack:
            idx = stack.pop()
            node = nodes[idx]

            distance_to_center = math.sqrt(sum((x - y) ** 2 for x, y in zip(target, node['center'])))

            # check pruning condition
            if len(neighbors) == k and distance_to_center > node['radius'] + math.sqrt(neighbors[-1][0]):
                continue



            if node['points'] is not None:
                for label, coord in node['points']:
                    d_sq = sum((x - y) ** 2 for x, y in zip(target, coord))
                    if len(neighbors) < k:
                        neighbors.append((d_sq, label))
                        neighbors.sort()
                    elif d_sq < neighbors[-1][0]:
                        neighbors[-1] = (d_sq, label)
                        neighbors.sort()


            else:
                # search closer child first
                l_child = nodes[node['left']]
                r_child = nodes[node['right']]

                if sum((x - y) ** 2 for x, y in zip(target, l_child['center'])) < sum(
                        (x - y) ** 2 for x, y in zip(target, r_child['center'])):
                    stack.append(node['right'])
                    stack.append(node['left'])
                else:
                    stack.append(node['left'])
                    stack.append(node['right'])

        return [n[1] for n in neighbors]
