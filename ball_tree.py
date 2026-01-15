import math


def get_dist_sq(X, Y):
    # Berechnet die quadrierte euklidische Distanz (schneller als mit sqrt)
    return sum([(x - y) ** 2 for x, y in zip(X, Y)])


class BallTree:
    def __init__(self, data, leaf_size=40):
        self.leaf_size = leaf_size
        self.nodes = []
        # root_idx speichert den Einstiegspunkt
        self.root_idx = self._build_iterative(data)

    def _build_iterative(self, data):
        if not data: return None
        self.nodes = []
        root_placeholder_idx = 0
        stack = [(data, root_placeholder_idx)]
        self.nodes.append({})

        while stack:
            current_data, node_idx = stack.pop()
            points_coords = [p[1] for p in current_data]

            # Mittelpunkt berechnen
            n_points = len(points_coords)
            dim = len(points_coords[0])
            center = [sum(p[i] for p in points_coords) / n_points for i in range(dim)]

            # Radius QUADRIERT berechnen
            max_dist_sq = 0.0
            for p_coords in points_coords:
                d_sq = sum([(p_coords[i] - center[i]) ** 2 for i in range(dim)])
                if d_sq > max_dist_sq: max_dist_sq = d_sq

            if len(current_data) <= self.leaf_size:
                self.nodes[node_idx] = {
                    'center': center,
                    'radius_sq': max_dist_sq,
                    'points': current_data,
                    'left': None, 'right': None
                }
            else:
                # Splitting nach der Dimension mit dem größten Spread
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
                    'center': center, 'radius_sq': max_dist_sq,
                    'points': None, 'left': l_idx, 'right': r_idx
                }
                stack.append((current_data[mid:], r_idx))
                stack.append((current_data[:mid], l_idx))
        return root_placeholder_idx

    def query(self, target, k):
        if self.root_idx is None: return []

        neighbors = []  # Speichert (dist_sq, label)
        stack = [self.root_idx]

        # Lokale Referenz für schnelleren Zugriff in der Schleife
        nodes = self.nodes
        target_indices = range(len(target))

        while stack:
            idx = stack.pop()
            node = nodes[idx]

            center = node['center']
            # Quadrierte Distanz zum Kugel-Zentrum
            d_sq_to_center = sum([(target[i] - center[i]) ** 2 for i in target_indices])

            # Pruning mit Quadraten (Wurzel nur wenn absolut nötig)
            if len(neighbors) == k:
                max_d_sq = neighbors[-1][0]
                # Mathematische Pruning-Bedingung: dist - radius >= max_d
                # Da wir Quadrate haben: sqrt(d_sq_to_center) - sqrt(radius_sq) >= sqrt(max_d_sq)
                radius = math.sqrt(node['radius_sq'])
                if math.sqrt(d_sq_to_center) - radius >= math.sqrt(max_d_sq):
                    continue

            if node['points'] is not None:
                # Blattknoten: Punkte prüfen
                for label, coords in node['points']:
                    d_sq = sum([(target[i] - coords[i]) ** 2 for i in target_indices])

                    if len(neighbors) < k:
                        neighbors.append((d_sq, label))
                        if len(neighbors) == k: neighbors.sort(key=lambda x: x[0])
                    elif d_sq < neighbors[-1][0]:
                        neighbors[-1] = (d_sq, label)
                        neighbors.sort(key=lambda x: x[0])
            else:
                # Innerer Knoten: Heuristik für Kinder-Besuch
                l_idx, r_idx = node['left'], node['right']
                c_l, c_r = nodes[l_idx]['center'], nodes[r_idx]['center']
                d_l = sum([(target[i] - c_l[i]) ** 2 for i in target_indices])
                d_r = sum([(target[i] - c_r[i]) ** 2 for i in target_indices])

                if d_l < d_r:
                    stack.append(r_idx);
                    stack.append(l_idx)
                else:
                    stack.append(l_idx);
                    stack.append(r_idx)

        return [n[1] for n in neighbors]
