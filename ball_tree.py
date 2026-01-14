import math

def distance(X, Y): #euklidische Distanz für zwei Punkte berechnen
    return math.sqrt(sum((x - y)**2 for x, y in zip(X, Y)))

class BallTree:
    def __init__(self, data, leaf_size=10):
        self.leaf_size = leaf_size
        self.nodes = [] # Liste zur Speicherung der Baumstruktur
        # Speicherstruktur eines Knotens: (center, radius, left_idx, right_idx, points)
        self.root_idx = self._build_iterative(data)

    def _build_iterative(self, data):   # erstellt Ball-Tree iterativ mittels eines Stacks
        if not data:
            return None

        # Liste aller Knoten
        self.nodes = []

        root_placeholder_idx = 0
        # Stack speichert Daten aus: Datenmenge, Ziel-Index in Knotenliste
        stack = [(data, root_placeholder_idx)]

        # Platzhalter für Wurzelknoten erstellen
        self.nodes.append({})

        while stack:
            current_data, node_idx = stack.pop()

            # Zentrum und Radius berechnen
            points_coords = [p[1] for p in current_data]
            center = self._compute_center(points_coords)
            radius = max(distance(p[1], center) for p in current_data)

            # Abbruchbedingung: Wenn Datenmenge klein genug --> Blattknoten erstellen
            if len(current_data) <= self.leaf_size:
                self.nodes[node_idx] = {
                    'center': center,
                    'radius': radius,
                    'points': current_data,  # Enthält (label, coords)
                    'left': None,
                    'right': None
                }
            else:
                # Splitting-Heuristik: Finde zwei weit entfernte Pole (p1, p2)
                p1 = max(current_data, key=lambda p: distance(p[1], center))
                p2 = max(current_data, key=lambda p: distance(p[1], p1[1]))

                # Punkte dem näheren Pol zuordnen
                left_data, right_data = [], []
                for p in current_data:
                    if distance(p[1], p1[1]) < distance(p[1], p2[1]):
                        left_data.append(p)
                    else:
                        right_data.append(p)

                # Indizes für Kinderknoten am Listenende reservieren
                left_idx = len(self.nodes)
                right_idx = len(self.nodes) + 1
                self.nodes.append({})  # für left
                self.nodes.append({})  # für right

                # Aktuellen Knoten als inneren Knoten (Verzweigung) speichern
                self.nodes[node_idx] = {
                    'center': center,
                    'radius': radius,
                    'points': None,
                    'left': left_idx,
                    'right': right_idx
                }

                # Kinder auf Stack legen
                stack.append((right_data, right_idx))
                stack.append((left_data, left_idx))

        return root_placeholder_idx

    def _compute_center(self, points):
        # Mittelpunkt der Punktemenge berechnen
        d = len(points[0])
        n = len(points)
        return [sum(p[i] for p in points) / n for i in range(d)]

    def query(self, target, k):
        # Sucht k nächsten Nachbarn mittels Pruning
        if self.root_idx is None: return []
        
        neighbors = [] # Liste von (distanz, label)
        stack = [self.root_idx]
        
        while stack:
            idx = stack.pop()
            node = self.nodes[idx]
            
            dist_to_center = distance(target, node['center'])
            
            # Pruning: Wenn Zielpunkt zu weit von Kugel weg, können darin keine besseren Nachbarn mehr liegen
            if len(neighbors) == k and dist_to_center - node['radius'] >= max(neighbors)[0]:
                continue
                
            if node['points'] is not None: # Blattknoten
                for label, coords in node['points']:
                    d = distance(target, coords)
                    # Erhalte Top-K der bisher nächsten Punkte
                    if len(neighbors) < k:
                        neighbors.append((d, label))
                        neighbors.sort()
                    elif d < neighbors[-1][0]:
                        neighbors[-1] = (d, label)
                        neighbors.sort()
            else: # Innerer Knoten
                left = self.nodes[node['left']]
                right = self.nodes[node['right']]

                # Heuristik: Kugel zuerst besuchen, deren Zentrum näher liegt
                if distance(target, left['center']) < distance(target, right['center']):
                    stack.append(node['right'])
                    stack.append(node['left'])
                else:
                    stack.append(node['left'])
                    stack.append(node['right'])
                    
        return [n[1] for n in neighbors]
