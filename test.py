from ball_tree import BallTree

def main():
    print("test")

    points_clusters = [
    (1,[-0.8, -0.8]), (1,[-0.7, -0.9]), (1,[-0.9, -0.7]),
    (1,[-0.8, -0.6]), (1,[-0.6, -0.8]),

    (-1,[0.0, 0.0]), (-1,[0.1, -0.1]), (-1,[-0.1, 0.1]),
    (-1,[0.2, 0.0]), (-1,[0.0, 0.2]),

    (1,[0.8, 0.8]), (1,[0.9, 0.7]), (1,[0.7, 0.9]),
    (1,[0.85, 0.85]), (1,[0.75, 0.8]),
    ]

    tree = BallTree(points_clusters, leaf_size=2)
    print("BallTree constructed successfully.")

    while tree.left is not None and tree.right is not None:
        print(f"Node center: {tree.center}, radius: {tree.radius}, points: {tree.data}")
        tree = tree.right  # Traverse to right child for demonstration
    print(f"Node center: {tree.center}, radius: {tree.radius}, points: {tree.data}")

    tree = BallTree(points_clusters, leaf_size=2)
    print(tree.knn_query([0.0, 0.0], k=3))
    print(tree.knn_query([1.0, 1.0], k=3))
    print(tree.knn_query([0.5, 0.6], k=4))
    print(tree.knn_query([0.2, 0.0], k=1))



if __name__ == "__main__":
    main()
