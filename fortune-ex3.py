import math
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Coordinate({self.x}, {self.y})"

class Polygon:
    def __init__(self, vertex1, vertex2, vertex3):
        self.points = [vertex1, vertex2, vertex3]

    def midpoint(self, point1, point2):
        return Coordinate((point1.x + point2.x) / 2, (point1.y + point2.y) / 2)

    def bisector(self, point1, point2):
        mid = self.midpoint(point1, point2)
        delta_x = point2.x - point1.x
        delta_y = point2.y - point1.y
        if delta_x == 0:
            perpendicular_slope = 0
        elif delta_y == 0:
            perpendicular_slope = None
        else:
            perpendicular_slope = -1 / (delta_y / delta_x)
        return mid, perpendicular_slope

    def intersect(self, mid1, slope1, mid2, slope2):
        if slope1 is None:
            x = mid1.x
            y = slope2 * (x - mid2.x) + mid2.y
        elif slope2 is None:
            x = mid2.x
            y = slope1 * (x - mid1.x) + mid1.y
        elif slope1 == slope2:
            return None
        else:
            x = (slope2 * mid2.x - slope1 * mid1.x + mid1.y - mid2.y) / (slope2 - slope1)
            y = slope1 * (x - mid1.x) + mid1.y
        return Coordinate(x, y)

    def circumcircle_properties(self):
        point_a, point_b, point_c = self.points
        mid1, slope1 = self.bisector(point_a, point_b)
        mid2, slope2 = self.bisector(point_b, point_c)
        center = self.intersect(mid1, slope1, mid2, slope2)
        if center is None:
            return None, None
        radius = math.sqrt((center.x - point_a.x) ** 2 + (center.y - point_a.y) ** 2)
        return center, radius

    def includes_point(self, point):
        center, radius = self.circumcircle_properties()
        if center is None:
            return False
        distance_sq = (point.x - center.x) ** 2 + (point.y - center.y) ** 2
        return distance_sq <= radius ** 2

    def __repr__(self):
        return f"Polygon({self.points[0]}, {self.points[1]}, {self.points[2]})"

def triangulate(vertices, epsilon=1e-7):
    def near(point1, point2):
        return abs(point1.x - point2.x) < epsilon and abs(point1.y - point2.y) < epsilon

    filtered_points = []
    for vertex in vertices:
        if not any(near(vertex, existing) for existing in filtered_points):
            filtered_points.append(vertex)

    vertices = filtered_points

    def aligned(point1, point2, point3):
        return abs((point2.x - point1.x) * (point3.y - point1.y) - (point2.y - point1.y) * (point3.x - point1.x)) < epsilon

    vertices.sort(key=lambda vertex: vertex.x)

    min_x, max_x = min(vertex.x for vertex in vertices), max(vertex.x for vertex in vertices)
    min_y, max_y = min(vertex.y for vertex in vertices), max(vertex.y for vertex in vertices)
    delta = max(max_x - min_x, max_y - min_y)

    super_polygon = Polygon(
        Coordinate(min_x - 7 * delta, min_y - delta),
        Coordinate(min_x + delta, max_y + 7 * delta),
        Coordinate(max_x + 7 * delta, min_y - delta)
    )

    polygons = [super_polygon]

    for vertex in vertices:
        invalid_polygons = [poly for poly in polygons if poly.includes_point(vertex)]

        boundary = []
        for poly in invalid_polygons:
            for edge in itertools.combinations(poly.points, 2):
                ordered_edge = tuple(sorted(edge, key=lambda p: (p.x, p.y)))
                if ordered_edge in boundary:
                    boundary.remove(ordered_edge)
                else:
                    boundary.append(ordered_edge)

        for poly in invalid_polygons:
            polygons.remove(poly)

        for edge in boundary:
            if not aligned(edge[0], edge[1], vertex):
                polygons.append(Polygon(edge[0], edge[1], vertex))

    polygons = [
        poly for poly in polygons
        if not any(vertex in super_polygon.points for vertex in poly.points)
    ]

    return polygons

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def triangulation_to_graph(polygons):
    connections = set()
    for poly in polygons:
        points = poly.points
        connections.add((points[0], points[1]))
        connections.add((points[1], points[2]))
        connections.add((points[2], points[0]))

    edges = []
    for start, end in connections:
        weight = distance(start, end)
        edges.append((start, end, weight))
    return edges

def minimum_spanning_tree(vertices, edges):
    parent_map = {vertex: vertex for vertex in vertices}

    def locate(vertex):
        if parent_map[vertex] != vertex:
            parent_map[vertex] = locate(parent_map[vertex])
        return parent_map[vertex]

    def connect(v1, v2):
        root1 = locate(v1)
        root2 = locate(v2)
        if root1 != root2:
            parent_map[root2] = root1

    mst_edges = []
    edges.sort(key=lambda edge: edge[2])

    for vertex1, vertex2, weight in edges:
        if locate(vertex1) != locate(vertex2):
            connect(vertex1, vertex2)
            mst_edges.append((vertex1, vertex2, weight))

    return mst_edges

def visualize_tree(vertices, tree):
    plt.figure(figsize=(8, 8))
    for vertex in vertices:
        plt.scatter(vertex.x, vertex.y, color='black')
    for start, end, _ in tree:
        plt.plot([start.x, end.x], [start.y, end.y], color='green')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def compute_mst_length(lambda_factor):
    vertices = [
        Coordinate(-1, 6),
        Coordinate(-1, -1),
        Coordinate(4, 7),
        Coordinate(6, 7),
        Coordinate(1, -1),
        Coordinate(-5, 3),
        Coordinate(-2, 3),
        Coordinate(2 - lambda_factor, 3)
    ]

    triangles = triangulate(vertices)
    edges = triangulation_to_graph(triangles)
    mst_edges = minimum_spanning_tree(vertices, edges)
    total_length = sum(edge[2] for edge in mst_edges)

    return total_length, mst_edges, vertices

def find_optimal_lambda():
    best_lambda = None
    shortest_length = float('inf')
    best_tree = []
    optimal_vertices = []

    start, end, resolution = -8, 7, 5000
    lambda_values = np.linspace(start, end, resolution)

    for lambda_val in lambda_values:
        length, mst_tree, points = compute_mst_length(lambda_val)
        if length < shortest_length:
            shortest_length = length
            best_lambda = lambda_val
            best_tree = mst_tree
            optimal_vertices = points

    print(f"Lambda for MST=4: {compute_mst_length(4)[0]:.2f}")
    print(f"Lambda for MST=7: {compute_mst_length(7)[0]:.2f}")
    print(f"Optimal Lambda: {best_lambda:.4f}, Minimum Length: {shortest_length:.2f}")

    visualize_tree(optimal_vertices, best_tree)

if __name__ == "__main__":
    find_optimal_lambda()
