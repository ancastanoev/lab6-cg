import math
import itertools
from collections import defaultdict
import matplotlib.pyplot as plt

class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Coordinate({self.x}, {self.y})"

class Polygon:
    def __init__(self, a, b, c):
        self.corners = [a, b, c]

    def midpoint(self, p1, p2):
        return Coordinate((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)

    def perpendicular_bisector(self, p1, p2):
        mid = self.midpoint(p1, p2)
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        if dx == 0:
            return mid, 0
        elif dy == 0:
            return mid, None
        else:
            return mid, -1 / (dy / dx)

    def intersection(self, mid1, slope1, mid2, slope2):
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

    def circumcircle(self):
        p1, p2, p3 = self.corners
        mid1, slope1 = self.perpendicular_bisector(p1, p2)
        mid2, slope2 = self.perpendicular_bisector(p2, p3)
        center = self.intersection(mid1, slope1, mid2, slope2)
        if not center:
            return None, None
        radius = math.sqrt((center.x - p1.x) ** 2 + (center.y - p1.y) ** 2)
        return center, radius

    def contains(self, point):
        center, radius = self.circumcircle()
        if not center:
            return False
        dist_sq = (point.x - center.x) ** 2 + (point.y - center.y) ** 2
        return dist_sq <= radius ** 2

    def __repr__(self):
        return f"Polygon({self.corners[0]}, {self.corners[1]}, {self.corners[2]})"

def triangulation(vertices, epsilon=1e-7):
    def close(p1, p2):
        return abs(p1.x - p2.x) < epsilon and abs(p1.y - p2.y) < epsilon

    filtered = []
    for v in vertices:
        if not any(close(v, existing) for existing in filtered):
            filtered.append(v)

    vertices = filtered

    def collinear(p1, p2, p3):
        return abs((p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)) < epsilon

    vertices.sort(key=lambda v: v.x)

    min_x, max_x = min(v.x for v in vertices), max(v.x for v in vertices)
    min_y, max_y = min(v.y for v in vertices), max(v.y for v in vertices)
    delta = max(max_x - min_x, max_y - min_y)

    bounding_polygon = Polygon(
        Coordinate(min_x - 7 * delta, min_y - delta),
        Coordinate(min_x + delta, max_y + 7 * delta),
        Coordinate(max_x + 7 * delta, min_y - delta)
    )

    polygons = [bounding_polygon]

    for vertex in vertices:
        invalid = [poly for poly in polygons if poly.contains(vertex)]

        edges = []
        for poly in invalid:
            for edge in itertools.combinations(poly.corners, 2):
                ordered = tuple(sorted(edge, key=lambda p: (p.x, p.y)))
                if ordered in edges:
                    edges.remove(ordered)
                else:
                    edges.append(ordered)

        for poly in invalid:
            polygons.remove(poly)

        for edge in edges:
            if not collinear(edge[0], edge[1], vertex):
                polygons.append(Polygon(edge[0], edge[1], vertex))

    polygons = [
        poly for poly in polygons
        if not any(corner in bounding_polygon.corners for corner in poly.corners)
    ]

    return polygons

def voronoi_from_triangulation(vertices, polygons):
    centers = []
    for poly in polygons:
        center, _ = poly.circumcircle()
        centers.append(center)

    edge_connections = defaultdict(list)
    for i, poly in enumerate(polygons):
        for edge in itertools.combinations(poly.corners, 2):
            ordered = tuple(sorted(edge, key=lambda p: (p.x, p.y)))
            edge_connections[ordered].append(i)

    def boundary_extension(center, p1, p2, p3, length=100):
        mid_x, mid_y = (p1.x + p2.x) / 2, (p1.y + p2.y) / 2
        dx, dy = p2.y - p1.y, p1.x - p2.x
        magnitude = math.sqrt(dx ** 2 + dy ** 2)
        dx, dy = dx / magnitude, dy / magnitude
        extended_x = mid_x + dx * length
        extended_y = mid_y + dy * length
        far_point = Coordinate(extended_x, extended_y)
        cross_check = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
        direction_check = (p2.x - p1.x) * (far_point.y - p1.y) - (p2.y - p1.y) * (far_point.x - p1.x)
        if cross_check * direction_check > 0:
            far_point = Coordinate(mid_x - dx * length, mid_y - dy * length)
        return far_point

    edges = []
    for edge, connected in edge_connections.items():
        if len(connected) == 2:
            t1, t2 = connected
            edges.append((centers[t1], centers[t2]))
        elif len(connected) == 1:
            t1 = connected[0]
            p1, p2 = edge
            center = centers[t1]
            p3 = next(corner for corner in polygons[t1].corners if corner not in edge)
            extension = boundary_extension(center, p1, p2, p3)
            edges.append((center, extension))

    return edges

def display_voronoi(vertices, edges):
    plt.figure(figsize=(8, 8))
    for vertex in vertices:
        plt.scatter(vertex.x, vertex.y, color='black', s=10)
    for edge in edges:
        x_coords = [edge[0].x, edge[1].x]
        y_coords = [edge[0].y, edge[1].y]
        plt.plot(x_coords, y_coords, color='red', lw=0.5)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def task_four():
    points = [
        Coordinate(1, -1),
        Coordinate(0, 0),
        Coordinate(-1, 1),
        Coordinate(-2, 2),
        Coordinate(-3, 3),
        Coordinate(-4, 4),
        Coordinate(0, 0),
        Coordinate(1, -1),
        Coordinate(2, -2),
        Coordinate(3, -3),
        Coordinate(4, -4),
        Coordinate(5, -5),
        Coordinate(0, 0),
        Coordinate(0, 1),
        Coordinate(0, 2),
        Coordinate(0, 3),
        Coordinate(0, 4),
        Coordinate(0, 5)
    ]

    processed_data = triangulation(points)
    edges = voronoi_from_triangulation(points, processed_data)
    display_voronoi(points, edges)

if __name__ == "__main__":
    task_four()