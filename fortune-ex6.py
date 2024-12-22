import math
import itertools
from itertools import combinations
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Coordinate({self.x}, {self.y})"

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

class Polygon:
    def __init__(self, a, b, c):
        self.vertices = [a, b, c]

    def midpoint(self, p1, p2):
        return Coordinate((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)

    def bisector(self, p1, p2):
        mid = self.midpoint(p1, p2)
        if p2.x - p1.x == 0:
            slope = None
        else:
            slope = (p2.y - p1.y) / (p2.x - p1.x)
        if slope is None:
            perpendicular_slope = 0
        elif slope == 0:
            perpendicular_slope = None
        else:
            perpendicular_slope = -1 / slope
        return mid, perpendicular_slope

    def intersect(self, m1, s1, m2, s2):
        if s1 is None:
            x = m1.x
            y = s2 * (x - m2.x) + m2.y if s2 is not None else None
        elif s2 is None:
            x = m2.x
            y = s1 * (x - m1.x) + m1.y
        elif s1 == s2:
            return None
        else:
            x = (s2 * m2.x - s1 * m1.x + m1.y - m2.y) / (s2 - s1)
            y = s1 * (x - m1.x) + m1.y
        return Coordinate(x, y)

    def circumcenter(self):
        p1, p2, p3 = self.vertices
        mid1, slope1 = self.bisector(p1, p2)
        mid2, slope2 = self.bisector(p2, p3)
        center = self.intersect(mid1, slope1, mid2, slope2)
        if not center:
            return None, None
        radius = math.sqrt((center.x - p1.x)**2 + (center.y - p1.y)**2)
        return center, radius

    def inside_circle(self, point):
        center, radius = self.circumcenter()
        if not center:
            return False
        return (point.x - center.x)**2 + (point.y - center.y)**2 <= radius**2

    def __repr__(self):
        return f"Polygon({self.vertices[0]}, {self.vertices[1]}, {self.vertices[2]})"

def triangulate(points, epsilon=1e-7):
    def identical(p1, p2):
        return abs(p1.x - p2.x) < epsilon and abs(p1.y - p2.y) < epsilon

    unique = []
    for p in points:
        if not any(identical(p, q) for q in unique):
            unique.append(p)

    def collinear(p1, p2, p3):
        return abs((p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)) < epsilon

    unique.sort(key=lambda p: (p.x, p.y))
    min_x, max_x = min(p.x for p in unique), max(p.x for p in unique)
    min_y, max_y = min(p.y for p in unique), max(p.y for p in unique)
    delta = max(max_x - min_x, max_y - min_y)
    super_poly = Polygon(
        Coordinate(min_x - 7 * delta, min_y - delta),
        Coordinate(min_x + delta, max_y + 7 * delta),
        Coordinate(max_x + 7 * delta, min_y - delta)
    )
    polys = [super_poly]

    for point in unique:
        invalid = [poly for poly in polys if poly.inside_circle(point)]
        edges = []
        for poly in invalid:
            for edge in itertools.combinations(poly.vertices, 2):
                sorted_edge = tuple(sorted(edge, key=lambda v: (v.x, v.y)))
                if sorted_edge in edges:
                    edges.remove(sorted_edge)
                else:
                    edges.append(sorted_edge)
        for poly in invalid:
            polys.remove(poly)
        for edge in edges:
            if not collinear(edge[0], edge[1], point):
                polys.append(Polygon(edge[0], edge[1], point))

    return [poly for poly in polys if not any(v in super_poly.vertices for v in poly.vertices)]

def visualize_triangulation(points, polys):
    plt.figure(figsize=(8, 8))
    for p in points:
        plt.plot(p.x, p.y, 'o', color='black', markersize=6)
    for poly in polys:
        x_vals = [v.x for v in poly.vertices] + [poly.vertices[0].x]
        y_vals = [v.y for v in poly.vertices] + [poly.vertices[0].y]
        plt.plot(x_vals, y_vals, '-', color='blue', linewidth=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def task_six():
    scale_factor = 1.5
    original = [
        Coordinate(1, 1),
        Coordinate(1, -1),
        Coordinate(-1, -1),
        Coordinate(-1, 1),
        Coordinate(0, -2),
        Coordinate(0, 1.5 * scale_factor)
    ]

    scaled = [Coordinate(p.x * scale_factor, p.y * scale_factor) for p in original]
    tri_original = triangulate(original)
    tri_scaled = triangulate(scaled)

    edges_original = len(
        set(tuple(sorted(edge, key=lambda v: (v.x, v.y)))
            for poly in tri_original
            for edge in combinations(poly.vertices, 2))
    )

    edges_scaled = len(
        set(tuple(sorted(edge, key=lambda v: (v.x, v.y)))
            for poly in tri_scaled
            for edge in combinations(poly.vertices, 2))
    )

    print(f"Scaling factor applied: {scale_factor}\n")
    print("Original Points:")
    print("Triangles:", len(tri_original))
    print("Edges:", edges_original)
    print("\nScaled Points:")
    print("Triangles:", len(tri_scaled))
    print("Edges:", edges_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for p in original:
        axes[0].plot(p.x, p.y, 'o', color='black', markersize=6)
    for poly in tri_original:
        x_vals = [v.x for v in poly.vertices] + [poly.vertices[0].x]
        y_vals = [v.y for v in poly.vertices] + [poly.vertices[0].y]
        axes[0].plot(x_vals, y_vals, '-', color='blue', linewidth=1)
    axes[0].set_xlim(-10, 10)
    axes[0].set_ylim(-10, 10)
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_title('Original Triangulation')

    for p in scaled:
        axes[1].plot(p.x, p.y, 'o', color='black', markersize=6)
    for poly in tri_scaled:
        x_vals = [v.x for v in poly.vertices] + [poly.vertices[0].x]
        y_vals = [v.y for v in poly.vertices] + [poly.vertices[0].y]
        axes[1].plot(x_vals, y_vals, '-', color='blue', linewidth=1)
    axes[1].set_xlim(-10, 10)
    axes[1].set_ylim(-10, 10)
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_title(f'Scaled Triangulation (Scale={scale_factor})')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    task_six()
