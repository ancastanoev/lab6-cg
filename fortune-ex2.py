import math
import itertools
from itertools import combinations
from collections import defaultdict
import matplotlib.pyplot as plt

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

class Triangle:
    def __init__(self, p1, p2, p3):
        self.vertices = [p1, p2, p3]

    def midpoint(self, p1, p2):
        return Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)

    def perpendicular_bisector(self, p1, p2):
        mid = self.midpoint(p1, p2)
        if p2.x - p1.x == 0:
            slope = None
        else:
            slope = (p2.y - p1.y) / (p2.x - p1.x)
        if slope is None:
            perp_slope = 0
        elif slope == 0:
            perp_slope = None
        else:
            perp_slope = -1 / slope
        return mid, perp_slope

    def line_intersection(self, mid1, slope1, mid2, slope2):
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
        return Point(x, y)

    def compute_circumcenter_and_radius(self):
        A, B, C = self.vertices
        mid1, slope1 = self.perpendicular_bisector(A, B)
        mid2, slope2 = self.perpendicular_bisector(B, C)
        circumcenter = self.line_intersection(mid1, slope1, mid2, slope2)
        if circumcenter is None:
            return None, None
        circumradius = math.sqrt((circumcenter.x - A.x) ** 2 + (circumcenter.y - A.y) ** 2)
        return circumcenter, circumradius

    def contains_point_in_circumcircle(self, p):
        circumcenter, circumradius = self.compute_circumcenter_and_radius()
        if circumcenter is None:
            return False
        distance_squared = (p.x - circumcenter.x) ** 2 + (p.y - circumcenter.y) ** 2
        return distance_squared <= circumradius ** 2

    def __repr__(self):
        return f"Triangle({self.vertices[0]}, {self.vertices[1]}, {self.vertices[2]})"

def delaunay_triangulation(points, epsilon=1e-7):
    def is_coincident(p1, p2):
        return abs(p1.x - p2.x) < epsilon and abs(p1.y - p2.y) < epsilon

    unique_points = []
    for p in points:
        if not any(is_coincident(p, existing_point) for existing_point in unique_points):
            unique_points.append(p)
    points = unique_points

    def are_collinear(p1, p2, p3):
        return abs((p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)) < epsilon

    points.sort(key=lambda p: p.x)
    min_x, max_x = min(p.x for p in points), max(p.x for p in points)
    min_y, max_y = min(p.y for p in points), max(p.y for p in points)
    delta_max = max(max_x - min_x, max_y - min_y)
    big_triangle = Triangle(
        Point(min_x - 7 * delta_max, min_y - delta_max),
        Point(min_x + delta_max, max_y + 7 * delta_max),
        Point(max_x + 7 * delta_max, min_y - delta_max)
    )
    triangles = [big_triangle]
    for p in points:
        bad_triangles = [tri for tri in triangles if tri.contains_point_in_circumcircle(p)]
        polygon = []
        for tri in bad_triangles:
            for edge in combinations(tri.vertices, 2):
                sorted_edge = tuple(sorted(edge, key=lambda v: (v.x, v.y)))
                if sorted_edge in polygon:
                    polygon.remove(sorted_edge)
                else:
                    polygon.append(sorted_edge)
        for tri in bad_triangles:
            triangles.remove(tri)
        for edge in polygon:
            if not are_collinear(edge[0], edge[1], p):
                new_tri = Triangle(edge[0], edge[1], p)
                triangles.append(new_tri)
    triangles = [
        tri for tri in triangles
        if not any(v in big_triangle.vertices for v in tri.vertices)
    ]
    return triangles

def compute_voronoi_from_delaunay(points, delaunay_triangles):
    circumcenters = []
    for tri in delaunay_triangles:
        circumcenter, _ = tri.compute_circumcenter_and_radius()
        if circumcenter:
            circumcenters.append(circumcenter)
        else:
            circumcenters.append(None)
    edge_map = defaultdict(list)
    for idx, tri in enumerate(delaunay_triangles):
        for edge in combinations(tri.vertices, 2):
            sorted_edge = tuple(sorted(edge, key=lambda v: (v.x, v.y)))
            edge_map[sorted_edge].append(idx)
    def extend_boundary(circumcenter, p1, p2, p3, length=100):
        mid_x, mid_y = (p1.x + p2.x) / 2, (p1.y + p2.y) / 2
        dx, dy = p2.y - p1.y, p1.x - p2.x
        magnitude = math.hypot(dx, dy)
        if magnitude == 0:
            return circumcenter
        dx, dy = dx / magnitude, dy / magnitude
        far_point = Point(mid_x + dx * length, mid_y + dy * length)
        cross_product = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
        far_point_test = (p2.x - p1.x) * (far_point.y - p1.y) - (p2.y - p1.y) * (far_point.x - p1.x)
        if (cross_product * far_point_test) > 0:
            far_point = Point(mid_x - dx * length, mid_y - dy * length)
        return far_point
    voronoi_edges = []
    for edge, tris in edge_map.items():
        if len(tris) == 2:
            t1, t2 = tris
            cc1, cc2 = circumcenters[t1], circumcenters[t2]
            if cc1 and cc2:
                voronoi_edges.append((cc1, cc2))
        elif len(tris) == 1:
            t1 = tris[0]
            p1, p2 = edge
            cc = circumcenters[t1]
            if cc is None:
                continue
            p3 = next(p for p in delaunay_triangles[t1].vertices if p not in edge)
            far_point = extend_boundary(cc, p1, p2, p3)
            voronoi_edges.append((cc, far_point))
    return voronoi_edges

def plot_voronoi_diagram(points, voronoi_edges):
    plt.figure(figsize=(8, 8))
    x_points = [p.x for p in points]
    y_points = [p.y for p in points]
    plt.scatter(x_points, y_points, color='black', zorder=5)
    for edge in voronoi_edges:
        cc1, cc2 = edge
        if cc1 and cc2:
            plt.plot([cc1.x, cc2.x], [cc1.y, cc2.y], color='red', linewidth=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-15, 25)
    plt.ylim(-20, 20)
    plt.title('Voronoi Diagram')
    plt.legend(['Voronoi Diagram'], loc='upper right')
    plt.show()

def exercise_2():
    points_ex2 = [
        Point(5, -1),
        Point(7, -1),
        Point(9, -1),
        Point(7, -3),
        Point(11, -1),
        Point(-9, 3),
        Point(7, -4),
        Point(13, 5)
    ]
    triangulation_ex2 = delaunay_triangulation(points_ex2)
    voronoi_ex2 = compute_voronoi_from_delaunay(points_ex2, triangulation_ex2)
    plot_voronoi_diagram(points_ex2, voronoi_ex2)

if __name__ == "__main__":
    exercise_2()
