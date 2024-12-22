import math
from itertools import combinations
from collections import defaultdict
import matplotlib.pyplot as plt

class Coordinate:
    def __init__(self, x_val, y_val):
        self.x_val = x_val
        self.y_val = y_val

    def __repr__(self):
        return f"Coordinate({self.x_val}, {self.y_val})"

class Simplex:
    def __init__(self, vertex_a, vertex_b, vertex_c):
        self.vertices = [vertex_a, vertex_b, vertex_c]

    def find_midpoint(self, vertex1, vertex2):
        return Coordinate((vertex1.x_val + vertex2.x_val) / 2,
                          (vertex1.y_val + vertex2.y_val) / 2)

    def bisect_edge(self, vertex1, vertex2):
        midpoint = self.find_midpoint(vertex1, vertex2)
        delta_x = vertex2.x_val - vertex1.x_val
        delta_y = vertex2.y_val - vertex1.y_val

        if delta_x == 0:
            perpendicular_slope = 0
        elif delta_y == 0:
            perpendicular_slope = None
        else:
            slope = delta_y / delta_x
            perpendicular_slope = -1 / slope

        return midpoint, perpendicular_slope

    def intersect_lines(self, mid1, slope1, mid2, slope2):
        if slope1 is None:
            x_coord = mid1.x_val
            y_coord = slope2 * (x_coord - mid2.x_val) + mid2.y_val
        elif slope2 is None:
            x_coord = mid2.x_val
            y_coord = slope1 * (x_coord - mid1.x_val) + mid1.y_val
        elif slope1 == slope2:
            return None
        else:
            x_coord = ((slope2 * mid2.x_val - slope1 * mid1.x_val) +
                       (mid1.y_val - mid2.y_val)) / (slope2 - slope1)
            y_coord = slope1 * (x_coord - mid1.x_val) + mid1.y_val

        return Coordinate(x_coord, y_coord)

    def circumcircle(self):
        V, W, U = self.vertices

        mid_vw, slope_vw = self.bisect_edge(V, W)
        mid_wu, slope_wu = self.bisect_edge(W, U)

        center = self.intersect_lines(mid_vw, slope_vw, mid_wu, slope_wu)
        if not center:
            return None, None

        radius = math.hypot(center.x_val - V.x_val, center.y_val - V.y_val)
        return center, radius

    def is_inside_circumcircle(self, point):
        center, radius = self.circumcircle()
        if not center:
            return False
        distance = math.hypot(point.x_val - center.x_val, point.y_val - center.y_val)
        return distance <= radius

    def __repr__(self):
        return f"Simplex({self.vertices[0]}, {self.vertices[1]}, {self.vertices[2]})"

def eliminate_duplicates(points, tolerance=1e-7):
    unique = []
    for pt in points:
        if not any(math.isclose(pt.x_val, existing.x_val, abs_tol=tolerance) and
                   math.isclose(pt.y_val, existing.y_val, abs_tol=tolerance)
                   for existing in unique):
            unique.append(pt)
    return unique

def are_collinear(v1, v2, v3, tol=1e-7):
    area = (v2.x_val - v1.x_val) * (v3.y_val - v1.y_val) - \
           (v2.y_val - v1.y_val) * (v3.x_val - v1.x_val)
    return abs(area) < tol

def create_large_simplex(bounds, delta):
    min_x, max_x, min_y, max_y = bounds
    return Simplex(
        Coordinate(min_x - 7 * delta, min_y - delta),
        Coordinate(min_x + delta, max_y + 7 * delta),
        Coordinate(max_x + 7 * delta, min_y - delta)
    )

def build_delaunay(points, eps=1e-7):
    clean_points = eliminate_duplicates(points, tolerance=eps)
    sorted_pts = sorted(clean_points, key=lambda p: p.x_val)

    min_x = min(p.x_val for p in sorted_pts)
    max_x = max(p.x_val for p in sorted_pts)
    min_y = min(p.y_val for p in sorted_pts)
    max_y = max(p.y_val for p in sorted_pts)
    delta = max(max_x - min_x, max_y - min_y)

    super_tri = create_large_simplex((min_x, max_x, min_y, max_y), delta)
    triangulation = [super_tri]

    for p in sorted_pts:
        bad_simplices = [s for s in triangulation if s.is_inside_circumcircle(p)]
        edge_counts = defaultdict(int)

        for simplex in bad_simplices:
            for edge in combinations(simplex.vertices, 2):
                sorted_edge = tuple(sorted(edge, key=lambda v: (v.x_val, v.y_val)))
                edge_counts[sorted_edge] += 1

        boundary = [edge for edge, count in edge_counts.items() if count == 1]

        for simplex in bad_simplices:
            triangulation.remove(simplex)

        for edge in boundary:
            if not are_collinear(edge[0], edge[1], p):
                new_simplex = Simplex(edge[0], edge[1], p)
                triangulation.append(new_simplex)

    final_triangulation = [
        s for s in triangulation
        if not any(v in super_tri.vertices for v in s.vertices)
    ]

    return final_triangulation

def extend_voronoi_edge(center, v1, v2, opposite_v, length=20):
    mid_x = (v1.x_val + v2.x_val) / 2
    mid_y = (v1.y_val + v2.y_val) / 2

    dx, dy = v2.y_val - v1.y_val, v1.x_val - v2.x_val
    norm = math.hypot(dx, dy)
    dx, dy = dx / norm, dy / norm

    far_x = mid_x + dx * length
    far_y = mid_y + dy * length
    far_point = Coordinate(far_x, far_y)

    cross_product_original = (v2.x_val - v1.x_val) * (opposite_v.y_val - v1.y_val) - \
                             (v2.y_val - v1.y_val) * (opposite_v.x_val - v1.x_val)
    cross_product_new = (v2.x_val - v1.x_val) * (far_point.y_val - v1.y_val) - \
                        (v2.y_val - v1.y_val) * (far_point.x_val - v1.x_val)

    if cross_product_original * cross_product_new > 0:
        far_x = mid_x - dx * length
        far_y = mid_y - dy * length
        far_point = Coordinate(far_x, far_y)

    return far_point

def generate_voronoi(pts, simplices):
    centers = []
    for simplex in simplices:
        center, _ = simplex.circumcircle()
        if center:
            centers.append(center)

    edge_map = defaultdict(list)
    for idx, simplex in enumerate(simplices):
        for edge in combinations(simplex.vertices, 2):
            sorted_edge = tuple(sorted(edge, key=lambda v: (v.x_val, v.y_val)))
            edge_map[sorted_edge].append(idx)

    voronoi = []
    for edge, indices in edge_map.items():
        if len(indices) == 2:
            t1, t2 = indices
            if t1 < len(centers) and t2 < len(centers):
                voronoi.append((centers[t1], centers[t2]))
        elif len(indices) == 1:
            t1 = indices[0]
            if t1 >= len(centers):
                continue
            v1, v2 = edge
            simplex = simplices[t1]
            for vertex in simplex.vertices:
                if vertex != v1 and vertex != v2:
                    opposite = vertex
                    break
            extended_point = extend_voronoi_edge(centers[t1], v1, v2, opposite, length=20)
            voronoi.append((centers[t1], extended_point))

    return voronoi

def plot_results(pts, simplices, voronoi_lines):
    plt.figure(figsize=(8, 8))
    for point in pts:
        plt.scatter(point.x_val, point.y_val, color='black')
    for simplex in simplices:
        x_coords = [vertex.x_val for vertex in simplex.vertices] + [simplex.vertices[0].x_val]
        y_coords = [vertex.y_val for vertex in simplex.vertices] + [simplex.vertices[0].y_val]
        plt.plot(x_coords, y_coords, color='blue')
    for line in voronoi_lines:
        x_vals = [line[0].x_val, line[1].x_val]
        y_vals = [line[0].y_val, line[1].y_val]
        plt.plot(x_vals, y_vals, color='green')
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Delaunay Triangulation and Voronoi Diagram')
    plt.legend(['Delaunay Triangulation', 'Voronoi Diagram'], loc='upper right')
    plt.show()

def run_exercise():
    sample_pts = [
        Coordinate(3, -5),
        Coordinate(-6, 6),
        Coordinate(6, -4),
        Coordinate(5, -5),
        Coordinate(9, 10)
    ]
    delaunay_simplices = build_delaunay(sample_pts)
    unique_edges = set()
    for simplex in delaunay_simplices:
        for edge in combinations(simplex.vertices, 2):
            sorted_edge = tuple(sorted(edge, key=lambda v: (v.x_val, v.y_val)))
            unique_edges.add(sorted_edge)
    total_edges = len(unique_edges)
    print("Exercise 1:")
    #print(f"Number of triangles in the triangulation: {len(delaunay_simplices)}")
   #print(f"Number of edges: {total_edges}")
    voronoi_segments = generate_voronoi(sample_pts, delaunay_simplices)
    plot_results(sample_pts, delaunay_simplices, voronoi_segments)

if __name__ == "__main__":
    run_exercise()
