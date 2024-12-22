import math
from itertools import combinations
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class Coordinate:
    def __init__(self, x_val, y_val):
        self.x_val = x_val
        self.y_val = y_val

    def __repr__(self):
        return f"Coordinate({self.x_val}, {self.y_val})"

    def __hash__(self):
        return hash((self.x_val, self.y_val))

    def __eq__(self, other):
        return (self.x_val, self.y_val) == (other.x_val, other.y_val)

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
            if slope2 is not None:
                y_coord = slope2 * (x_coord - mid2.x_val) + mid2.y_val
            else:
                return None
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
                triangulation.append(Simplex(edge[0], edge[1], p))

    final_triangulation = [
        s for s in triangulation
        if not any(v in super_tri.vertices for v in s.vertices)
    ]

    return final_triangulation

def generate_voronoi(simplices):
    centers = {}
    for idx, simplex in enumerate(simplices):
        center, _ = simplex.circumcircle()
        if center:
            centers[idx] = center

    edge_map = defaultdict(list)
    for idx, simplex in enumerate(simplices):
        for edge in combinations(simplex.vertices, 2):
            sorted_edge = tuple(sorted(edge, key=lambda v: (v.x_val, v.y_val)))
            edge_map[sorted_edge].append(idx)

    voronoi = []
    for edge, tris in edge_map.items():
        if len(tris) == 2:
            center1 = centers.get(tris[0])
            center2 = centers.get(tris[1])
            if center1 and center2:
                voronoi.append((center1, center2))
        elif len(tris) == 1:
            t1 = tris[0]
            v1, v2 = edge
            simplex = simplices[t1]
            opposite = next(v for v in simplex.vertices if v != v1 and v != v2)
            mid_x = (v1.x_val + v2.x_val) / 2
            mid_y = (v1.y_val + v2.y_val) / 2
            dx = v2.y_val - v1.y_val
            dy = v1.x_val - v2.x_val
            norm = math.hypot(dx, dy)
            dx, dy = dx / norm, dy / norm
            far_x = mid_x + dx * 20
            far_y = mid_y + dy * 20
            far_point = Coordinate(far_x, far_y)
            cross_original = (v2.x_val - v1.x_val) * (opposite.y_val - v1.y_val) - \
                             (v2.y_val - v1.y_val) * (opposite.x_val - v1.x_val)
            cross_new = (v2.x_val - v1.x_val) * (far_point.y_val - v1.y_val) - \
                        (v2.y_val - v1.y_val) * (far_point.x_val - v1.x_val)
            if cross_original * cross_new > 0:
                far_x = mid_x - dx * 20
                far_y = mid_y - dy * 20
                far_point = Coordinate(far_x, far_y)
            voronoi.append((centers[t1], far_point))
    return voronoi

def visualize(ax, coords, simplices, voronoi_edges, title):
    for coord in coords:
        ax.scatter(coord.x_val, coord.y_val, color='black', s=10)
    for simplex in simplices:
        x = [v.x_val for v in simplex.vertices] + [simplex.vertices[0].x_val]
        y = [v.y_val for v in simplex.vertices] + [simplex.vertices[0].y_val]
        ax.plot(x, y, color='blue', lw=0.5)
    for edge in voronoi_edges:
        x = [edge[0].x_val, edge[1].x_val]
        y = [edge[0].y_val, edge[1].y_val]
        ax.plot(x, y, color='green', lw=0.5)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')

def compare_sets():
    M1 = [
        Coordinate(0, 0),
        Coordinate(4, 0),
        Coordinate(2, 3),
        Coordinate(2, 1),
    ]

    M2 = [
        Coordinate(0, 0),
        Coordinate(4, 0),
        Coordinate(2, 3),
        Coordinate(3, 2),
        Coordinate(1, 2)
    ]

    tri_1 = build_delaunay(M1)
    tri_2 = build_delaunay(M2)

    voronoi_1 = generate_voronoi(tri_1)
    voronoi_2 = generate_voronoi(tri_2)

    edges_1 = len({tuple(sorted(edge, key=lambda c: (c.x_val, c.y_val))) for simplex in tri_1 for edge in combinations(simplex.vertices, 2)})
    edges_2 = len({tuple(sorted(edge, key=lambda c: (c.x_val, c.y_val))) for simplex in tri_2 for edge in combinations(simplex.vertices, 2)})

    print("First Set:")
    print("Triangles:", len(tri_1))
    print("Edges:", edges_1)

    print("Second Set:")
    print("Triangles:", len(tri_2))
    print("Edges:", edges_2)

    if len(tri_1) == len(tri_2) and len(M1) != len(M2):
        print("Two sets with different points result in the same triangle count.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    visualize(axes[0], M1, tri_1, voronoi_1, "First Set Triangulation with Voronoi")
    visualize(axes[1], M2, tri_2, voronoi_2, "Second Set Triangulation with Voronoi")

    line_style = [Line2D([0], [0], color='blue', lw=0.5), Line2D([0], [0], color='green', lw=0.5)]
    axes[0].legend(line_style, ['Triangulation', 'Voronoi'], loc='upper right')
    axes[1].legend(line_style, ['Triangulation', 'Voronoi'], loc='upper right')

    plt.show()

if __name__ == "__main__":
    compare_sets()
