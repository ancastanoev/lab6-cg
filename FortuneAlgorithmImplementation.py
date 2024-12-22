import math
import heapq
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt


@dataclass(order=True)
class Event:
    y: float
    x: float
    event_type: str
    point: Optional[Tuple[float, float]] = field(compare=False, default=None)
    arc: Optional['Arc'] = field(compare=False, default=None)
    center: Optional[Tuple[float, float]] = field(compare=False, default=None)


@dataclass
class Arc:
    point: Tuple[float, float]
    prev: Optional['Arc'] = None
    next: Optional['Arc'] = None
    circle_event: Optional[Event] = None


@dataclass
class Vertex:
    x: float
    y: float


@dataclass
class Edge:
    start: Vertex
    end: Optional[Vertex] = None


class FortuneAlgorithm:
    def __init__(self, points: List[Tuple[float, float]]):
        self.points = sorted(points, key=lambda p: (-p[1], p[0]))  # we sort by descending y
        self.event_queue: List[Event] = []
        self.beach_line: Optional[Arc] = None
        self.vertices: List[Vertex] = []
        self.edges: List[Edge] = []
        self.sweep_line = float('inf')  # start with sweep line at infinity

        # initialize event queue with site events
        for point in self.points:
            event = Event(y=point[1], x=point[0], event_type='site', point=point)
            heapq.heappush(self.event_queue, event)

    def run(self):
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.sweep_line = event.y

            if event.event_type == 'site':
                self.handle_site_event(event)
            elif event.event_type == 'circle':
                self.handle_circle_event(event)

    def handle_site_event(self, event: Event):
        new_arc = Arc(point=event.point)

        if not self.beach_line:
            self.beach_line = new_arc
            return

        # find the arc above the new site
        arc = self.find_arc(event.point)
        if not arc:
            return

        # split the arc into two
        new_arc.prev = arc
        new_arc.next = arc.next
        if arc.next:
            arc.next.prev = new_arc
        arc.next = new_arc

        # check for circle events
        self.check_circle_event(arc.prev, arc, new_arc)
        self.check_circle_event(new_arc, arc, arc.next)

    def handle_circle_event(self, event: Event):
        arc = event.arc
        if not arc.prev or not arc.next:
            return

        # create a vertex at the circle event
        vertex = Vertex(x=event.center[0], y=event.center[1])
        self.vertices.append(vertex)

        # close the edges
        if arc.prev:
            for edge in self.edges:
                if edge.end is None and edge.start == Vertex(*arc.prev.point):
                    edge.end = vertex
                    break

        if arc.next:
            for edge in self.edges:
                if edge.end is None and edge.start == Vertex(*arc.next.point):
                    edge.end = vertex
                    break

        # remove the arc from the beach line
        arc.prev.next = arc.next
        arc.next.prev = arc.prev

        # check for new circle events
        self.check_circle_event(arc.prev.prev, arc.prev, arc.next)
        self.check_circle_event(arc.prev, arc.next, arc.next.next)

    def find_arc(self, point: Tuple[float, float]) -> Optional[Arc]:
        # simple linear search for the arc above the new point
        arc = self.beach_line
        while arc:
            intersect = self.get_intersection(arc, point)
            if intersect is not None and intersect <= point[0]:
                return arc
            arc = arc.next
        return None

    def get_intersection(self, arc: Arc, point: Tuple[float, float]) -> Optional[float]:
        # calculate the intersection x-coordinate of the arc's parabola with the vertical line of the new point
        if self.sweep_line == arc.point[1]:
            return arc.point[0]

        a = 1 / (2 * (arc.point[1] - self.sweep_line))
        b = -arc.point[0] / (arc.point[1] - self.sweep_line)
        c = (arc.point[0] ** 2 + arc.point[1] ** 2 - self.sweep_line ** 2) / (2 * (arc.point[1] - self.sweep_line))

        discriminant = b ** 2 - 4 * a * (c - point[0])

        if discriminant < 0:
            return None

        sqrt_disc = math.sqrt(discriminant)
        x1 = (-b + sqrt_disc) / (2 * a)
        x2 = (-b - sqrt_disc) / (2 * a)

        return min(x1, x2)

    def check_circle_event(self, arc1: Optional[Arc], arc2: Optional[Arc], arc3: Optional[Arc]):
        if not arc1 or not arc2 or not arc3:
            return

        if are_collinear(arc1.point, arc2.point, arc3.point):
            return

        center, radius = circumcircle(arc1.point, arc2.point, arc3.point)
        if not center:
            return

        if center[1] >= self.sweep_line:
            return

        # create a circle event
        event = Event(y=center[1], x=center[0], event_type='circle', center=center, arc=arc2)
        arc2.circle_event = event
        heapq.heappush(self.event_queue, event)

    def plot_voronoi(self):
        fig, ax = plt.subplots(figsize=(8, 8))

        # plot Voronoi edges
        for edge in self.edges:
            if edge.end:
                ax.plot([edge.start.x, edge.end.x], [edge.start.y, edge.end.y], 'g-')

        # plot points
        for point in self.points:
            ax.plot(point[0], point[1], 'ko')
            ax.text(point[0] + 0.2, point[1] + 0.2, f"P{self.points.index(point) + 1}", fontsize=12)

        # plot vertices
        for vertex in self.vertices:
            ax.plot(vertex.x, vertex.y, 'ro')

        ax.set_xlim(min(p[0] for p in self.points) - 5, max(p[0] for p in self.points) + 5)
        ax.set_ylim(min(p[1] for p in self.points) - 5, max(p[1] for p in self.points) + 5)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Voronoi Diagram (Simplified Fortune's Algorithm)")
        plt.grid(True)
        plt.show()


def are_collinear(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float], tol=1e-7) -> bool:
    """Check if three points are collinear."""
    area = (p2[0] - p1[0]) * (p3[1] - p1[1]) - \
           (p2[1] - p1[1]) * (p3[0] - p1[0])
    return abs(area) < tol


def circumcircle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> Optional[
    Tuple[Tuple[float, float], float]]:

    ax, ay = p1
    bx, by = p2
    cx, cy = p3

    A = bx - ax
    B = by - ay
    C = cx - ax
    D = cy - ay

    E = A * (ax + bx) + B * (ay + by)
    F = C * (ax + cx) + D * (ay + cy)
    G = 2 * (A * (cy - by) - B * (cx - bx))

    if G == 0:
        return None

    center_x = (D * E - B * F) / G
    center_y = (A * F - C * E) / G
    radius = math.hypot(center_x - ax, center_y - ay)
    return (center_x, center_y), radius



if __name__ == "__main__":

    points = [
        (3, -5),  # Point A
        (-6, 6),  # Point B
        (6, -4),  # Point C
        (5, -5),  # Point D
        (9, 10)  # Point E
    ]


    fa = FortuneAlgorithm(points)
    fa.run()
    fa.plot_voronoi()
