from collections import deque
import heapq

from data.warehouse.pathfinder import AStarPathfinder


class LanePathPlanner:
    """High-level planner that prefers aisle/lane cells and bridges with short spurs."""

    _NEIGHBORS = (
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
    )

    def __init__(self):
        self._spur_planner = AStarPathfinder()

    @staticmethod
    def _heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start, goal, lane_map, chargers_grid):
        start_node = (int(start[0]), int(start[1]))
        goal_node = (int(goal[0]), int(goal[1]))
        if start_node == goal_node:
            return []

        grid_h, grid_w = lane_map.shape
        start_lane = self._nearest_lane_cell(start_node, lane_map, chargers_grid)
        goal_lane = self._nearest_lane_cell(goal_node, lane_map, chargers_grid)
        if start_lane is None or goal_lane is None:
            return []

        lane_backbone = self._lane_astar(start_lane, goal_lane, lane_map, chargers_grid)
        if not lane_backbone:
            return []

        start_spur = self._spur_path(start_node, start_lane, grid_w, grid_h, chargers_grid)
        goal_spur = self._spur_path(goal_lane, goal_node, grid_w, grid_h, chargers_grid)
        if start_spur is None or goal_spur is None:
            return []

        full_path = [list(start_node)]
        full_path.extend(start_spur[1:])
        full_path.extend([list(p) for p in lane_backbone[1:]])
        full_path.extend(goal_spur[1:])
        _expanded = self._expand_polyline(full_path)
        _smoothed = self._round_corners(_expanded, lane_map, chargers_grid)
        return _smoothed[1:]

    def _nearest_lane_cell(self, start_node, lane_map, chargers_grid):
        sx, sy = start_node
        grid_h, grid_w = lane_map.shape
        if lane_map[sy][sx] and not chargers_grid[sy][sx]:
            return start_node

        q = deque([start_node])
        seen = {start_node}
        while q:
            cx, cy = q.popleft()
            for dx, dy in self._NEIGHBORS:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < grid_w and 0 <= ny < grid_h):
                    continue
                node = (nx, ny)
                if node in seen or chargers_grid[ny][nx]:
                    continue
                if lane_map[ny][nx]:
                    return node
                seen.add(node)
                q.append(node)
        return None

    def _lane_astar(self, start_node, goal_node, lane_map, chargers_grid):
        if start_node == goal_node:
            return [start_node]

        open_heap = []
        tie = 0
        g_score = {start_node: 0.0}
        parent = {}
        incoming = {start_node: None}
        heapq.heappush(open_heap, (self._heuristic(start_node, goal_node), tie, start_node))
        tie += 1
        closed = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal_node:
                return self._reconstruct_path(parent, current)
            closed.add(current)

            cx, cy = current
            prev_dir = incoming.get(current)
            base_g = g_score[current]
            for dx, dy in self._NEIGHBORS:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < lane_map.shape[1] and 0 <= ny < lane_map.shape[0]):
                    continue
                nbr = (nx, ny)
                if nbr in closed or chargers_grid[ny][nx]:
                    continue
                if not lane_map[ny][nx] and nbr != goal_node:
                    continue
                turn_cost = 0.0 if prev_dir is None or prev_dir == (dx, dy) else 0.2
                tentative = base_g + 1.0 + turn_cost
                if tentative >= g_score.get(nbr, float('inf')):
                    continue
                parent[nbr] = current
                incoming[nbr] = (dx, dy)
                g_score[nbr] = tentative
                f = tentative + self._heuristic(nbr, goal_node)
                heapq.heappush(open_heap, (f, tie, nbr))
                tie += 1

        return []

    def _spur_path(self, start_node, goal_node, grid_w, grid_h, chargers_grid):
        if start_node == goal_node:
            return [list(start_node)]
        path = self._spur_planner.find_path(
            start_node,
            goal_node,
            grid_w,
            grid_h,
            chargers_grid,
            set(),
        )
        if not path and start_node != goal_node:
            return None
        return [list(start_node)] + [list(p) for p in path]

    @staticmethod
    def _expand_polyline(points):
        if not points:
            return []
        expanded = [list(points[0])]
        for idx in range(1, len(points)):
            x0, y0 = int(points[idx - 1][0]), int(points[idx - 1][1])
            x1, y1 = int(points[idx][0]), int(points[idx][1])
            dx = x1 - x0
            dy = y1 - y0
            step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
            step_y = 0 if dy == 0 else (1 if dy > 0 else -1)
            steps = max(abs(dx), abs(dy))
            for step in range(1, steps + 1):
                cell = [x0 + step_x * step, y0 + step_y * step]
                if cell != expanded[-1]:
                    expanded.append(cell)
        return expanded

    @staticmethod
    def _round_corners(points, lane_map, chargers_grid):
        """Replace strict L-corners with a diagonal cut when physically feasible.

        This reduces repeated 90-degree pivots that are hard for the current
        heading-constrained controller to track cleanly.
        """
        if len(points) < 3:
            return points

        h, w = lane_map.shape
        out = [list(points[0])]
        i = 1
        while i < len(points) - 1:
            px, py = int(out[-1][0]), int(out[-1][1])
            cx, cy = int(points[i][0]), int(points[i][1])
            nx, ny = int(points[i + 1][0]), int(points[i + 1][1])

            v1x, v1y = (cx - px), (cy - py)
            v2x, v2y = (nx - cx), (ny - cy)

            # Cardinal right-angle turn with unit steps.
            is_unit = (abs(v1x) + abs(v1y) == 1) and (abs(v2x) + abs(v2y) == 1)
            is_right_turn = (v1x * v2x + v1y * v2y) == 0

            if is_unit and is_right_turn:
                # Diagonal corner-cut target from previous point.
                dx = (1 if v1x > 0 else (-1 if v1x < 0 else 0)) + (1 if v2x > 0 else (-1 if v2x < 0 else 0))
                dy = (1 if v1y > 0 else (-1 if v1y < 0 else 0)) + (1 if v2y > 0 else (-1 if v2y < 0 else 0))
                cut_x, cut_y = px + dx, py + dy

                if (0 <= cut_x < w and 0 <= cut_y < h
                        and not chargers_grid[cut_y][cut_x]):
                    # Accept corner cut and skip strict corner vertex.
                    if out[-1] != [cut_x, cut_y]:
                        out.append([cut_x, cut_y])
                    i += 2
                    continue

            out.append([cx, cy])
            i += 1

        if out[-1] != list(points[-1]):
            out.append(list(points[-1]))
        return out

    @staticmethod
    def _reconstruct_path(parent, goal_node):
        out = [goal_node]
        cur = goal_node
        while cur in parent:
            cur = parent[cur]
            out.append(cur)
        out.reverse()
        return out