import heapq
import math


class AStarPathfinder:
    """Grid-based A* with 8-direction movement and soft occupancy costs."""

    _NEIGHBORS = (
        (1, 0, 1.0),
        (1, 1, math.sqrt(2.0)),
        (0, 1, 1.0),
        (-1, 1, math.sqrt(2.0)),
        (-1, 0, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (0, -1, 1.0),
        (1, -1, math.sqrt(2.0)),
    )

    @staticmethod
    def _heuristic(a, b):
        # Octile distance for 8-direction grids.
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        dmin = min(dx, dy)
        dmax = max(dx, dy)
        return dmax + (math.sqrt(2.0) - 1.0) * dmin

    @staticmethod
    def _turn_penalty(prev_dir, new_dir):
        if prev_dir is None or prev_dir == new_dir:
            return 0.0
        dot = prev_dir[0] * new_dir[0] + prev_dir[1] * new_dir[1]
        if dot >= 0:
            return 0.1
        return 0.25

    def find_path(self, start, goal, grid_w, grid_h, chargers_grid, occupied_cells):
        sx, sy = int(start[0]), int(start[1])
        gx, gy = int(goal[0]), int(goal[1])
        start_node = (sx, sy)
        goal_node = (gx, gy)

        if start_node == goal_node:
            return []
        if not (0 <= sx < grid_w and 0 <= sy < grid_h and 0 <= gx < grid_w and 0 <= gy < grid_h):
            return []

        open_heap = []
        tie = 0
        g_score = {start_node: 0.0}
        parent = {}
        incoming = {start_node: None}

        heapq.heappush(open_heap, (self._heuristic(start_node, goal_node), tie, start_node))
        tie += 1

        closed = set()
        goal_is_charger = bool(chargers_grid[gy][gx])

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

            for dx, dy, step_cost in self._NEIGHBORS:
                nx = cx + dx
                ny = cy + dy
                if not (0 <= nx < grid_w and 0 <= ny < grid_h):
                    continue

                nbr = (nx, ny)
                if nbr in closed:
                    continue

                if chargers_grid[ny][nx] and not (goal_is_charger and nbr == goal_node):
                    continue

                soft_cost = 5.0 if nbr in occupied_cells else 0.0
                turn_cost = self._turn_penalty(prev_dir, (dx, dy))
                tentative = base_g + step_cost + soft_cost + turn_cost

                if tentative >= g_score.get(nbr, float("inf")):
                    continue

                parent[nbr] = current
                incoming[nbr] = (dx, dy)
                g_score[nbr] = tentative
                f = tentative + self._heuristic(nbr, goal_node)
                heapq.heappush(open_heap, (f, tie, nbr))
                tie += 1

        return []

    @staticmethod
    def _reconstruct_path(parent, goal_node):
        out = [goal_node]
        cur = goal_node
        while cur in parent:
            cur = parent[cur]
            out.append(cur)
        out.reverse()
        # Start node is first element; path consumers only need waypoints ahead.
        # Return every cell so movement can step cell-by-cell without relying
        # on heading→cardinal conversion (which caused flickering).
        return [[x, y] for x, y in out[1:]]
