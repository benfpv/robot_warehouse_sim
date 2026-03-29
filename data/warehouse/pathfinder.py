import heapq
import math
import numpy as np


class AStarPathfinder:
    """Grid-based A* with 8-direction movement, road preference, and traffic awareness."""

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

    # Road preference: multiplier on step_cost for each road tier.
    # Lower = cheaper = preferred.  Off-road cells (tier 0) pay a penalty.
    _ROAD_COST = {
        0: 1.6,    # no road: 60% surcharge (discourages off-road)
        1: 1.1,    # branch: slight discount
        2: 0.7,    # collector: fast
        3: 0.6,    # arterial: fastest — strong magnet
    }
    # Traffic congestion: per-robot cost added when a cell is occupied.
    _TRAFFIC_OCCUPANCY_COST = 4.0

    @staticmethod
    def _heuristic(a, b):
        # Octile distance for 8-direction grids.
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        dmin = min(dx, dy)
        dmax = max(dx, dy)
        # Use 0.6 multiplier so heuristic stays admissible even with road discounts
        return (dmax + (math.sqrt(2.0) - 1.0) * dmin) * 0.6

    @staticmethod
    def _turn_penalty(prev_dir, new_dir):
        if prev_dir is None or prev_dir == new_dir:
            return 0.0
        dot = prev_dir[0] * new_dir[0] + prev_dir[1] * new_dir[1]
        if dot >= 0:
            return 0.1
        return 0.25

    def find_path(self, start, goal, grid_w, grid_h, chargers_grid, occupied_cells,
                  road_map=None, traffic_ema=None):
        sx, sy = int(start[0]), int(start[1])
        gx, gy = int(goal[0]), int(goal[1])
        start_node = (sx, sy)
        goal_node = (gx, gy)

        if start_node == goal_node:
            return []
        if not (0 <= sx < grid_w and 0 <= sy < grid_h and 0 <= gx < grid_w and 0 <= gy < grid_h):
            return []

        # Pre-compute normalised traffic congestion map (0..1 range)
        _has_traffic = (traffic_ema is not None
                        and isinstance(traffic_ema, np.ndarray)
                        and traffic_ema.max() > 0)
        if _has_traffic:
            _t_max = float(np.percentile(traffic_ema[traffic_ema > 0], 95)) if np.any(traffic_ema > 0) else 1.0
            _t_max = max(_t_max, 1e-9)

        _has_roads = road_map is not None and isinstance(road_map, np.ndarray)

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

                # Road preference: scale step cost by road tier
                if _has_roads:
                    tier = int(road_map[ny, nx])
                    road_mult = self._ROAD_COST.get(tier, self._ROAD_COST[0])
                else:
                    road_mult = 1.0
                effective_step = step_cost * road_mult

                # Occupancy: other robots on this cell
                soft_cost = self._TRAFFIC_OCCUPANCY_COST if nbr in occupied_cells else 0.0

                # Traffic congestion: avoid cells with heavy recent traffic
                if _has_traffic:
                    congestion = min(float(traffic_ema[ny, nx]) / _t_max, 1.0)
                    soft_cost += congestion * 1.5  # up to 1.5 extra cost on hottest cells

                turn_cost = self._turn_penalty(prev_dir, (dx, dy))
                tentative = base_g + effective_step + soft_cost + turn_cost

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
        return [[x, y] for x, y in out[1:]]
