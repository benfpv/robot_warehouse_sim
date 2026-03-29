# Robot Warehouse Sim

A real-time autonomous warehouse simulation. Robots dynamically pick up, transport, and deliver packages across import, storage, and export zones — managing battery, charging, and deadline priorities — all rendered in a live composite dashboard window.

https://github.com/user-attachments/assets/5e9a1aa5-320e-4aa3-a2d9-9d493e5cc5b6

---

## Features

- **Warehouse zones** — three zone types (import / storage / export) painted on an 80×70 grid; packages enter via import and exit when exported from the export zone before their deadline.  Zones can be painted interactively or loaded from a PNG map file.
- **Adaptive zone optimizer** — monitors per-zone utilization via EMA and automatically swaps boundary cells between zone types to rebalance capacity.  Zone-swap only: no zone is ever eliminated and no new zones are created from neutral space.
- **Packages** — each has a unique ID, real-time position & status, a deadline-to-export (10–90 s), colour-coded urgency, age tracking (`createdAt`), and a full import-to-export history log.  Urgency-based routing sends near-deadline packages to export and others to storage.
- **Robots** — up to 60 robots, each with a priority-based action queue, A*-based pathfinding with speed profiling, dynamic battery depletion with an adaptive Power Policy, automatic charging dispatch, age tracking (`createdAt`), live fleet scaling via `[−]`/`[+]` buttons (decommissioned robots return to their birth location), and single-package carrying capacity.
- **Chargers** — up to 10 charging stations; robots are dispatched when battery falls below a Power Policy threshold (profile-dependent range, roughly 27–50 %, adapts to charger pressure and uses extra damping/smoothing to avoid abrupt flips). Charger exclusivity is enforced via action-queue ground-truth checks with per-tick stale-reservation reconciliation and an arrival guard.
- **Flow control** — adaptive import cap scaled to fleet size (`pipeline_depth × n_robots`).  Idle-robot bonus pulls more work when capacity is available; overdue penalty backs off only when robots are genuinely busy; 15 %-per-tick smoothing ramps quickly; emergency floor jumps cap if majority of robots idle.  Ceiling is `n_robots × 10`.
- **Dynamic planning depth** — the package move pipeline is capped at `available_robots × 3` (floor 6) to keep the queue fed without over-planning when many robots are charging or busy.
- **Task scheduler** — the warehouse assigns packages to robots deadline-first, respects zone capacity, and avoids duplicate movement effort via `packagesMoveList` / `packagesMovingList`.
- **Physarum-inspired road network** — an organic, traffic-driven road builder observes robot traffic via lifetime + EMA heatmaps, detects hotspots, routes A* paths between them, and classifies the resulting network into three tiers (branch / collector / arterial) with plaza detection around charger clusters.  Roads are rebuilt periodically and erased/restored as zones change.
- **Traffic-aware pathfinding** — A* pathfinder applies per-tier road preference multipliers (arterial 0.6×, collector 0.7×, branch 1.1×, off-road 1.6×), real-time traffic congestion costs, turn penalties, and soft robot-occupancy costs to produce natural-looking, road-preferring routes.
- **Road & plaza overlays** — subtle road tier tints and warm-amber plaza tints are composited on the main view and the heatmap sub-view; toggleable in real time with keyboard shortcuts (`H` heatmap, `R` roads, `P` plazas).
- **Sim / draw decoupling** — simulation ticks at 40 Hz; display repaints at 15 fps independently to save CPU.
- **Rolling chart histories** — 1 800 samples at 1 sample/s (~30 min), with a dynamic time-window that grows from 30 s to the full history length.
- **Overflow-safe display** — all number fields use `_n()` (K/M abbreviation), `_fmt_elapsed()` / `_fmt_age()` (decade-aware), `_fmt_rate()` (K/M/s), and a 3-tier cascade (full → drop rate suffix → char-trim + …). The sim is designed to run for days, months, or years without text bleeding out of any panel.

## Dashboard Layout (960 × 680 px)

```
┌──────────────────────┬─────────┬─────────┬─────────┬──────────┐
│                      │Chargers │Packages │Zone Map │Deadlines │  ← sub-views (160×140)
│   Main view          ├─────────┼─────────┼─────────┤Fleet Batt│
│   (320×280, 4× zoom) │ Robots  │Pkg Tgts │Heatmap  │Flow Ctrl │
├──────────┬───────────┴─────────┼─────────┼─────────┴──────────┤
│  ROBOTS  │  CHARGERS           │SIM STATS│  PACKAGES          │  ← info panel 4×240px
├──────────┴──────────┬──────────┴─────────┬──────────┬─────────┤
│ Exports/Overdue     │ Flow Control       │Zone Bars │ Fleet   │  ← chart strip 4×240px
└─────────────────────┴────────────────────┴──────────┴─────────┘
```

**Main view colour legend:**

| Colour | Meaning |
|--------|---------|
| Dark green area | Import zone |
| Dark blue area | Storage zone |
| Dark red area | Export zone |
| Amber dots (perimeter) | Charging stations |
| White dots | Robots |
| Slate-blue dots | Idle packages (on-time, no assignment) |
| Lime-green dots | Packages with an assigned robot & target (move planned) |
| Cyan dots | Packages being carried by a robot |
| Magenta dots | Overdue packages with no plan |
| Pink dots | Overdue packages assigned/en-route |
| Lavender dots | Export zone, idle, deadline 10–60 s |
| Bright lavender dots | Export zone, idle, deadline < 10 s |
| Faint white tint (main view) | Road network overlay (branch / collector / arterial, increasing brightness) |
| Warm amber tint (main view) | Plaza overlay (charger-cluster gathering zones) |

**Info panel** (4 columns × 240 px) shows:
- **ROBOTS** — top-7 active robots (sorted by task urgency) + top-7 lowest-battery robots, colour-coded by state. Includes `[−]` / `[+]` buttons for live fleet scaling.
- **CHARGERS** — charger status + fleet Power Policy summary (avg battery, drain, smoothed/raw pressure, smoothed/raw threshold, working/charging/idle counts), plus mode buttons (`ECO`, `BAL`, `PERF`) for live power-policy override.
- **SIM STATS** — elapsed time, loop count, exports, package count, robot count, zone fill levels with rates, zone utilization, optimizer status, and flow-policy buttons (`STDY`, `BAL`, `THRU`) for live flow-control override.
- **PACKAGES** — all packages sorted by deadline urgency with zone, target, carrier, status, weight, item name and destination.

All text is clipped to its column boundary with overflow protection.

**Keyboard shortcuts:**

| Key | Action |
|-----|--------|
| `H` | Toggle traffic heatmap overlay |
| `R` | Toggle road network overlay |
| `P` | Toggle plaza overlay |

**Chart strip** (4 charts × 240 px) shows a dynamically growing time window (starts at 30 s, expands to 30 min):
- *Exports & Overdue* — cumulative exports (+ rate), total late (+ rate), and current instantaneous overdue count (+ rate)
- *Flow Control* — import cap, throughput/min, avg delivery time, overdue % (fixed 0–100 scale)
- *Zone Capacity* — live fill bars for import / storage / export with count, rate, and occupancy percentage
- *Fleet Health* — mean battery % (+ rate), working robots (actively delivering/moving), robots charging (+ rate), idle count

**Top-right panels:**
- *Deadlines* — 5-band urgency histogram (OVR/CRIT/URG/NRML/CMFT) with per-band count and percentage
- *Fleet Batt* — horizontal bar chart showing every robot's battery level sorted ascending, with status indicators (charging/en-route/saving/carrying), a dynamic threshold line, and a compact Power Policy readout
- *Flow Ctrl* — live flow-control readout with cap-utilisation bar, package-count bar, pipeline depth, pkg/bot ratio, idle/overdue counts, throughput/min, and avg delivery time

---

## Requirements

- Python 3.9+
- See [requirements.txt](requirements.txt)

```
pip install -r requirements.txt
```

> `ctypes` (standard library, Windows) is used for the borderless Win32 window. The sim runs on Windows only in its current form.

---

## Quick Start

```bash
python main.py
```

Press **Ctrl-C** in the terminal to exit (the window is borderless/frameless — there is no close button by default).

---

## Configuration

All key parameters live in two places:

**`main.py` — `MainGame.__init__`**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `warehouse_res` | `(80, 70)` | Grid cell dimensions (width × height) |
| `warehouse_windowRes` | `(320, 280)` | Display size of main view (4× upscale) |
| `sim_frametime` | `1/40` | Simulation tick rate (s) |
| `draw_frametime` | `1/15` | Display repaint rate (s) |
| `panel_h` | `165` | Info panel height (px) |
| `chart_h` | `95` | Chart strip height (px) |

**`data/warehouse/warehouse.py` — `Warehouse.__init__`**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `robotsMaxQuantity` | `60` | Max robots in simulation |
| `chargersMaxQuantity` | `10` | Max charging stations |
| `packagesMaxMoveQuantity` | `120` | Max packages in the delivery pipeline at once |
| `robotsTaskAssignmentMaxQuantity` | `3` | Max queued tasks per robot |
| `_pkg_target_mode` | `nearest` | Target placement mode inside a destination zone (`random`, `nearest`, `zone_edge`) |
| `_power_policy_mode` | `balanced` | Power policy profile (`eco`, `balanced`, `performance`) |
| `_flow_pipeline_depth` | `4` | Packages per robot in the delivery pipeline (adjusted by flow policy) |

Zone geometry (pad, thirds) is set in `data/warehouse/warehouse_init.py` → `init_zoneMap`.

## Package Target Modes

Package target placement inside a zone is live-switchable from the button group in the `PKGS` panel.

- `random` spreads packages across the zone. This is best when the warehouse is dense or congested and you want to avoid edge clustering.
- `nearest` picks the closest free cell in the target zone. This is best when minimizing immediate travel distance matters more than even distribution.
- `zone_edge` picks the closest free cell on the target zone boundary. This is best for pipeline flow, because packages stage near the next likely handoff between import, storage, and export.

The active mode also appears in the `SIM STATS` panel as `target:`, with a short rationale hint: `spread`, `min dist`, or `pipeline`.

## Strategy Override Buttons

Three strategy groups are now UI-selectable directly from the dashboard:

- **Power Policy** (`CHARGERS` header): `ECO`, `BAL`, `PERF`
- `ECO`: larger battery buffers and earlier charging
- `BAL`: default profile
- `PERF`: lower buffers for higher utilization
- **Flow Policy** (`SIM STATS` header): `STDY`, `BAL`, `THRU`
  - `STDY`: lower pipeline depth (3 pkg/bot) for stability
  - `BAL`: default pipeline depth (4 pkg/bot)
  - `THRU`: higher pipeline depth (6 pkg/bot) for throughput
- **Package Target Mode** (`PACKAGES` header): `RND`, `NEAR`, `EDGE`
- Controls placement strategy inside the destination zone
- **Robot Count** (`ROBOTS` header): `[−]`, `[+]`
  - `[−]`: decommission one robot (finishes tasks, returns to birth location, exits)
  - `[+]`: spawn one new robot on the perimeter

---

## Project Structure

```
robot_warehouse_sim/
├── main.py                          # Entry point, MainGame, composite display, charts
├── cvplt.py                         # Minimal OpenCV line-plot utility
├── requirements.txt
├── resources/
│   ├── list_items.csv               # Item catalogue loaded at startup
│   ├── list_addresses.csv           # Address list for package origin/destination
│   ├── zone_map.png                 # (auto-generated) zone layout loaded at startup
│   ├── charger_map.png              # (optional) custom charger spawn positions
│   └── robot_map.png                # (optional) custom robot spawn positions
└── data/
    ├── functions.py                 # General utility functions (grid, perimeter, cardinal)
    ├── functions_timeseries.py      # Rolling array helpers
    ├── importer.py                  # CSV → Item / Address object loaders
    ├── item.py                      # Item dataclass
    ├── address.py                   # Address dataclass
    ├── display/
    │   └── display_functions.py     # Borderless Win32 window setup (Win32 API via ctypes)
    ├── draw/
    │   └── draw_warehouse.py        # OpenCV draw helpers (zones, robots, packages, arrows)
    ├── paint/
    │   ├── paint_handler.py         # Interactive zone painting UI (mouse + keyboard)
    │   ├── map_importer.py          # PNG zone/spawn map loader and example generators
    │   ├── zone_strategy.py         # Adaptive zone rebalancing (zone-swap only)
    │   └── warehouse_optimizer.py   # Strategy orchestrator (tick budget, strategy dispatch)
    └── warehouse/
        ├── warehouse.py             # Core simulation engine — tick loop, flow control, scheduling
        ├── warehouse_init.py        # Grid / zone / lane map initialisation
        ├── warehouse_data.py        # Rolling timeseries data container
        ├── warehouse_log.py         # Log record classes (Packages_Log, Robots_Log)
        ├── pathfinder.py            # A* pathfinder with road preference and traffic awareness
        ├── road_builder.py          # Physarum-inspired organic road network builder
        ├── package.py               # Package dataclass
        ├── package_functions.py     # Package spawn, target selection, deadline generation
        ├── charger.py               # Charger dataclass
        ├── charger_functions.py     # Charger spawn and placement
        ├── robot.py                 # Robot dataclass with motion model
        └── robot_functions.py       # Robot spawn and generation
```

---

## Additional Systems

### Decommission

Robots can be retired gracefully via `set_robot_count()` or the `[−]` / `[+]` buttons on the ROBOTS header.  The system flags excess robots as `decommissioning`, lets them finish current tasks, then pathfinds back to the robot's original spawn location (`birthLocation`) and removes them from the simulation.  If no birth location is available it falls back to the nearest perimeter cell.  Flow configuration rescales automatically.

### Idle Parking

Robots that finish all tasks and remain in a zone (import/storage/export) are sent to the nearest neutral (`ZONE_NONE`) cell at end-of-tick.  This prevents idle robots from blocking zone slots.

### Delivery Cooldown

After a package is dropped off, a 2-second cooldown prevents instant re-pickup.  This avoids the ping-pong problem where a robot drops a package in storage then immediately picks it up again.

### Head-on Deadlock Avoidance

Two mechanisms: (1) A* pathfinder treats other robot positions as soft-cost cells (cost 4.0), routing around occupied paths where possible.  (2) When two robots detect a head-on collision, the lower-numbered robot yields by stepping perpendicular.

### Pathfinding

Dense A* pathfinder: 8-direction, octile heuristic (0.6× for admissibility with road discounts), every-cell waypoints.  Speed profiling annotates each waypoint with a planned speed based on turn severity (corner-only limits: ≥120°→20%, ≥90°→28%, ≥60°→40%, ≥25°→52% of maxVelocity).  Road preference multipliers steer robots toward higher-tier roads; traffic congestion costs (up to 1.5) discourage heavily used cells.

### Traffic-Aware Road Network

A Physarum-inspired organic road builder runs periodically (gated on all robots having charged at least once):

1. **Traffic observation** — cumulative lifetime traffic (`traffic_total`) and an exponential moving average (`traffic_ema`, 60 s half-life) are recorded per cell.
2. **Hotspot detection** — Gaussian blur + non-maximum suppression finds traffic centroids (max 10).
3. **MST + kNN routing** — minimum spanning tree plus 1 nearest-neighbour shortcut edge connects hotspots; A* paths are computed frame-by-frame (3 per tick) with a flat turn penalty (1.8) and Ramer–Douglas–Peucker smoothing (ε = 2.5).
4. **Tier classification** — accumulated flow determines branch (50th pct), collector (70th pct), and arterial (90th pct) tiers.
5. **Road widening** — genuinely thin (1-cell-wide) segments are widened to exactly 2 cells.
6. **Plaza detection** — dense traffic clusters and charger-adjacent areas are merged into plaza zones (warm-amber tint).
7. **Zone erasure / restoration** — roads inside zones are erased; roads in neutral space are preserved; displaced robots are evicted.

---

## Known Issues

- When the export zone fills completely, already-planned package movements are not cancelled or re-prioritised in real time; robots may continue heading toward a full export zone until the next scheduler pass.
- Robots are assigned packages deadline-first without considering proximity; a robot far from a package may be assigned over a nearer idle robot.

## Limitations

- Windows-only: borderless window setup uses Win32 via `ctypes`.
- No obstacle support: the grid is fully traversable with no walls or blocked cells.
- Zone painting and auto-zoning operate on a flat 2D grid; non-rectangular or multi-level layouts are not supported.

## Future Directions

### Scheduling & Capacity Optimisation
- Optimisation calculations for managing zone capacities, fleet sizes, and time-based throughput targets.
- Dynamic priority switching when export is full: cancel or re-route in-flight package movements based on real-time zone capacities rather than waiting for the next scheduler tick.
- Dynamically adaptive zone shapes and sizes — e.g., complex non-rectangular footprints (spirals, L-shapes, multi-level). In real life, a 3D helical layout could exploit gravity to move packages passively from import → storage → export.
- Dispatch robots to the nearest available charger rather than any idle charger.
- Allow swapping lower-priority packages out of export to unblock higher-priority ones.
- **Package location swap** — operator-initiated or scheduler-driven swap of two packages' grid positions (e.g. re-slot a high-urgency package into a more accessible cell), dispatching the minimum robot pair needed to exchange them atomically.

### Traffic & Fleet Navigation
- Collision avoidance between robots.
- 3rd-party fleet navigation/traffic/planning management integration.
- Dynamic real-time optimisation solver for fleet assignment.
- AI / LLM / VLM-powered real-time warehouse management layer.
- Power-saving mode (reduce tick rate when throughput demand is low).
- Human-hybrid mode (human operators co-exist with autonomous robots).
- Additional vehicle types (forklifts, conveyor belts, drones, etc.).

### Fleet Characteristics
- Robot health, defects, wear, scheduled maintenance, repair, and recovery simulation.
- Robot-carries-robot recovery for dead/stranded units.
- Varied robot types: different movement styles, speeds, carrying capacities, and navigation systems.
- Per-robot cost model (purchase, energy, maintenance).
- Movement type/style simulation: wheeled, legged, aerial.
- Individual robot navigation and sensory/vision system simulation.
- Dynamic task-priority counts per robot (currently fixed at 3).
- AI / LLM / VLM-based individual robot decision-making.

### Package Characteristics
- Package defects, damage states, untracked/lost packages.
- Illegal or flagged items, scanning and classification systems.
- Weight, size, and volume constraints affecting robot assignment.
- Package classes/types with different handling requirements.

### Warehouse Infrastructure
- Warehouse infrastructure health, defects, and repair simulation.
- Dynamic upsizing and downsizing of the warehouse footprint.
- Renovation and construction events (temporary zone closures, new sections).
- Destruction/incident simulation and recovery.

### Interfaces & Display
- Dark mode, light mode, and high-contrast accessibility modes.
- Graphics/display optimisation for local system (async rendering, GPU acceleration).
- Cross-platform window management (remove Win32 dependency).
- Continue hardening all text/number fields against display overflow for multi-decade runs.

### Simulation Fidelity
- Finances simulation (operating costs, revenue per export, penalty for late deliveries).
- Data-driven and configurable scheduling behaviours.
- Self-healing systems: automatic fault detection, workaround routing, and recovery.
- Robot and charger random fault simulation.

