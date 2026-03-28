# Robot Warehouse Sim

A real-time autonomous warehouse simulation. Robots dynamically pick up, transport, and deliver packages across import, storage, and export zones — managing battery, charging, and deadline priorities — all rendered in a live composite dashboard window.

![robot_warehouse_sim](https://github.com/benfpv/robot_warehouse_sim/assets/55154673/d94b5d34-b65e-41d7-97d8-b29c454b6042)

---

## Features

- **Warehouse zones** — three zone types (import / storage / export) painted on an 80×70 grid; packages enter via import and exit when exported from the export zone before their deadline.  Zones can be painted interactively or loaded from a PNG map file.
- **Adaptive zone optimizer** — monitors per-zone utilization via EMA and automatically swaps boundary cells between zone types to rebalance capacity.  Zone-swap only: no zone is ever eliminated and no new zones are created from neutral space.
- **Packages** — each has a unique ID, real-time position & status, a deadline-to-export (10–90 s), colour-coded urgency, age tracking (`createdAt`), and a full import-to-export history log.  Urgency-based routing sends near-deadline packages to export and others to storage.
- **Robots** — up to 60 robots, each with a priority-based action queue, dynamic battery depletion with an adaptive Power Policy, automatic charging dispatch, age tracking (`createdAt`), and single-package carrying capacity.
- **Chargers** — up to 10 charging stations; robots are dispatched when battery falls below a Power Policy threshold (profile-dependent range, roughly 27–50 %, adapts to charger pressure and uses extra damping/smoothing to avoid abrupt flips). Charger exclusivity is enforced via action-queue ground-truth checks with per-tick stale-reservation reconciliation and an arrival guard.
- **Flow control** — adaptive import cap targeting ~50 % warehouse occupancy with proportional correction, overdue/stress penalties, and idle-robot bonus.
- **Dynamic planning depth** — the package move pipeline is capped at `available_robots × 2` to avoid over-planning when many robots are charging or busy.
- **Task scheduler** — the warehouse assigns packages to robots deadline-first, respects zone capacity, and avoids duplicate movement effort via `packagesMoveList` / `packagesMovingList`.
- **Sim / draw decoupling** — simulation ticks at 40 Hz; display repaints at 15 fps independently to save CPU.
- **Rolling chart histories** — 1 800 samples at 1 sample/s (~30 min), with a dynamic time-window that grows from 30 s to the full history length.
- **Overflow-safe display** — all number fields use `_n()` (K/M abbreviation), `_fmt_elapsed()` / `_fmt_age()` (decade-aware), `_fmt_rate()` (K/M/s), and a 3-tier cascade (full → drop rate suffix → char-trim + …). The sim is designed to run for days, months, or years without text bleeding out of any panel.

## Dashboard Layout (960 × 540 px)

```
┌──────────────────────┬─────────┬─────────┬─────────┬──────────┐
│                      │Chargers │Packages │Zone Map │Deadlines │  ← sub-views (160×140)
│   Main view          ├─────────┼─────────┼─────────┤Fleet Batt│
│   (320×280, 4× zoom) │ Robots  │Pkg Tgts │Heatmap  │ (empty)  │
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
| Teal dots (perimeter) | Charging stations |
| White dots | Robots |
| Grey dots | Idle packages |
| Bright green dots | Packages with an assigned robot & target |
| Dark blue dots | Export zone, idle, deadline 10–60 s |
| Light blue dots | Export zone, idle, deadline < 10 s |
| Orange dots | Overdue packages planned for movement |
| Red dots | Overdue packages with no plan |

**Info panel** (4 columns × 240 px) shows:
- **ROBOTS** — top-7 active robots (sorted by task urgency) + top-7 lowest-battery robots, colour-coded by state.
- **CHARGERS** — charger status + fleet Power Policy summary (avg battery, drain, smoothed/raw pressure, smoothed/raw threshold, working/charging/idle counts), plus mode buttons (`ECO`, `BAL`, `PERF`) for live power-policy override.
- **SIM STATS** — elapsed time, loop count, exports, package count, robot count, zone fill levels with rates, zone utilization, optimizer status, and flow-policy buttons (`STDY`, `BAL`, `THRU`) for live flow-control override.
- **PACKAGES** — all packages sorted by deadline urgency with zone, target, carrier, status, weight, item name and destination.

All text is clipped to its column boundary with overflow protection.

**Chart strip** (4 charts × 240 px) shows a dynamically growing time window (starts at 30 s, expands to 30 min):
- *Exports & Overdue* — cumulative exports (+ rate), total late (+ rate), and current instantaneous overdue count (+ rate)
- *Flow Control* — import cap, throughput/min, avg delivery time, overdue % (fixed 0–100 scale)
- *Zone Capacity* — live fill bars for import / storage / export with count, rate, and occupancy percentage
- *Fleet Health* — mean battery % (+ rate), robot count, robots needing charge (+ rate), robots actively charging (+ rate), idle count

**Top-right panels:**
- *Deadlines* — 5-band urgency histogram (OVR/CRIT/URG/NRML/CMFT) with per-band count and percentage
- *Fleet Batt* — horizontal bar chart showing every robot's battery level sorted ascending, with status indicators (charging/en-route/saving/carrying), a dynamic threshold line, and a compact Power Policy readout

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
- `STDY`: lower target occupancy for stability
- `BAL`: default 50% occupancy target
- `THRU`: higher target occupancy for throughput
- **Package Target Mode** (`PACKAGES` header): `RND`, `NEAR`, `EDGE`
- Controls placement strategy inside the destination zone

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
    │   └── display_functions.py     # Borderless Win32 window setup
    ├── draw/
    │   └── draw_warehouse.py        # All OpenCV draw helpers (zones, robots, packages, arrows)
    ├── paint/
    │   ├── paint_handler.py         # Interactive zone painting UI (mouse + keyboard)
    │   ├── map_importer.py          # PNG zone/spawn map loader and example generators
    │   ├── zone_strategy.py         # Adaptive zone rebalancing (zone-swap only)
    │   └── warehouse_optimizer.py   # Strategy orchestrator (tick budget, strategy dispatch)
    └── warehouse/
        ├── warehouse.py             # Core simulation loop, flow control, and all update methods
        ├── warehouse_init.py        # Grid / zone map initialisation
        ├── warehouse_data.py        # Rolling timeseries data container
        ├── warehouse_log.py         # Log record classes (Packages_Log, Robots_Log)
        ├── package.py               # Package dataclass
        ├── package_functions.py     # Package spawn, target selection, deadline generation
        ├── charger.py               # Charger dataclass
        ├── charger_functions.py     # Charger spawn and generation
        ├── robot.py                 # Robot dataclass
        └── robot_functions.py       # Robot spawn and generation
```

---

## Known Issues

- When the export zone fills completely, already-planned package movements are not cancelled or re-prioritised in real time; robots may continue heading toward a full export zone until the next scheduler pass.
- Robot movement is straight-line (beeline via atan2 → cardinal direction); there is no pathfinding or collision avoidance.

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
- **Road/aisle layer** — user-painted or PNG-loaded travel-lane map; robots prefer (soft) or are restricted to (hard) designated road cells, enabling realistic aisle layouts and directional one-way lanes.
- Collision avoidance between robots.
- Optimised robot trajectories (shortest path, A* or similar, acceleration modelling).
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

