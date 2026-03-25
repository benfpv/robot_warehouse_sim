# Robot Warehouse Sim

A real-time autonomous warehouse simulation. Robots dynamically pick up, transport, and deliver packages across import, storage, and export zones — managing battery, charging, and deadline priorities — all rendered in a live composite dashboard window.

![robot_warehouse_sim](https://github.com/benfpv/robot_warehouse_sim/assets/55154673/d94b5d34-b65e-41d7-97d8-b29c454b6042)

---

## Features

- **Warehouse zones** — three horizontal bands (import / storage / export); packages enter via import and exit when exported from the export zone before their deadline.
- **Packages** — each has a unique ID, real-time position & status, a deadline-to-export, colour-coded urgency, age tracking (`createdAt`), and a full import-to-export history log.
- **Robots** — up to 60 robots, each with a priority-based action queue, dynamic battery depletion, automatic charging dispatch, age tracking (`createdAt`), and single-package carrying capacity.
- **Chargers** — up to 10 charging stations spawned along the warehouse perimeter; robots are dispatched when battery ≤ 20 %. Charger exclusivity is enforced via action-queue ground-truth checks with per-tick stale-reservation reconciliation and an arrival guard.
- **Task scheduler** — the warehouse assigns packages to robots deadline-first, respects zone capacity, and avoids duplicate movement effort via `packagesMoveList` / `packagesMovingList`.
- **Sim / draw decoupling** — simulation ticks at 40 Hz; display repaints at 20 fps independently to save CPU.
- **Rolling chart histories** — 1 800 samples at 1 sample/s (~30 min), with a dynamic time-window that grows from 30 s to the full history length.
- **Overflow-safe display** — all number fields use `_n()` (K/M abbreviation), `_fmt_elapsed()` / `_fmt_age()` (decade-aware), `_fmt_rate()` (K/M/s), and a 3-tier cascade (full → drop rate suffix → char-trim + …). The sim is designed to run for days, months, or years without text bleeding out of any panel.

## Dashboard Layout (800 × 575 px)

```
┌──────────────────────┬─────────┬─────────┬─────────┐
│                      │Chargers │Packages │Zone Map │  ← sub-views (160×140 each)
│   Main view          ├─────────┼─────────┼─────────┤
│   (320×280, 4× zoom) │ Robots  │Pkg Tgts │Sim Stats│
├──────────────────────┴─────────┴─────────┴─────────┤
│  Info panel — ROBOTS | CHARGERS | PACKAGES  (165px) │
├─────────────────────────────────────────────────────┤
│  Charts — Exports/Overdue | Zone Bars | Fleet (130px)│
└─────────────────────────────────────────────────────┘
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

**Info panel** shows top-7 active robots (sorted by task urgency) + top-3 lowest-battery robots, top-10 chargers, and top-10 packages nearest deadline — all colour-coded by state. All text is clipped to its column boundary with overflow protection.

**Chart strip** shows a dynamically growing time window (starts at 30 s, expands to 30 min):
- *Exports & Overdue* — cumulative exports (+ rate), total late (+ rate), and current instantaneous overdue count (+ rate)
- *Zone Capacity* — live fill bars for import / storage / export with count, rate, and occupancy percentage
- *Fleet Health* — mean battery % (+ rate), robot count, robots needing charge (+ rate), robots actively charging (+ rate)

**Sim Stats panel** shows: elapsed time (decade-aware), loop count, exports, package count, robot count, charger count, zone fill levels with rates, and newest/oldest package (identified by P#).

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
| `draw_frametime` | `1/20` | Display repaint rate (s) |
| `panel_h` | `165` | Info panel height (px) |
| `chart_h` | `130` | Chart strip height (px) |

**`data/warehouse/warehouse.py` — `Warehouse.__init__`**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `robotsMaxQuantity` | `60` | Max robots in simulation |
| `chargersMaxQuantity` | `10` | Max charging stations |
| `packagesMaxMoveQuantity` | `120` | Max packages in the delivery pipeline at once |
| `robotsTaskAssignmentMaxQuantity` | `3` | Max queued tasks per robot |

Zone geometry (pad, thirds) is set in `data/warehouse/warehouse_init.py` → `init_zoneMap`.

---

## Project Structure

```
robot_warehouse_sim/
├── main.py                          # Entry point, MainGame, composite display, charts
├── requirements.txt
├── resources/
│   ├── list_items.csv               # Item catalogue loaded at startup
│   └── list_addresses.csv           # Address list for package origin/destination
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
    └── warehouse/
        ├── warehouse.py             # Core simulation loop and all update methods
        ├── warehouse_init.py        # Grid / zone map initialisation
        ├── warehouse_functions.py   # (Reserved for future warehouse-level helpers)
        ├── warehouse_data.py        # Rolling timeseries data container
        ├── warehouse_log.py         # Log record classes
        ├── package.py               # Package dataclass
        ├── package_functions.py     # Package spawn, target selection, generation
        ├── charger.py               # Charger dataclass
        ├── charger_functions.py     # Charger spawn and generation
        ├── robot.py                 # Robot dataclass
        └── robot_functions.py       # Robot spawn and generation
```

---

## Known Issues

- When the export zone fills completely, already-planned package movements are not cancelled or re-prioritised in real time; robots may continue heading toward a full export zone until the next scheduler pass. (See *Dynamic priority switching* in Future Directions.)

## Limitations

- Import / storage / export zones must be rectangular and stacked vertically (horizontal bands, top: import → middle: storage → bottom: export). More complex area geometries are not currently supported.
- Windows-only: borderless window setup uses Win32 via `ctypes`.

## Future Directions

### Scheduling & Capacity Optimisation
- Optimisation calculations for managing zone capacities, fleet sizes, and time-based throughput targets.
- Dynamic priority switching when export is full: cancel or re-route in-flight package movements based on real-time zone capacities rather than waiting for the next scheduler tick.
- Dynamically adaptive zone shapes and sizes — e.g., complex non-rectangular footprints (spirals, L-shapes, multi-level). In real life, a 3D helical layout could exploit gravity to move packages passively from import → storage → export.
- Dispatch robots to the nearest available charger rather than any idle charger.
- Allow swapping lower-priority packages out of export to unblock higher-priority ones.

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

