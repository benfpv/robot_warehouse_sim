"""Shared constants used across the warehouse simulation."""

# ── Zone type identifiers ──────────────────────────────────────────────
ZONE_NONE = 0
ZONE_IMPORT = 1
ZONE_STORAGE = 2
ZONE_EXPORT = 3
ZONE_NAMES = {
    ZONE_NONE: 'neutral',
    ZONE_IMPORT: 'import',
    ZONE_STORAGE: 'storage',
    ZONE_EXPORT: 'export',
}

# ── Simulation timing ──────────────────────────────────────────────────
SIM_TICK_RATE = 40  # simulation ticks per second
STALL_TICKS = SIM_TICK_RATE  # 1-second action pause (pickup, dropoff, charger dock/undock)

# ── Robot tuning ───────────────────────────────────────────────────────
ROBOT_CARRY_SPEED_PENALTY = 0.92  # max-velocity multiplier when carrying a package
BATTERY_FULL_PCT = 98  # battery % at which charging is considered complete
