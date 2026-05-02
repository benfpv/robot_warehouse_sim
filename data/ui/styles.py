"""Button and policy style dictionaries for the MainGame dashboard.

Each dict maps a mode-name string to a style record with these standard keys:
    label  — short button label text
    bg     — BGR background colour tuple
    edge   — BGR border/edge colour tuple
    text   — BGR label text colour tuple
    hint   — (PKG_TARGET only) short descriptor shown in overlay text

REASON and HINT dicts provide human-readable descriptions used in the UI tooltips
and info-panel text.
"""

POWER_POLICY_STYLE = {
    'eco':         {'label': 'ECO',  'bg': (20, 30, 24), 'edge': (80, 180, 120),  'text': (140, 220, 170)},
    'balanced':    {'label': 'BAL',  'bg': (22, 24, 30), 'edge': (120, 140, 220), 'text': (180, 190, 250)},
    'performance': {'label': 'PERF', 'bg': (30, 24, 22), 'edge': (220, 140, 100), 'text': (255, 210, 170)},
}

POWER_POLICY_REASON = {
    'eco':         'higher buffers, safer charging behavior',
    'balanced':    'balanced safety and throughput',
    'performance': 'lower buffers, higher utilization',
}

POWER_POLICY_HINT = {
    'eco':         'safe charge',
    'balanced':    'balanced',
    'performance': 'high util',
}

FLOW_POLICY_STYLE = {
    'steady':     {'label': 'STDY', 'bg': (20, 30, 24), 'edge': (80, 180, 120),  'text': (140, 220, 170)},
    'balanced':   {'label': 'BAL',  'bg': (22, 24, 30), 'edge': (120, 140, 220), 'text': (180, 190, 250)},
    'throughput': {'label': 'THRU', 'bg': (30, 24, 22), 'edge': (220, 140, 100), 'text': (255, 210, 170)},
}

FLOW_POLICY_REASON = {
    'steady':     'lower occupancy target for stability',
    'balanced':   'balanced occupancy target',
    'throughput': 'higher occupancy target for throughput',
}

FLOW_POLICY_HINT = {
    'steady':     'stable queue',
    'balanced':   'balanced',
    'throughput': 'max throughput',
}

PKG_TARGET_STYLE = {
    'random':    {'label': 'RND',  'hint': 'spread',    'bg': (18, 22, 24), 'edge': (48, 58, 66),    'text': (110, 120, 128)},
    'nearest':   {'label': 'NEAR', 'hint': 'min dist',  'bg': (20, 30, 40), 'edge': (40, 160, 220),  'text': (100, 210, 255)},
    'zone_edge': {'label': 'EDGE', 'hint': 'pipeline',  'bg': (22, 30, 25), 'edge': (80, 180, 180),  'text': (170, 230, 230)},
}

PKG_TARGET_REASON = {
    'random':    'spread load and avoid clustering',
    'nearest':   'minimize immediate travel distance',
    'zone_edge': 'stage packages for the next pipeline handoff',
}
