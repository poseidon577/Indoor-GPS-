import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# IEEE standard plotting parameters
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05
})

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(BASE_DIR, "figures")
DATA_DIR = os.path.join(BASE_DIR, "datasets")
TABLE_DIR = os.path.join(BASE_DIR, "tables")
DIAG_DIR = os.path.join(BASE_DIR, "diagrams")

for d in [FIG_DIR, DATA_DIR, TABLE_DIR, DIAG_DIR]:
    os.makedirs(d, exist_ok=True)

def save_fig(fig, filename_no_ext):
    """Saves figure in required IEEE formats."""
    formats = ["pdf", "svg", "png"]
    for fmt in formats:
        path = os.path.join(FIG_DIR, f"{filename_no_ext}.{fmt}")
        fig.savefig(path, format=fmt)
    print(f"Saved figure: {filename_no_ext}")

def save_data(df, filename):
    """Saves dataset as CSV."""
    if not filename.endswith('.csv'):
        filename += '.csv'
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved dataset: {filename}")

def get_standard_env():
    import sys
    backend_path = os.path.abspath(os.path.join(BASE_DIR, "..", "backend"))
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    
    from VLCL_AI.environment.state import EnvironmentState
    
    return EnvironmentState(
        current_time=0.0, frame_index=0, fps=60.0,
        receiver_velocity=[0,0,0], receiver_acceleration=[0,0,0], receiver_angles={'roll':0,'pitch':0,'yaw':0}, room_dims=[5.0, 5.0, 3.0],
        receiver_position=[2.5, 2.5, 0.85],
        receiver_orientation=[0, 0, 1],
        led_positions={1: [1.25, 1.25, 3.0], 2: [3.75, 1.25, 3.0], 3: [1.25, 3.75, 3.0], 4: [3.75, 3.75, 3.0]},
        led_powers={1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0},
        led_active={1: True, 2: True, 3: True, 4: True},
        led_orientations={1: [0,0,-1], 2: [0,0,-1], 3: [0,0,-1], 4: [0,0,-1]},
        led_beam_angles={1: 60.0, 2: 60.0, 3: 60.0, 4: 60.0},
        distances={1: 2.5, 2: 2.5, 3: 2.5, 4: 2.5},
        incident_angles_rad={1: 0, 2: 0, 3: 0, 4: 0},
        irradiance_angles_rad={1: 0, 2: 0, 3: 0, 4: 0},
        visibility_matrix={1: True, 2: True, 3: True, 4: True},
        los_matrix={1: True, 2: True, 3: True, 4: True},
        blocking_obstacles={},
        obstacles=[]
    )
