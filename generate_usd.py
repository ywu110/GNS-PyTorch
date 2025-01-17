import warp
import warp.render as render
from warp import vec3
import pickle
import numpy as np
from typing import Dict, List, Tuple


def render_usd_per_frame(
    renderer,
    xpos_gt: np.ndarray,
    xpos_pred: np.ndarray,
    nonk_mask: np.ndarray,
    time: float) -> None:
    r"""
    Render a sim step with a USD renderer.

    Parameters:
        renderer: The USD renderer to render the step
        xpos_gt: Positions of source system particles in GT motion
            (num_particles_gt, 3)
        time: The simulation time in seconds
        render_src_sys: Whether to render the source system or not
        springs_gt: Ground truth system's spring topology
            (num_springs_gt, 2)

    """
    renderer.begin_frame(time)
        
    # render each GT particle mapped to surrogate space
    num_particles = xpos_gt.shape[0]
    for particle_idx in range(num_particles):
        if nonk_mask[particle_idx]:
            renderer.render_sphere(
                f"particle_{particle_idx:03d}_gt",
                xpos_gt[particle_idx],
                (0.0, 0.0, 0.0, 1.0),
                0.01, # self.ball_radius
                color=(0.2, 0.4, 0.6))
            renderer.render_sphere(
                f"particle_{particle_idx:03d}_pred",
                xpos_pred[particle_idx],
                (0.0, 0.0, 0.0, 1.0),
                0.01, # self.ball_radius
                color=(0.6, 0.2, 0.2))
        else:
            renderer.render_sphere(
                f"particle_{particle_idx:03d}_fixed",
                xpos_gt[particle_idx],
                (0.0, 0.0, 0.0, 1.0),
                0.01, # self.ball_radius
                color=(0.0, 0.0, 0.0))
    
    renderer.end_frame()
    return

# read into particle_data
pickle_file_path = "eval_vis/fem/iter_300000-FEM-test/0.pkl"
with open(pickle_file_path, 'rb') as f:
    pickle_data = pickle.load(f)

predicted_positions = pickle_data['pred_poss']
target_positions = pickle_data['tgt_poss']
nonk_mask = pickle_data['nonk_mask']

# Set up the WARP renderer
stage = "animation.usd"
renderer = render.UsdRenderer(stage=stage)

# render
num_frames = predicted_positions.shape[0]

sim_dt = 0.001

for frame_idx in range(num_frames):
    target_positions_frame = target_positions[frame_idx]
    predicted_positions_frame = predicted_positions[frame_idx]
    sim_time = frame_idx * sim_dt
    render_usd_per_frame(
        renderer,
        target_positions_frame,
        predicted_positions_frame,
        nonk_mask,
        sim_time)

# save
renderer.save()
print(f"USD file saved to {stage}")