"""Single scientific contract shared by all FlowContour entry points."""

PROTOCOL_VERSION = "flowcontour.full2.v2"
STAGE_VALUES = (0.0, 1.0)  # update / refinement, NOT a GT completion fraction
ODE_STEPS = 4
CONTOUR_POINTS = 128
NSD_SCALES_PX = (1.0, 2.0, 4.0, 8.0)


def protocol_metadata():
    return {
        "version": PROTOCOL_VERSION,
        "contour_points": CONTOUR_POINTS,
        "stage_values": list(STAGE_VALUES),
        "stage_semantics": "update_then_refinement_not_gt_progress",
        "full_displacement": True,
        "outer_steps": 2,
        "solver": "ab2",
        "nfe_per_stage": ODE_STEPS,
        "total_nfe": 2 * ODE_STEPS,
        "latent_scale": 1.0,
        "model_compute_dtype": "float32",
        "reward_scales_input_px": list(NSD_SCALES_PX),
        "reward_coordinate_space": "network_input_pixels",
        "credit": "terminal_advantage_times_mean_one_stage_quality",
    }
