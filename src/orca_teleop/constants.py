JOIN_TIMEOUT = 3.0
HEARTBEAT_INTERVAL = 0.1
QUEUES_MAXSIZE = 8
INGRESS_FPS = 30
MOTION_NUM_STEPS = 1

WRIST_MOTOR_IDX = 16
DEFAULT_PORT = 50051
DEFAULT_HAND = "right"

# MediaPipe teleop
DEFAULT_CONFIDENCE = 0.7
_NUM_KEYPOINTS = 21
_COORDS_PER_POINT = 3
_EXPECTED_LEN = _NUM_KEYPOINTS * _COORDS_PER_POINT

# Meta Quest teleop
# Per-side, per-axis asymmetric reach envelope of the carpals frame, expressed
# as (lo, hi) deltas in meters relative to the side's teleop home pose, in URDF
# world frame (FLU: x_fwd, y_left, z_up). Source: scripts/fk_workspace_sweep.py
# (1500-sample joint-space sweep, seed=0). The numbers are bilaterally
# consistent — left/right mirror on y, agree on x and z — which is the
# kinematic sanity check that the FK chain is correct.
# Consumed in scripts/teleop_arm_quest.py:
#   - clip site: ``dp = np.clip(dp, lo, hi)`` element-wise (asymmetric).
#   - span re-fit: ``(hi - lo) / 2`` recovers a per-axis half-width that plays
#     the same role the old symmetric constant did in the scale ratio.
WORKSPACE_DELTA_LIMITS_M: dict[
    str, tuple[tuple[float, float, float], tuple[float, float, float]]
] = {
    "left": ((-0.542, -0.272, -0.143), (+0.380, +0.465, +0.783)),
    "right": ((-0.541, -0.465, -0.142), (+0.383, +0.264, +0.783)),
}
AUTO_FIT_MARGIN = 0.7
CUTOFF_MIN = 0.05

MIN_SPAN_SAMPLES = 150
SPAN_BUFFER_SECONDS = 60.0
SPAN_REFIT_PERIOD_S = 60.0
SPAN_CHANGE_THRESHOLD = 0.10
BOOTSTRAP_SCALE = 0.15

STILL_THRESHOLD_M = 0.01
STILL_WINDOW_SAMPLES = 30
CLUTCH_GRACE_S = 2.0

# Per-tick joint-step clamp applied AFTER the IK solve in the Quest teleop
# loop. Caps |Δq| = |q_new - q_prev| at this many radians per joint per IK
# tick, so a large target jump (clutch re-anchor, scale change, lost-tracking
# blip) integrates over multiple ticks instead of yanking the arm in one.
# At IK_RATE_HZ = 60, 0.05 rad/tick = 3.0 rad/s ≈ 172 deg/s — well above
# anything a Quest operator can move, but well below the URDF velocity limits.
MAX_JOINT_STEP_RAD = 0.05
