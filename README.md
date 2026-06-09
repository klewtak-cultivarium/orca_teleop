# orca_teleop

Repo for teleoperating the ORCA Hand consisting of an Ingress Source (for example Mediapipe, Apple Vision Pro, Rokoko Gloves, etc.) and a URDF-based Retargeter.

The repository follows a standard `src/` layout:

```text
src/orca_teleop/
tests/
```

## Development setup

Create a local virtual environment with `uv` and install the project in editable mode with the development extras used in this repository:

```bash
uv venv
source .venv/bin/activate
uv sync --extra test --extra mediapipe --extra adaptive
```

This installs the package itself plus the testing tools, MediaPipe dependencies, and the
optional solver stack used by the default adaptive analytical retargeter.

## Retargeting

The default teleop backend is `adaptive_analytical`, an Orca-native port of the Wuji-style
analytical retargeting strategy. It uses the full MediaPipe hand pose, explicit Orca frame
mappings from YAML, Pinocchio forward kinematics/Jacobians, and `nlopt` for bounded
per-frame optimization.

The legacy fingertip key-vector retargeter is still available for comparison:

```bash
python scripts/teleop_sim.py --env right --local --show-video --retargeter rmsprop
```

The adaptive backend loads `src/orca_teleop/retargeting/configs/adaptive_analytical_orca.yaml`
by default. Pass `--retarget-config path/to/config.yaml` to experiment with frame maps or
weights.

Steer your own ORCA hand using just your webcam:

```
python scripts/mediapipe_teleop_demo.py     path/to/your_orcahand_model     path/to/corresponding_urdf_file
```

Tests always run on CI. Run the regression suite from the repository root with:

```bash
pytest tests/
```
