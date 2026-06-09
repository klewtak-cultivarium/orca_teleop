"""Run Manus glove teleop in sim. Start SharpaManusClient.out first."""

from orca_teleop.pipeline import run_manus_local
from orca_teleop.sim import OrcaHandSimSink

run_manus_local(sink=OrcaHandSimSink(env_name="right", render_mode="human"))
