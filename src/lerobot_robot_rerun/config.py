from dataclasses import dataclass, field
from pathlib import Path

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("rerun_robot")
@dataclass
class RerunRobotConfig(RobotConfig):
    """
    Configuration for the Rerun URDF visualizer robot.

    Acts as a virtual LeRobot robot that renders any URDF model in a
    rerun viewer window. Accepts joint actions and returns them as
    observations, making it a transparent pass-through for testing
    teleop pipelines without real hardware.
    """

    # Path to the URDF file
    urdf_path: str = ""

    # Directory containing mesh assets referenced by the URDF
    # If empty, defaults to the directory containing the URDF file
    mesh_dir: str = ""

    # Joint names to expose as action/observation features.
    # If empty, all non-fixed joints from the URDF are used.
    joint_names: list[str] = field(default_factory=list)

    # Rerun application ID shown in the viewer title
    rerun_app_id: str = "lerobot_robot_rerun"

    # Whether to spawn a new rerun viewer process on connect
    spawn_viewer: bool = True
