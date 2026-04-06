from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rerun as rr
import yourdfpy

from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config import RerunRobotConfig

logger = logging.getLogger(__name__)

# RerunRobot accepts normalized [-100, +100] (same as physical robots like SO101)
# and maps to radians internally using URDF joint limits.
try:
    from lerobot_action_space.modes import ActionMode
    RERUN_ROBOT_MODES: list[ActionMode] = [
        ActionMode(
            name="joint_absolute_norm",
            space_type="joint",
            unit="normalized",
            command_mode="absolute",
            description="Joint positions in normalized [-100, +100] range (mapped to rad via URDF limits)",
            is_default=True,
        )
    ]
except ImportError:
    RERUN_ROBOT_MODES = []

# Link colours by common name fragments (RGB 0-255)
_LINK_COLORS: list[tuple[str, list[int]]] = [
    ("base",    [50,  50,  65 ]),
    ("shoulder",[220, 85,  25 ]),
    ("upper",   [220, 85,  25 ]),
    ("lower",   [230, 155, 25 ]),
    ("wrist",   [230, 155, 25 ]),
    ("gripper", [50,  155, 210]),
    ("jaw",     [50,  155, 210]),
    ("finger",  [50,  155, 210]),
    ("elbow",   [180, 100, 20 ]),
]
_DEFAULT_COLOR = [140, 140, 155]


def _link_color(link_name: str) -> list[int]:
    low = link_name.lower()
    for fragment, color in _LINK_COLORS:
        if fragment in low:
            return color
    return _DEFAULT_COLOR


def _build_node_to_link(robot: yourdfpy.URDF) -> dict[str, str]:
    """Map yourdfpy scene graph node names → URDF link names."""
    mesh_counter: dict[str, int] = {}
    node_to_link: dict[str, str] = {}
    for link in robot.robot.links:
        for visual in link.visuals:
            if visual.geometry.mesh is None:
                continue
            base = Path(visual.geometry.mesh.filename).name
            count = mesh_counter.get(base, 0)
            node_name = base if count == 0 else f"{base}_{count}"
            mesh_counter[base] = count + 1
            node_to_link[node_name] = link.name
    return node_to_link


def _stl_fname(geom_node_name: str) -> str:
    """Strip yourdfpy's _N dedup suffix to get the real STL filename."""
    if ".stl_" in geom_node_name:
        return geom_node_name[:geom_node_name.rindex(".stl_") + 4]
    return geom_node_name


class RerunRobot(Robot):
    """
    Virtual LeRobot robot that visualizes any URDF model in rerun.

    Accepts joint position actions via send_action(), updates the URDF
    forward kinematics, and streams Transform3D updates to rerun.
    Returns the current joint positions as observations.
    """

    config_class = RerunRobotConfig
    name = "rerun_robot"

    def __init__(self, config: RerunRobotConfig):
        super().__init__(config)
        self.config = config
        self._robot: yourdfpy.URDF | None = None
        self._node_to_link: dict[str, str] = {}
        self._joint_names: list[str] = []
        self._joint_state: dict[str, float] = {}  # in radians
        self._joint_limits: dict[str, tuple[float, float]] = {}  # (lo_rad, hi_rad)
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def robot_modes(self):
        """Action modes this robot accepts (joint_absolute_rad)."""
        return RERUN_ROBOT_MODES

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def observation_features(self) -> dict:
        return {f"{j}.pos": float for j in self._joint_names}

    @property
    def action_features(self) -> dict:
        return {f"{j}.pos": float for j in self._joint_names}

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        urdf_path = Path(self.config.urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        mesh_dir = Path(self.config.mesh_dir) if self.config.mesh_dir else urdf_path.parent

        logger.info(f"Loading URDF: {urdf_path}")
        self._robot = yourdfpy.URDF.load(str(urdf_path), mesh_dir=str(mesh_dir))
        self._node_to_link = _build_node_to_link(self._robot)

        # Determine joint names and limits
        all_joints = [j for j in self._robot.robot.joints if j.type != "fixed"]
        names = self.config.joint_names if self.config.joint_names else [j.name for j in all_joints]
        self._joint_names = names
        self._joint_state = {j: 0.0 for j in self._joint_names}
        self._joint_limits = {}
        for j in all_joints:
            if j.name in self._joint_names and j.limit:
                self._joint_limits[j.name] = (j.limit.lower, j.limit.upper)

        # Init rerun
        rr.init(self.config.rerun_app_id, spawn=self.config.spawn_viewer)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        # Log static meshes
        self._log_meshes()

        # Log initial pose
        self._log_transforms()

        self._is_connected = True
        logger.info(f"RerunRobot connected — joints: {self._joint_names}")

    def calibrate(self) -> None:
        pass  # No calibration needed

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} not connected")
        return {f"{j}.pos": self._norm(j) for j in self._joint_names}

    def send_action(self, action: dict) -> dict:
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} not connected")

        for j in self._joint_names:
            key = f"{j}.pos"
            if key in action:
                norm = float(action[key])
                # Convert normalized [-100, +100] → radians via joint limits
                if j in self._joint_limits:
                    lo, hi = self._joint_limits[j]
                    mid = (hi + lo) / 2
                    amp = (hi - lo) / 2
                    self._joint_state[j] = mid + (norm / 100.0) * amp
                else:
                    self._joint_state[j] = norm  # fallback: no limits known

        self._log_transforms()
        # Return normalized (matches input convention)
        return {f"{j}.pos": self._norm(j) for j in self._joint_names}

    def _norm(self, joint: str) -> float:
        """Convert internal rad state back to normalized [-100, +100]."""
        rad = self._joint_state[joint]
        if joint in self._joint_limits:
            lo, hi = self._joint_limits[joint]
            mid = (hi + lo) / 2
            amp = (hi - lo) / 2
            return (rad - mid) / amp * 100.0 if amp else 0.0
        return rad

    def disconnect(self) -> None:
        if not self._is_connected:
            return
        self._is_connected = False
        logger.info("RerunRobot disconnected")

    # ------------------------------------------------------------------
    # Rerun helpers
    # ------------------------------------------------------------------

    def _log_meshes(self) -> None:
        urdf_path = Path(self.config.urdf_path)
        mesh_dir = Path(self.config.mesh_dir) if self.config.mesh_dir else urdf_path.parent

        for geom_node_name in self._robot.scene.geometry:
            link_name = self._node_to_link.get(geom_node_name, "")
            color = _link_color(link_name) + [255]
            stl_fname = _stl_fname(geom_node_name)

            # Try assets/ subdirectory first, then mesh_dir root
            stl_path = mesh_dir / "assets" / stl_fname
            if not stl_path.exists():
                stl_path = mesh_dir / stl_fname
            if not stl_path.exists():
                logger.warning(f"Mesh not found: {stl_fname}")
                continue

            entity = f"world/robot/{geom_node_name}"
            rr.log(entity, rr.Asset3D(path=stl_path, albedo_factor=color), static=True)

    def _log_transforms(self) -> None:
        self._robot.update_cfg(self._joint_state)
        for geom_node_name in self._robot.scene.geometry:
            T, _ = self._robot.scene.graph.get(geom_node_name)
            entity = f"world/robot/{geom_node_name}"
            rr.log(entity, rr.Transform3D(mat3x3=T[:3, :3], translation=T[:3, 3]))
