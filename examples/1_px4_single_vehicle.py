#!/usr/bin/env python
"""
| File: 1_px4_single_vehicle.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: This files serves as an example on how to build an app that makes use of the Pegasus API to run a simulation with a single vehicle, controlled using the MAVLink control backend.
"""

# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp

# Start Isaac Sim's simulation environment
# Note: this simulation app must be instantiated right after the SimulationApp import, otherwise the simulator will crash
# as this is the object that will load all the extensions and load the actual simulator.
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
# Auxiliary scipy and numpy modules
import os.path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import UsdGeom, Gf, UsdPhysics

# --- Rusko Summer (3dgrut Gaussian-splat export) spawn geometry --------------
# The world is a 3D Gaussian splat (rendered, no collision) bundled with a
# triangle mesh used purely for physics. The mesh is referenced under
# /World/layout/mesh and the splat under /World/layout/gaussians.
#
# The terrain is small (~9 x 15 x 11 units, Z-up, metersPerUnit=1) and bumpy
# everywhere, so resting the drone directly on it tilts it and trips PX4's
# "Preflight Fail: Attitude failure (roll)" check. We instead drop a small,
# invisible, level collision pad just above the surface near the scene centre
# and spawn the drone on that. Tune these if you want a different launch spot.
RUSKO_MESH_PATH = "/World/layout/mesh"      # collider geometry
RUSKO_SPLAT_PATH = "/World/layout/gaussians"  # rendered splat
RUSKO_SPAWN_XY = (0.18, 1.31)               # scene centre (mesh bbox centre)
RUSKO_PAD_TOP_Z = 1.65                      # just above local surface top (~1.55)
RUSKO_PAD_HALF = 1.0                        # pad half-extent in X/Y


class PegasusApp:
    """
    A Template class that serves as an example on how to build a simple Isaac Sim standalone App.
    """

    def __init__(self):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """

        # Acquire the timeline that will be used to start/stop the simulation
        self.timeline = omni.timeline.get_timeline_interface()

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Acquire the World, .i.e, the singleton that controls that is a one stop shop for setting up physics, 
        # spawning asset primitives, etc.
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Launch one of the worlds provided by NVIDIA
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Rusko Summer"])

        # load_environment() is ASYNCHRONOUS: the USD is referenced over the next few
        # frames. For a large terrain we must wait until it is actually present before
        # spawning the vehicle and resetting physics, otherwise the drone is created and
        # physics is initialized before the terrain collider exists and it free-falls
        # straight through the not-yet-loaded ground.
        # Wait until the collision mesh itself is composed in (not just /World/layout).
        for _ in range(4000):
            simulation_app.update()
            mesh = self.world.stage.GetPrimAtPath(RUSKO_MESH_PATH)
            if mesh.IsValid() and UsdGeom.Mesh(mesh).GetPointsAttr().HasValue():
                break

        # Turn the bundled mesh into a static collider for the splat scene and add
        # the level spawn pad the drone takes off from.
        self._setup_rusko_collision()

        # Create the vehicle
        # Try to spawn the selected robot in the world to the specified namespace
        config_multirotor = MultirotorConfig()
        # Create the multirotor configuration
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe # CHANGE this line to 'iris' if using PX4 version bellow v1.14
        })
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]

        spawn_z = RUSKO_PAD_TOP_Z + 0.08  # rest the drone just above the pad surface
        self.vehicle = Multirotor(
            "/World/quadrotor",
            ROBOTS['Pegasus'],
            0,
            [RUSKO_SPAWN_XY[0], RUSKO_SPAWN_XY[1], spawn_z],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        # --- Additional third-person (chase) camera that follows the drone ---
        # This only creates an extra camera PRIM and keeps it aimed at the drone every
        # frame. The default viewport is left alone; pick "chase_camera" from the
        # viewport's camera dropdown whenever you want the third-person view.
        self.chase_camera_path = "/World/chase_camera"
        UsdGeom.Camera.Define(self.world.stage, self.chase_camera_path)

        # Chase-camera offset from the drone, in the drone's BODY (FLU) frame (meters):
        # behind (-X = aft), and above (+Z). This offset is rotated by the drone's yaw
        # each frame so the camera always sits behind the drone as it turns. Tune to taste.
        self.camera_offset = np.array([-4.0, 0.0, 1.5])

        # Point the viewport at the (small) splat scene so it's framed on startup.
        self.pg.set_viewport_camera(
            [RUSKO_SPAWN_XY[0] + 6.0, RUSKO_SPAWN_XY[1] - 6.0, RUSKO_PAD_TOP_Z + 4.0],
            [RUSKO_SPAWN_XY[0], RUSKO_SPAWN_XY[1], RUSKO_PAD_TOP_Z],
        )

        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

    def _setup_rusko_collision(self):
        """Make the bundled terrain mesh a static collider (hidden, so only the
        Gaussian splat shows) and add an invisible level pad for the drone to
        take off from."""
        stage = self.world.stage

        mesh = stage.GetPrimAtPath(RUSKO_MESH_PATH)
        if mesh.IsValid():
            UsdPhysics.CollisionAPI.Apply(mesh)
            mca = UsdPhysics.MeshCollisionAPI.Apply(mesh)
            # "none" = exact triangle-mesh collider (valid for static geometry).
            mca.CreateApproximationAttr().Set("none")
            UsdGeom.Imageable(mesh).MakeInvisible()
        else:
            carb.log_warn(f"Rusko collider mesh not found at {RUSKO_MESH_PATH}")

        # Invisible, level static collision pad placed just above the local
        # surface at the spawn point (the splat terrain itself is too bumpy for a
        # clean level takeoff). Cube spans [-1,1]; scale to a thin slab.
        pad = UsdGeom.Cube.Define(stage, "/World/spawn_pad")
        pad.GetSizeAttr().Set(2.0)
        half_thickness = 0.1
        xf = UsdGeom.Xformable(pad.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(RUSKO_SPAWN_XY[0], RUSKO_SPAWN_XY[1],
                                         RUSKO_PAD_TOP_Z - half_thickness))
        xf.AddScaleOp().Set(Gf.Vec3f(RUSKO_PAD_HALF, RUSKO_PAD_HALF, half_thickness))
        UsdPhysics.CollisionAPI.Apply(pad.GetPrim())
        UsdGeom.Imageable(pad.GetPrim()).MakeInvisible()

    def _aim_chase_camera(self, eye, target):
        """Place the chase-camera prim at `eye` looking at `target` (world frame, +Z up)."""
        eye = Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2]))
        target = Gf.Vec3d(float(target[0]), float(target[1]), float(target[2]))
        f = (target - eye).GetNormalized()              # forward (camera looks down -Z)
        r = Gf.Cross(f, Gf.Vec3d(0, 0, 1)).GetNormalized()  # right
        u = Gf.Cross(r, f)                              # up
        # Row-vector transform: rows are the world directions of camera local X, Y, Z.
        m = Gf.Matrix4d(
            r[0],  r[1],  r[2],  0.0,
            u[0],  u[1],  u[2],  0.0,
            -f[0], -f[1], -f[2], 0.0,
            eye[0], eye[1], eye[2], 1.0,
        )
        cam = UsdGeom.Xformable(self.world.stage.GetPrimAtPath(self.chase_camera_path))
        cam.ClearXformOpOrder()
        cam.AddTransformOp().Set(m)

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """

        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        while simulation_app.is_running() and not self.stop_sim:

            # Keep the chase-camera prim behind the drone (third-person follow). Rotate
            # the body-frame offset by the drone's yaw so it stays behind as it turns;
            # yaw-only keeps the horizon level (camera doesn't flip when the drone rolls).
            # Does not change the active viewport; select "chase_camera" to use it.
            drone_pos = self.vehicle.state.position
            yaw = Rotation.from_quat(self.vehicle.state.attitude).as_euler("xyz")[2]
            world_offset = Rotation.from_euler("z", yaw).apply(self.camera_offset)
            self._aim_chase_camera(drone_pos + world_offset, drone_pos)

            # Update the UI of the app and perform the physics step
            self.world.step(render=True)
        
        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()

def main():

    # Instantiate the template app
    pg_app = PegasusApp()

    # Run the application loop
    pg_app.run()

if __name__ == "__main__":
    main()
