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

# The Rusko Summer world is an Omniverse NuRec (neural-reconstruction) volume. Its
# prims use the OmniNuRecFieldAsset schema, which is NOT in the default experience;
# without it the RTX renderer can't recognise the volume and the photoreal terrain
# never draws. Enable the schema BEFORE the world USD is composed (pulled from the
# NVIDIA registry on first run, then cached locally).
from isaacsim.core.utils.extensions import enable_extension
if not enable_extension("omni.usd.schema.omni_nurec_types"):
    carb.log_warn("Could not enable omni.usd.schema.omni_nurec_types; "
                  "the NuRec terrain may not render.")

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
from pxr import UsdGeom, UsdLux, Gf, UsdPhysics

# --- Rusko Summer (NuRec neural-reconstruction export) world setup -----------
# Unlike the winter export, the summer NuRec ships ONLY the photoreal volume (no
# collision proxy mesh). It is therefore purely visual; the drone takes off from a
# flat, invisible ground collider we add ourselves. Pegasus references the world
# under /World/layout, so in-sim the NuRec volume prim is:
RUSKO_VOLUME_PATH = "/World/layout/gauss"   # the OmniNuRec Volume prim (carries xform + extent)
RUSKO_SPAWN_XY = (0.0, 0.0)                 # where the drone spawns (world XY)
RUSKO_GROUND_Z = 0.0                        # top of the flat ground collider / takeoff surface
# The reconstruction has NO real-world scale and its native extent is ~660k units
# across. Auto-fit: scale the volume uniformly (about its extent centre) so its widest
# horizontal extent maps to this many metres next to the (metric) drone.
RUSKO_TARGET_SPAN_M = 170.0
# Where the scaled volume's extent centre is placed in world. Tune Z to slide the
# terrain up/down until its ground meets RUSKO_GROUND_Z — we can't know the ground's
# height inside the volume without rendering it.
RUSKO_TERRAIN_CENTER = (0.0, 0.0, 0.0)
# Extra rotation (degrees, applied XYZ about the terrain centre) in case the capture's
# "up" isn't world +Z and the terrain looks tilted / upside-down. (0,0,0) = as-shipped.
RUSKO_EXTRA_EULER = (0.0, 0.0, 0.0)
RUSKO_ENV_NAME = "Rusko Summer"


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

        # Load the photoreal Rusko Summer NuRec world.
        self.pg.load_environment(SIMULATION_ENVIRONMENTS[RUSKO_ENV_NAME])

        # load_environment() is ASYNCHRONOUS: the USD is referenced over the next few
        # frames. Wait until the NuRec volume (and its extent) is actually composed in
        # before we scale it and add the ground, so the auto-fit has real bounds to work
        # with and physics doesn't initialize against a not-yet-loaded world.
        for _ in range(4000):
            simulation_app.update()
            vol = self.world.stage.GetPrimAtPath(RUSKO_VOLUME_PATH)
            if vol.IsValid() and vol.GetAttribute("extent").HasValue():
                break

        # Scale/orient the photoreal volume and add the flat ground + spawn surface
        # the drone takes off from (the summer NuRec ships no collider of its own).
        self._setup_rusko_summer()

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

        spawn_z = self._ground_top_z + 0.08  # rest the drone just above the ground surface
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

        # Frame the viewport on the scene at startup; the offset tracks the world span
        # so the whole terrain stays in view.
        fr = 0.6 * RUSKO_TARGET_SPAN_M
        self.pg.set_viewport_camera(
            [RUSKO_SPAWN_XY[0] + fr, RUSKO_SPAWN_XY[1] - fr, self._ground_top_z + 0.7 * fr],
            [RUSKO_SPAWN_XY[0], RUSKO_SPAWN_XY[1], self._ground_top_z],
        )

        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

    def _setup_rusko_summer(self):
        """Scale/orient the photoreal NuRec volume to a sane metric size and add the
        flat, invisible ground collider the drone takes off from. The summer NuRec is
        render-only (no collision proxy mesh), so the ground is ours, not the terrain's."""
        stage = self.world.stage
        self._ground_top_z = RUSKO_GROUND_Z

        # --- Auto-fit + place the photoreal NuRec volume ---
        vol = stage.GetPrimAtPath(RUSKO_VOLUME_PATH)
        if vol.IsValid():
            ext = vol.GetAttribute("extent").Get()
            if ext:
                mn = np.array(ext[0], dtype=float)
                mx = np.array(ext[1], dtype=float)
                c = (mn + mx) / 2.0
                span = float(max(mx[0] - mn[0], mx[1] - mn[1]))
                s = RUSKO_TARGET_SPAN_M / span if span > 1e-6 else 1.0
            else:
                carb.log_warn("Rusko: NuRec volume has no extent; loading unscaled.")
                c = np.zeros(3)
                s = 1.0
            cv = Gf.Vec3d(float(c[0]), float(c[1]), float(c[2]))
            tgt = Gf.Vec3d(float(RUSKO_TERRAIN_CENTER[0]),
                           float(RUSKO_TERRAIN_CENTER[1]),
                           float(RUSKO_TERRAIN_CENTER[2]))
            R = Gf.Matrix4d(1.0).SetRotate(
                Gf.Rotation(Gf.Vec3d(1, 0, 0), RUSKO_EXTRA_EULER[0])
                * Gf.Rotation(Gf.Vec3d(0, 1, 0), RUSKO_EXTRA_EULER[1])
                * Gf.Rotation(Gf.Vec3d(0, 0, 1), RUSKO_EXTRA_EULER[2]))
            S = Gf.Matrix4d(1.0).SetScale(Gf.Vec3d(s, s, s))
            # row-vector order: move extent centre to origin, rotate, scale, move to target
            M = (Gf.Matrix4d(1.0).SetTranslate(-cv) * R * S
                 * Gf.Matrix4d(1.0).SetTranslate(tgt))
            x = UsdGeom.Xformable(vol)
            x.ClearXformOpOrder()
            x.AddTransformOp().Set(M)
        else:
            carb.log_warn(f"Rusko NuRec volume not found at {RUSKO_VOLUME_PATH}")

        # --- Flat, invisible ground collider + takeoff surface ---
        # Cube spans [-1,1]; scale to a wide, thin slab whose TOP sits at RUSKO_GROUND_Z.
        ground = UsdGeom.Cube.Define(stage, "/World/rusko_ground")
        ground.GetSizeAttr().Set(2.0)
        half_thickness = 0.5
        half_extent = max(RUSKO_TARGET_SPAN_M, 100.0)
        gx = UsdGeom.Xformable(ground.GetPrim())
        gx.ClearXformOpOrder()
        gx.AddTranslateOp().Set(Gf.Vec3d(RUSKO_SPAWN_XY[0], RUSKO_SPAWN_XY[1],
                                         RUSKO_GROUND_Z - half_thickness))
        gx.AddScaleOp().Set(Gf.Vec3f(half_extent, half_extent, half_thickness))
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        UsdGeom.Imageable(ground.GetPrim()).MakeInvisible()

        # The NuRec volume is self-emissive, but add a dome light so the (unlit) ground
        # and any composited synthetic assets are visible too.
        dome = UsdLux.DomeLight.Define(stage, "/World/rusko_dome_light")
        dome.CreateIntensityAttr(1000.0)

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
