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
from pxr import UsdGeom, Gf

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
        for _ in range(2000):
            simulation_app.update()
            layout = self.world.stage.GetPrimAtPath("/World/layout")
            if layout.IsValid() and layout.GetChildren():
                break

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

        self.vehicle = Multirotor(
            "/World/quadrotor",
            ROBOTS['Pegasus'],
            0,
            [0.0, 0.0, 0.07],
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
        self.camera_offset = np.array([-20.0, 0.0, 5.0])

        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

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
