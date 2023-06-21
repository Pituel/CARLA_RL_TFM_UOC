import time
import numpy as np
import pygame
from simulation.connection import carla
from simulation.sensors import CameraSensor, CameraSensorEnv, CollisionSensor
from settings import *
import sys

# Functions to create the route are taken from latest version of CARLA (0.9.14)
sys.path.insert(0,'/Users/Pituel/Documents/Master/TFM/Carla/CARLA_Latest/WindowsNoEditor/PythonAPI/carla/agents/navigation')
sys.path.insert(0,'/Users/Pituel/Documents/Master/TFM/Carla/CARLA_Latest/WindowsNoEditor/PythonAPI/carla/agents')
sys.path.insert(0,'/Users/Pituel/Documents/Master/TFM/Carla/CARLA_Latest/WindowsNoEditor/PythonAPI/carla')
from global_route_planner import GlobalRoutePlanner

class CarlaEnvironment():

    def __init__(self, client, world) -> None:


        self.client = client
        self.world = world
        # Library of blueprints to get the car.
        self.blueprint_library = self.world.get_blueprint_library()
        # Map of the world (town2) to generate spawn ponts and route.
        self.map = self.world.get_map()
        self.action_space = self.get_action_space()
        # Show RGB Camera during training/test
        self.display_on = VISUAL_DISPLAY
        self.vehicle = None
        self.current_waypoint_index = 0
        
        # Objects to be kept alive
        self.camera_obj = None
        self.env_camera_obj = None
        self.collision_obj = None

        # Two very important lists for keeping track of our actors and their observations.
        self.sensor_list = []
        self.actor_list = []

        # Generate the route
        self.get_route(50, 60)


    def reset(self):
        """
        Reset the environment

        Returns:
            Data for trining/test (state). 
        """

        try:
            
            # Destroy the actors/sensors amd empty the list.
            if len(self.actor_list) != 0 or len(self.sensor_list) != 0:
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
                self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
                self.sensor_list.clear()
                self.actor_list.clear()
            self.remove_sensors()


            # Create the vehicle and spawn it in the start position.
            vehicle_bp = self.blueprint_library.filter(CAR_NAME)[0]
            transform = self.map.get_spawn_points()[50]
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
            self.actor_list.append(self.vehicle)


            # Semantic Segmentation camera Sensor.
            # Generate the images for the trainig/test.
            self.camera_obj = CameraSensor(self.vehicle)
            while(len(self.camera_obj.front_camera) == 0):
                time.sleep(0.0001)
            self.image_obs = self.camera_obj.front_camera.pop(-1)
            self.sensor_list.append(self.camera_obj.sensor)

            # RGB Camera sensor for Third person view.
            if self.display_on:
                self.env_camera_obj = CameraSensorEnv(self.vehicle)
                self.sensor_list.append(self.env_camera_obj.sensor)

            # Collision sensor
            self.collision_obj = CollisionSensor(self.vehicle)
            self.collision_history = self.collision_obj.collision_data
            self.sensor_list.append(self.collision_obj.sensor)

            # Initialize all the variables
            self.timesteps = 0
            self.rotation = self.vehicle.get_transform().rotation.yaw
            self.center_lane_deviation = 0.0
            self.target_speed = 22
            self.max_speed = 25.0
            self.max_distance_from_center = 3
            self.throttle = float(0.0)
            self.previous_steer = float(0.0)
            self.velocity = float(0.0)
            self.distance_from_center = float(0.0)
            self.angle = float(0.0)
            self.distance_covered = 0.0
            self.current_waypoint_index = 0

            # Create an array with the 5 variables to return.
            self.navigation_obs = np.array([self.throttle, self.velocity, self.previous_steer, self.distance_from_center, self.angle])

            # Clear the collision history in case the spawn of the vehicle is taken as a collision.            
            time.sleep(0.5)
            self.collision_history.clear()

            # Timer for the stop condition of the episode.
            self.episode_start_time = time.time()


            return [self.image_obs, self.navigation_obs]

        # In case of exception the actors and sensor are removed.
        except:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()


    def step(self, action_idx):
        """
        Take inputs gerated by neural network.

        Args:
            action_idx: action.

        Returns:
            Data for trining/test.
        """
        try:
            
            # Update steps in the episode.
            self.timesteps+=1

            # Velocity of the vehicle
            velocity = self.vehicle.get_velocity()
            # Convert mph tp km/h.
            self.velocity = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
            
            # Steer take the value of the given action.
            steer = self.action_space[action_idx]

            # Previos steer have more weight than current steer to make the movement of the vehicle smoother.
            # If velocity < 20 km/h, acelerate the vehicle.
            if self.velocity < 20.0:
                self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1, throttle=1.0))
                self.throttle = 1.0
            else:
                self.vehicle.apply_control(carla.VehicleControl(steer=self.previous_steer*0.9 + steer*0.1))
                self.throttle = 0.0
            self.previous_steer = steer

    
            # Collect ocllision data
            self.collision_history = self.collision_obj.collision_data            

            # Rotation of the vehicle in correlation to the map/lane (z axe)
            self.rotation = self.vehicle.get_transform().rotation.yaw

            # Location of the car
            self.location = self.vehicle.get_location()

            # Keep track of closest waypoint on the route
            waypoint_index = self.current_waypoint_index
            for _ in range(len(self.route_waypoints)):
                # Check if we passed the next waypoint along the route
                next_waypoint_index = waypoint_index + 1
                wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
                dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2],self.vector(self.location - wp.transform.location)[:2])
                if dot > 0.0:
                    waypoint_index += 1
                else:
                    break

            self.current_waypoint_index = waypoint_index
            # Calculate deviation from center of the lane
            self.current_waypoint = self.route_waypoints[ self.current_waypoint_index    % len(self.route_waypoints)]
            self.next_waypoint = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]
            self.distance_from_center = self.distance_to_line(self.vector(self.current_waypoint.transform.location),self.vector(self.next_waypoint.transform.location),self.vector(self.location))
            self.center_lane_deviation += self.distance_from_center

            # Get angle difference between closest waypoint and vehicle forward vector
            fwd    = self.vector(self.vehicle.get_velocity())
            wp_fwd = self.vector(self.current_waypoint.transform.rotation.get_forward_vector())
            self.angle  = self.angle_diff(fwd, wp_fwd)

            
            # Rewards
            done = False
            reward = 0

            if len(self.collision_history) != 0:
                done = True
                reward = -30
            elif self.distance_from_center > self.max_distance_from_center:
                done = True
                reward = -30
            elif self.episode_start_time + 10 < time.time() and self.velocity < 1.0:
                reward = -10
                done = True
            elif self.velocity > self.max_speed:
                reward = -10
                done = True

            # Interpolated from 1 when centered to 0 when 3 m from center
            centering_factor = max(1.0 - self.distance_from_center / self.max_distance_from_center, 0.0)
            # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
            angle_factor = max(1.0 - abs(self.angle / np.deg2rad(20)), 0.0)

            if not done:
                reward = 1.0 * centering_factor * angle_factor

            # if the vehicle reaches the goal, he recives reward 100.
            if self.current_waypoint_index >= len(self.route_waypoints) - 2:
                reward = reward + 100
                done = True

            # Wait for the ss-camera to take an image.
            while(len(self.camera_obj.front_camera) == 0):
                time.sleep(0.0001)

            # Take the last image.
            self.image_obs = self.camera_obj.front_camera.pop(-1)

            # Normalize the variables.
            normalized_velocity = self.velocity/self.target_speed
            normalized_distance_from_center = self.distance_from_center / self.max_distance_from_center
            normalized_angle = abs(self.angle / np.deg2rad(20))
            self.navigation_obs = np.array([self.throttle, self.velocity, normalized_velocity, normalized_distance_from_center, normalized_angle])
            
            # Remove everything that has been spawned in the env.
            if done:
                self.center_lane_deviation = self.center_lane_deviation / self.timesteps
                
                for sensor in self.sensor_list:
                    sensor.destroy()
                
                self.remove_sensors()
                
                for actor in self.actor_list:
                    actor.destroy()
            
            return [self.image_obs, self.navigation_obs], reward, done, [self.distance_covered, self.center_lane_deviation]

        # if there is an error, destroy the actors.
        except:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensor_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
            self.sensor_list.clear()
            self.actor_list.clear()
            self.remove_sensors()
            if self.display_on:
                pygame.quit()


    def get_route(self, start, end):
        """
        Define the waypoints of the route.

        Args:
            start: start of the route.
            end: end of the route

        Returns:
            Generate the route.
        """

        sampling_resolution = 0.5
        grp = GlobalRoutePlanner(self.map, sampling_resolution)
        self.route_waypoints = []
        spawn_points = self.world.get_map().get_spawn_points()
        a = carla.Location(spawn_points[start].location)
        b = carla.Location(spawn_points[end].location)
        w2 = grp.trace_route(a, b) 

        for v in w2:
            vp = self.map.get_waypoint(v[0].transform.location, project_to_road=True, lane_type=(carla.LaneType.Driving))
            self.route_waypoints.append(vp)

    def angle_diff(self, v0, v1):
        """
        Calculate angel between 2 vectos

        Args:
            v0: Vector 1: Velocity vector of vehicle.
            v1: Vector 2: Waypoint vector.

        Return:
            angel
        """
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        if angle > np.pi: angle -= 2 * np.pi
        elif angle <= -np.pi: angle += 2 * np.pi
        return angle


    def distance_to_line(self, A, B, p):
        """
        Calculate distance between the line A-B and p.

        Args:
            A: Vector 1. Current Waypoint.
            B: Vector 2. Next Waypoint.
            C: Vector 3. Vehicle

        Returns:
            Distance
        """
        num   = np.linalg.norm(np.cross(B - A, A - p))
        denom = np.linalg.norm(B - A)
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom


    def vector(self, v):
        """
        Get componentes of the given vector as array.

        Args:
            v: vector.

        Returns:
            Array with the components of the vector.
        """
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])


    def get_action_space(self):
        """
        Define the action space.

        Returns:
            Action space.
        """
        action_space = \
            np.array([
            -0.50,
            -0.30,
            -0.10,
            0.0,
            0.10,
            0.30,
            0.50
            ])
        return action_space

    def remove_sensors(self):
        """
        Clean up method
        """
        self.camera_obj = None
        self.collision_obj = None
        self.env_camera_obj = None
        self.front_camera = None
        self.collision_history = None

