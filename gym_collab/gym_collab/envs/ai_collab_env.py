import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import re

import json_numpy

from magnebot import ActionStatus, Arm
import cv2
from aiohttp import web
import pdb
import numpy as np
import time

from enum import Enum

import gymnasium as gym
from gymnasium import spaces

from .action import Action

class AICollabEnv(gym.Env):
    def __init__(self, use_occupancy, view_radius, robot_id, config):
        print("Initializing env " + str(robot_id) + "...")

        self.pcs = set()
        self.tracks_received = 0

        self.frame_queue = ""

        self.robot_id = robot_id
        self.use_occupancy = use_occupancy
        self.view_radius = view_radius
        self.centered_view = 0
        self.setup_ready = False
        self.confirm_time_threshold = 2  # Seconds to confirm

        self.ask_info_agents_str = "Ask for agent information to "
        self.ask_info_objects_str = "Ask for object information to "

        self.own_neighbors_info_entry = []
        self.waiting_output = False
        self.requested_output = []
        self.object_key_to_index = {}
        self.robot_key_to_index = {}
        self.goal_coords = []
        self.objects_in_goal = []
        self.last_sensed = []
        self.extra = {}
        
        #For debugging webrtc
        self.logger = logging.getLogger("pc")

        logging.basicConfig(level=logging.DEBUG)

        # Must get config from sim before setting up environments
        self.map_config = config

        for robot_idx, robot in enumerate(
                self.map_config['all_robots']):
            if robot[0] == str(self.robot_id):
                break
        del self.map_config['all_robots'][robot_idx] # Remove self from list of robots received

        self.gym_setup()
        
        self.action_message = []
        self.new_output = ()
        self.map = np.array([], dtype=np.int16)

        print("Finished initializing env " + str(robot_id) + "!")

    # When sharing robots information, remove the receiver robot info and
    # append yours
    def get_corrected_neighbors_info(self, target_id):
        corrected_neighbors_info = self.neighbors_info.copy()

        for ni_idx in range(len(corrected_neighbors_info)):
            if corrected_neighbors_info[ni_idx][0] == target_id:
                del corrected_neighbors_info[ni_idx]
                break

        corrected_neighbors_info.append(self.own_neighbors_info_entry)
        return corrected_neighbors_info
        
    def compute_real_distance(self,neighbor_location,ego_location):
        res = np.linalg.norm(np.array([neighbor_location[0],neighbor_location[1]]) - np.array([ego_location[0],ego_location[1]]))*self.map_config['cell_size']
        return res

    # GYM SETUP ###########################################################

    def gym_setup(self):
        map_size = self.map_config['num_cells'][0]
        self.action_space = spaces.Dict(
            {
                "action": spaces.Discrete(len(Action)),
                "item": spaces.Discrete(self.map_config['num_objects']),
                # Allow for 0
                "robot": spaces.Discrete(len(self.map_config['all_robots']) + 1),
                "message" : spaces.Text(min_length=0,max_length=100),
                "num_cells_move": spaces.Discrete(map_size),
            }
        )

        self.observation_space = spaces.Dict(
            {
                "frame": spaces.Box(low=-2, high=5, shape=(map_size, map_size), dtype=np.int16),
                "objects_held": spaces.Discrete(3, start=-1),
                "action_status": spaces.MultiDiscrete(np.array([2] * 4), dtype=np.int16),

                "item_output": spaces.Dict(
                    {
                        "item_weight": spaces.Discrete(len(self.map_config['all_robots'])+1),
                        "item_danger_level": spaces.Discrete(3),
                        "item_danger_confidence": spaces.Box(low=0, high=1, shape=(1,), dtype=float),
                        "item_location": spaces.Box(low=-np.infty, high=np.infty, shape=(2,), dtype=np.int16),
                        "item_time": spaces.Box(low=0, high=np.infty, shape=(1,), dtype=np.int16)
                    }
                ),
                "num_items": spaces.Discrete(self.map_config['num_objects'] + 1),

                "neighbors_output": spaces.Dict(
                    {
                        "neighbor_type": spaces.Discrete(3, start=-1),
                        "neighbor_location": spaces.Box(low=-np.infty, high=np.infty, shape=(2,), dtype=np.int16)
                    }

                ),
                # Strength starts from zero
                "strength": spaces.Discrete(len(self.map_config['all_robots']) + 2),
                "num_messages": spaces.Discrete(100)

                # "objects_danger_level" : spaces.Box(low=1,high=2,shape=(self.map_config['num_objects'],), dtype=int)
            }
        )

        self.goal_count = 0
        
    def reward(self, world_state, sensing_output):
        reward = 0
        occupancy_map = world_state[0]
        object_metadata = world_state[1]
        objects_held = world_state[2]

        if any(objects_held):
            for oh in objects_held:
                if oh:
                    self.extra['carrying_object'] = oh          
        elif 'carrying_object' in self.extra and self.extra['carrying_object']:
            self.extra['dropped_object'] = self.extra['carrying_object']
            self.extra['carrying_object'] = ""
                    
        #print(object_metadata, self.extra, objects_held)
        #Give a reward whenever agent drops object in desired area
        if 'dropped_object' in self.extra and self.extra['dropped_object'] and self.extra['dropped_object'] not in self.objects_in_goal: #Check that the agent just dropped object, and that this object has not been put in the goal area in the past
            for xy in self.goal_coords:
                key = str(xy[0]) + '_' + str(xy[1])
                if key in object_metadata:
                    for object_map in object_metadata[key]:
                        if isinstance(object_map, list) and self.extra['dropped_object'] in object_map:
                            
                            print(object_map)
                            
                            if object_map[2] == 2:
                                reward += 10 #Dangerous object dropped in safe area
                            elif object_map[2] == 1:
                                reward += -5 #Benign object dropped in safe area
                                
                            self.objects_in_goal.append(object_map[0])
            
        self.extra['dropped_object'] = ''
        
        ego_location = np.where(occupancy_map == 5)
        old_ego_location = np.where(sensing_output['occupancy_map'] == 5)
        
        if ego_location[0][0] != old_ego_location[0][0] or ego_location[1][0] != old_ego_location[1][0]:
            reward += -0.01
       
        return reward

    def step(self, action):
        world_state = self.old_output
        sensing_output = self.sensing_output 
        action_terminated = self.terminated
        action_truncated = self.truncated
        
        observation = {"frame": sensing_output["occupancy_map"],
                       "objects_held": sensing_output["objects_held"],
                       "action_status": np.array([int(action_terminated[0]),
                                                  int(action_truncated[0]),
                                                  int(action_terminated[1]),
                                                  int(action_truncated[1])], dtype=np.int64),
                       "num_items": len(self.object_info),
                       "item_output": sensing_output["item_output"],
                       "neighbors_output": sensing_output["neighbors_output"],
                       "num_messages": len(self.messages),
                       "strength": sensing_output["strength"]}  # Occupancy map

        info = {}
        info['map_metadata'] = sensing_output["objects_metadata"]
        info['messages'] = sensing_output["messages"]
        info['robot_key_to_index'] = self.robot_key_to_index
        info['object_key_to_index'] = self.object_key_to_index
        info['last_sensed'] = self.last_sensed
        info['time'] = float(world_state[7])
        
        #REWARD ESTIMATION
        if not self.truncated:
            reward = self.reward(world_state,sensing_output)
        else:
            reward = 0
        
        if len(self.objects_in_goal) == self.map_config['num_objects']: # When all dangerous objects are put in the middle the episode should terminate
            terminated = True
        else:
            terminated = False
            
        return observation, reward, terminated, self.truncated, info

    def reset():
        super().reset()
        map_size = self.map_config['num_cells'][0]
        
        self.truncated = False

        observation = {
            "frame": np.ones((map_size, map_size), dtype=np.int16)*(-2),
            "objects_held": -1,
            "action_status": np.zeros(4, dtype=np.int16),

            "item_output": {
                "item_weight": 0,
                "item_danger_level": 0,
                "item_danger_confidence": np.zeros(1, dtype=float),
                "item_location": np.ones(2, dtype=np.int16)*(-1),
                "item_time": np.zeros(1, dtype=np.int16)
            },
            "num_items": 0,

            "neighbors_output": {
                "neighbor_type": -1,
                "neighbor_location": np.ones(2, dtype=np.int16)*(-1)
            },
            "strength": 1,
            "num_messages": 0,
        }

        info = {}

        map_coordinates = np.arange(self.map_config['edge_coordinate'][0], self.map_config['edge_coordinate'][0]+self.map_config['cell_size']*map_size,1)
        
        half_cell = self.map_config['cell_size']/2
        for mc_idx in range(len(map_coordinates)):
            if map_coordinates[mc_idx] > 0:
                map_coordinates[mc_idx] += half_cell
            elif map_coordinates[mc_idx] < 0:
                map_coordinates[mc_idx] -= half_cell 
        
        self.goal_coords = [(x,y) for x,x_coord in enumerate(map_coordinates) for y,y_coord in enumerate(map_coordinates) if np.linalg.norm([x_coord,y_coord]) <= self.map_config['goal_radius']]
        
        self.objects_in_goal = []
        self.extra = {}
        
        self.internal_state = [self.State.take_action, self.State.take_sensing_action]
        self.internal_data = {}
        self.object_info = []
        self.neighbors_info = [[um[0], 0 if um[1] == 'human' else 1,0,0,-1] for um in self.map_config['all_robots']]
        self.robot_key_to_index = {self.neighbors_info[i][0]:i for i in range(len(self.neighbors_info))}
        self.own_neighbors_info_entry = [self.robot_id, 1, 0, 0, -1]
        self.last_sensed = []

        #self.sio.emit("disable")
        #self.sio.emit("reset") #self.sio.emit("reset_ai")
        print("Resetting agent")
        #while not self.agent_reset:
        #    continue

        #self.agent_reset = False

        self.map = np.array([], dtype=np.int16)
        
        #print("Waiting for location")
        #while self.map.size == 0:
        #    continue

        self.old_output = self.new_output
        print("Got location")

        observation["frame"] = self.map

        self.messages = []
        self.goal_count = 0

        return observation, info

    # ROBOT API ###########################################################

    # Forwarded magnebot API from
    # https://github.com/alters-mit/magnebot/blob/main/doc/manual/magnebot/actions.md\

    def turn_by(self, angle, aligned_at=1):
        return ["turn_by", str(angle), str(aligned_at)]

    def turn_to(self, target, aligned_at=1):
        return ["turn_to", str(target), str(aligned_at)]

    def move_by(self, distance, arrived_at=0.1):
        return ["move_by", str(distance), str(arrived_at)]

    def move_to(self, target, arrived_at=0.01, aligned_at=1, arrived_offset=0):
        return ["move_to", json.dumps(target), str(arrived_at),
                str(aligned_at), str(arrived_offset)]

    def reach_for(self, target, arm):
        return ["reach_for", str(target), str(arm.value)]

    def grasp(self, target, arm):
        return ["grasp", str(target), str(arm.value)]

    def drop(self, target, arm):
        return ["drop", str(target), str(arm.value)]

    def reset_arm(self, arm):
        return ["reset_arm", str(arm.value)]

    def reset_position(self):
        return ["reset_position"]

    def rotate_camera(self, roll, pitch, yaw):
        return ["rotate_camera", str(roll), str(pitch), str(yaw)]

    def look_at(self, target):
        return ["look_at", str(target)]

    def move_camera(self, position):
        return ["move_camera", str(position)]

    def reset_camera(self):
        return ["reset_camera"]

    def slide_torso(self, height):
        return ["slide_torso", str(height)]

    def danger_sensor_reading(self):
        return ["send_danger_sensor_reading"]

    def get_occupancy_map(self):
        print("get_occupancy_map")
        return ["send_occupancy_map"]

    def get_objects_held_status(self):
        return ["send_objects_held_status"]

    # CONTROLLER DEFINITION ###############################################

    # Controller states

    class State(Enum):
        take_action = 1
        waiting_ongoing = 2
        grasping_object = 3
        reseting_arm = 4
        reverse_after_dropping = 5
        take_sensing_action = 6
        wait_sensing = 7
        action_end = 8
        wait_get_objects = 9
        wait_get_agents = 10

    def take_action(self, action):

        terminated = False
        truncated = False
        objects_obs = []
        neighbors_obs = []

        # print(action)

        action_message, self.internal_state, self.internal_data, sensing_output, terminated, truncated = self.controller(
            action, self.old_output, self.internal_state, self.internal_data)

        self.sensing_output = sensing_output
        self.terminated = terminated
        self.truncated = truncated

        if action_message:  # Action message is an action to take by the robot that will be communicated to the simulator
            print("action", action_message)
            #self.sio.emit("ai_action", (action_message))
            self.action_message = action_message # use self.action_message and self.robot_id in simulation

    # Only works for occupancy maps not centered in magnebot

    def controller(self, complete_action, observations, internal_state, data):

        action_message = []
        movement_commands = 8
        grab_commands = 16

        occupancy_map = observations[0]
        objects_metadata = observations[1]
        objects_held = observations[2]
        danger_sensing_data = observations[3]
        action_status = observations[4]
        extra_status = observations[5]
        strength = observations[6]
        timer = observations[7]
        terminated = [False, False]
        truncated = [False, False]
        state = internal_state[0]
        sensing_state = internal_state[1]
        action = Action(complete_action["action"])

        sensing_output = {
            "occupancy_map": occupancy_map,
            "item_output": {
                "item_weight": 0,
                "item_danger_level": 0,
                "item_danger_confidence": np.zeros(1, dtype=float),
                "item_location": np.array([-1, -1], dtype=np.int16),
                "item_time": np.array([0], dtype=np.int16),
                },
            "messages": "",
            "neighbors_output": {
                "neighbor_type": -1,
                "neighbor_location": np.array([-1, -1], dtype=np.int16)},
            "objects_held": -1,
            "strength": strength,
            "objects_metadata": {}
        }

        ego_location = np.where(occupancy_map == 5)
        ego_location = np.array([ego_location[0][0], ego_location[1][0]])

        self.own_neighbors_info_entry[2] = float(ego_location[0])
        self.own_neighbors_info_entry[3] = float(ego_location[1])
        self.own_neighbors_info_entry[4] = float(timer)

        if state == self.State.take_action:

            #if action_status != ActionStatus.ongoing:
            #print("Original ", action)
            #print(occupancy_map)
        
            #self.action_status = -1
            data['timer_locomotion'] = time.time()
            
            if action.value < movement_commands:

                action_index = [
                    Action.move_up,
                    Action.move_right,
                    Action.move_down,
                    Action.move_left,
                    Action.move_up_right,
                    Action.move_up_left,
                    Action.move_down_right,
                    Action.move_down_left].index(action)

                original_location = np.copy(ego_location)

                ego_location = self.check_bounds(
                    action_index, ego_location, occupancy_map, complete_action["num_cells_move"])

                if not np.array_equal(ego_location, original_location):
                    target_coordinates = np.array(
                        self.map_config['edge_coordinate']) + ego_location * self.map_config['cell_size']
                    target = {
                        "x": target_coordinates[0],
                        "y": 0,
                        "z": target_coordinates[1]}
                    state = self.State.waiting_ongoing
                    data["next_state"] = self.State.action_end

                    action_message.append(self.move_to(target=target))
                else:
                    print("Movement not possible")
                    truncated[0] = True

            elif action.value < grab_commands:

                if any(objects_held):
                    print("Cannot grab another object")
                    truncated[0] = True
                else:
                    object_location = np.copy(ego_location)

                    action_index = [
                        Action.grab_up,
                        Action.grab_right,
                        Action.grab_down,
                        Action.grab_left,
                        Action.grab_up_right,
                        Action.grab_up_left,
                        Action.grab_down_right,
                        Action.grab_down_left].index(action)

                    object_location = self.check_bounds(action_index, object_location, occupancy_map,1)
                    key = str(object_location[0]) + '_' + str(object_location[1])
                    
                    if (not np.array_equal(object_location,ego_location)) and key in objects_metadata: #occupancy_map[object_location[0],object_location[1]] == 2:
                        print("Grabbing object")
                        #object_location = np.where(occupancy_map == 2)
                        #key = str(object_location[0][0]) + str(object_location[1][0])

                        
                        if isinstance(objects_metadata[key][0], list): 
                            object_id = objects_metadata[key][0][0]
                        else:
                            object_id = objects_metadata[key][0]
                            
                        target_coordinates = np.array(
                            self.map_config['edge_coordinate']) + object_location * self.map_config['cell_size']
                        target = {
                            "x": target_coordinates[0],
                            "y": 0,
                            "z": target_coordinates[1]}
                        action_message.append(self.move_to(target=target,arrived_offset=0.5))#self.turn_to(object_id))
                       
                        state = self.State.waiting_ongoing
                        data["next_state"] = self.State.grasping_object
                        data["object"] = object_id

                    else:
                        print("No object to grab")
                        truncated[0] = True

            elif action == Action.drop_object:


                if objects_held[0]:
                    action_message.append(self.drop(objects_held[0], Arm.left))

                    state = self.State.waiting_ongoing
                    data["next_state"] = self.State.reverse_after_dropping
                    #self.extra['dropped_object'] = objects_held[0]

                else:
                    print("No object to drop")
                    truncated[0] = True

            else: # not implemented
                pass

            '''
            else:
                truncated[0] = 1
                print("Ongoing truncated", timer)
            '''

        elif state == self.State.waiting_ongoing:
            # or action_status == ActionStatus.success:
            if action_status == ActionStatus.ongoing:
                print("waiting", action_status, timer)
                state = data["next_state"]
            elif time.time() - data['timer_locomotion'] > 5 and action_status == ActionStatus.success:
                print("waiting", action_status, timer)
                state = data["next_state"]
            elif time.time() - data['timer_locomotion'] > 10 and action_status == ActionStatus.collision:
                truncated[0] = True
                print("Action stuck", action_status)
            elif time.time() - data['timer_locomotion'] > 10: #Timer if it gets stuck
                truncated[0] = True
                print("Action stuck 2", action_status)
                    
        elif state == self.State.grasping_object:
            if action_status != ActionStatus.ongoing:
                state = self.State.waiting_ongoing
                print("waited to grasp objective")
                action_message.append(self.grasp(data["object"], Arm.left))
                del data["object"]
                data["next_state"] = self.State.reseting_arm

        elif state == self.State.reseting_arm:

            if action_status != ActionStatus.ongoing:
                print("waited to reset arm")
                action_message.append(self.reset_arm(Arm.left))
                state = self.State.waiting_ongoing
                data["next_state"] = self.State.action_end

        elif state == self.State.reverse_after_dropping:
            if action_status != ActionStatus.ongoing:
                print("waited to reverse after dropping")
                action_message.append(self.move_by(-0.5))
                state = self.State.waiting_ongoing
                data["next_state"] = self.State.action_end

        elif state == self.State.action_end:
            if action_status != ActionStatus.ongoing:
                print("action end", action_status, timer)
                terminated[0] = True

        if terminated[0] or truncated[0]:
            state = self.State.take_action

        # Check for sensing/comms actions
        if sensing_state == self.State.take_sensing_action:

            if action == Action.danger_sensing:
                action_message.append(self.danger_sensor_reading())
                sensing_state = self.State.wait_sensing
                self.waiting_output = True

            elif action == Action.get_occupancy_map:
                action_message.append(self.get_occupancy_map())
                sensing_state = self.State.wait_sensing
                self.waiting_output = True

            elif action == Action.get_objects_held:
                action_message.append(self.get_objects_held_status())
                sensing_state = self.State.wait_sensing
                self.waiting_output = True

            elif action == Action.check_item:
                if complete_action["item"] >= len(self.object_info):
                    truncated[1] = True
                else:
                    sensing_output["item_output"]["item_weight"] = self.object_info[complete_action["item"]][1]
                    sensing_output["item_output"]["item_danger_level"],sensing_output["item_output"]["item_danger_confidence"] = self.combine_danger_info(
                        self.object_info[complete_action["item"]][2])
                    sensing_output["item_output"]["item_location"] = np.array(self.object_info[complete_action["item"]][3:5])
                    sensing_output["item_output"]["item_time"] = np.array([self.object_info[complete_action["item"]][5]])
                    terminated[1] = True

            elif action == Action.check_robot:

                # If 0, it means broadcast so we ignore it
                if complete_action["robot"] > 0:
                    robot_idx = complete_action["robot"] - 1
                    sensing_output["neighbors_output"]["neighbor_type"] = self.neighbors_info[robot_idx][1]
                    sensing_output["neighbors_output"]["neighbor_location"] = self.neighbors_info[robot_idx][2:4]
                    terminated[1] = True
                else:
                    truncated[1] = True

            elif action == Action.get_messages:
                if self.messages:
                    sensing_output["messages"] = self.messages.copy()
                    self.messages = []
                    terminated[1] = True
                else:
                    truncated[1] = True

            elif action == Action.send_message:
            
                neighbors_dict = {}
                
                if complete_action["robot"] > 0:

                    robot_data = self.neighbors_info[complete_action["robot"] - 1]
                    if robot_data[4] >= 0 and np.linalg.norm((np.array([robot_data[2],robot_data[3]]) - np.array(ego_location))*self.map_config['cell_size']) < self.map_config['communication_distance_limit']:
                        neighbors_dict = {
                            robot_data[0]: "human" if not robot_data[1] else "ai"}
                else:
                    for robot_data in self.neighbors_info:
                        if robot_data[4] >= 0 and np.linalg.norm((np.array([robot_data[2],robot_data[3]]) - np.array(ego_location))*self.map_config['cell_size']) < self.map_config['communication_distance_limit']:
                            if not robot_data[1]:
                                robot_type = "human"
                            else:
                                robot_type = "ai"
                            neighbors_dict[robot_data[0]] = robot_type

                self.sio.emit(
                    "message", (complete_action["message"], timer, neighbors_dict))

                terminated[1] = True

            elif action == Action.request_item_info or action == Action.request_agent_info:

                if complete_action["robot"] > 0:
                    if action == Action.request_item_info:
                        message_str = self.ask_info_objects_str
                        sensing_state = self.State.wait_get_objects

                    elif action == Action.request_agent_info:
                        message_str = self.ask_info_agents_str
                        sensing_state = self.State.wait_get_agents

                    data['timer'] = time.time()

                    robot_data = self.neighbors_info[complete_action["robot"] - 1]
                    neighbors_dict = {
                        robot_data[0]: "human" if not robot_data[1] else "ai"}
                    data['agent_type'] = neighbors_dict[robot_data[0]]
                    #print(message_str + str(robot_data[0]), neighbors_dict)
                    self.sio.emit("message", (message_str +
                                              str(robot_data[0]), timer, neighbors_dict))
                else:
                    truncated[1] = True

            else: # not implemented
                pass

        elif sensing_state == self.State.wait_sensing:
            if not self.waiting_output:
                occupancy_map = self.requested_output[0]
                objects_metadata = self.requested_output[1]
                objects_held = self.requested_output[2]
                danger_sensing_data = self.requested_output[3]
                action_status = self.requested_output[4]
                extra_status = self.requested_output[5]
                strength = self.requested_output[6]
                timer = self.requested_output[7]

                if any(extra_status):
                    terminated[1] = True

                    if extra_status[0]:  # Occupancy map received

                        sensing_output["occupancy_map"] = occupancy_map
                        sensing_output["objects_metadata"] = objects_metadata
                        
                        #Update objects locations
                        object_locations = np.where((occupancy_map == 2) | (occupancy_map == 4) | (occupancy_map == 3))
                        #object_locations = np.array([object_locations[0][:],object_locations[1][:]])
                        

                        for ol_idx in range(len(object_locations[0])):
                        
                            key = str(
                                object_locations[0][ol_idx]) + '_' + str(object_locations[1][ol_idx])

                            for object_info in objects_metadata[key]: #One cell can have multiple objects
                            
                                if isinstance(object_info, list): #Only if it's an object
                                    self.update_objects_info(
                                        object_info[0], timer, {}, [
                                            object_locations[0][ol_idx], object_locations[1][ol_idx]], object_info[1], False)
  
                        # Update robots locations
                        robots_locations = np.where(occupancy_map == 3)
                        for ol_idx in range(len(robots_locations[0])):
                            key = str(
                                robots_locations[0][ol_idx]) + '_' + str(robots_locations[1][ol_idx])
                            self.update_neighbors_info(
                                objects_metadata[key][0], timer, [
                                    robots_locations[0][ol_idx], robots_locations[1][ol_idx]], False)

                    if extra_status[1]:  # Danger estimate received
                        self.last_sensed = []
                        for object_key in danger_sensing_data.keys():
                            
                            self.last_sensed.append(object_key)

                            if "sensor" not in danger_sensing_data[object_key]:
                                sensor_data = {}
                            else:
                                sensor_data = danger_sensing_data[object_key]['sensor']

                            try:
                                self.update_objects_info(
                                    object_key,
                                    danger_sensing_data[object_key]['time'],
                                    sensor_data,
                                    [
                                        danger_sensing_data[object_key]['location'][0],
                                        danger_sensing_data[object_key]['location'][2]],
                                    danger_sensing_data[object_key]['weight'],
                                    True)
                            except:
                                pdb.set_trace()

                    if extra_status[2]:  # Objects held
                        sensing_output["objects_held"] = int(
                            any(oh != 0 for oh in objects_held))

        elif sensing_state == self.State.wait_get_objects or sensing_state == self.State.wait_get_agents:

            if self.new_objects or self.new_neighbors:

                if data['agent_type'] == 'human':
                    coords_conversion = True
                else:
                    coords_conversion = False

                if sensing_state == self.State.wait_get_objects:

                    for ob_idx in range(len(self.new_objects)):
                        self.update_objects_info(
                            self.new_objects[ob_idx][0],
                            self.new_objects[ob_idx][5],
                            self.new_objects[ob_idx][2],
                            [
                                self.new_objects[ob_idx][3],
                                self.new_objects[ob_idx][4]],
                            self.new_objects[ob_idx][1],
                            coords_conversion)
                    self.new_objects = []

                elif sensing_state == self.State.wait_get_agents:

                    for ob_idx in range(len(self.new_neighbors)):
                        self.update_neighbors_info(
                            self.new_neighbors[ob_idx][0], self.new_neighbors[ob_idx][4], [
                                self.new_neighbors[ob_idx][2], self.new_neighbors[ob_idx][3]], coords_conversion)
                    self.new_agents = []

                terminated[1] = True
            elif time.time() - data['timer'] > self.confirm_time_threshold:
                truncated[1] = True

        if terminated[1] or truncated[1]:
            sensing_state = self.State.take_sensing_action
        
        return action_message, [
            state, sensing_state], data, sensing_output, terminated, truncated

    # Design an intelligent way of combining danger estimates
    def combine_danger_info(self, estimates):
        if estimates:
            key = list(estimates.keys())[0]
            return_value = estimates[key]['value']
            return_confidence = np.array([estimates[key]['confidence']])
        else:
            return_value = 0
            return_confidence = np.array([])

        return return_value,return_confidence

    # When receiving info about objects, update your internal representation
    def update_objects_info(self, object_key, timer,
                            danger_data, position, weight, convert_coordinates):

        if convert_coordinates:
            position = self.convert_to_grid_coordinates(position)

        known_object = False

        for ob_idx, ob in enumerate(self.object_info):
            if ob[0] == object_key:

                if danger_data:
                    self.object_info[ob_idx][2].update(danger_data)

                if ob[5] < timer:  # If data is fresh
                    self.object_info[ob_idx][3] = float(position[0])
                    self.object_info[ob_idx][4] = float(position[1])
                    self.object_info[ob_idx][5] = float(timer)
                known_object = True
                break
        if not known_object:

            self.object_info.append([object_key,int(weight),danger_data,float(position[0]),float(position[1]),float(timer)])
            self.object_key_to_index[object_key] = len(self.object_info)-1
          
    #When receiving info about robots, update your internal representation  
    def update_neighbors_info(self, agent_key, timer, position, convert_coordinates):

        if convert_coordinates:
            position = self.convert_to_grid_coordinates(position)

        for ob_idx, ob in enumerate(self.neighbors_info):

            if ob[0] == str(agent_key) and (ob[4] == -1 or ob[4] < timer):
                self.neighbors_info[ob_idx][2] = float(position[0])
                self.neighbors_info[ob_idx][3] = float(position[1])
                self.neighbors_info[ob_idx][4] = float(timer)
                break

    # This AI controller relies on having coordinates relative to the grid
    # world, which is not the same system the simulator uses
    def convert_to_grid_coordinates(self, location):
        min_pos = self.map_config['edge_coordinate']
        multiple = self.map_config['cell_size']
        pos_new = [round((location[0] + abs(min_pos[0])) / multiple),
                   round((location[1] + abs(min_pos[1])) / multiple)]

        return pos_new

    #Convert from grid coordinates to real
    def convert_to_real_coordinates(self, position):
        min_pos = self.map_config['edge_coordinate']
        multiple = self.map_config['cell_size']
        pos_new = [position[0]*multiple - abs(min_pos[0]), position[1]*multiple - abs(min_pos[1])]

        return pos_new
    
    # Check movement limits
    def check_bounds(self, action_index, location, occupancy_map, num_cells):
        try:
            if action_index == 0:  # Up
                if location[0] < occupancy_map.shape[0] - 1:
                    location[0] += 1*num_cells
            elif action_index == 1:  # Right
                if location[1] > 0:
                    location[1] -= 1*num_cells
            elif action_index == 2:  # Down
                if location[0] > 0:
                    location[0] -= 1*num_cells
            elif action_index == 3:  # Left
                if location[1] < occupancy_map.shape[1] - 1:
                    location[1] += 1*num_cells
            elif action_index == 4:  # Up Right
                if location[0] < occupancy_map.shape[0] - 1 and location[1] > 0:
                    location += [1*num_cells, -1*num_cells]
            elif action_index == 5:  # Up Left
                if location[0] < occupancy_map.shape[0] - \
                        1 and location[1] < occupancy_map.shape[1] - 1:
                    location += [1*num_cells, 1*num_cells]
            elif action_index == 6:  # Down Right
                if location[0] > 0 and location[1] > 0:
                    location += [-1*num_cells, -1*num_cells]
            elif action_index == 7:  # Down Left
                if location[0] > 0 and location[1] < occupancy_map.shape[1] - 1:
                    location += [-1*num_cells, 1*num_cells]
        except:
            pdb.set_trace()

        return location
