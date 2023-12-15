import numpy as np
import pdb
import cv2
import time
import socketio
import csv
import json_numpy
from scipy.spatial.transform import Rotation
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.object_manager import ObjectManager
from tdw.add_ons.ui import UI
from tdw.quaternion_utils import QuaternionUtils
from tdw.output_data import OutputData, Images, ScreenPosition, Transforms, Raycast, Keyboard as KBoard, Framerate
from tdw.add_ons.keyboard import Keyboard
from magnebot import Magnebot, Arm, ActionStatus, ImageFrequency
from magnebot.util import get_default_post_processing_commands

from tdw.add_ons.occupancy_map import OccupancyMap
from tdw.add_ons.logger import Logger

from PIL import Image, ImageOps

import datetime
import json
import os
import random
import sys

from enhanced_magnebot import Enhanced_Magnebot
from stats import Stats

##############################################################
# Global variables
##############################################################

#Dimension of our camera view
width = 0
height = 0

num_users = 0
num_ais = 2

global_refresh_sensor = 0

dateTime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

extra_commands = []
duration = []

game_finished = 0

# Given three collinear points p, q, r, the function checks if 
# point q lies on line segment 'pr' 
def onSegment(p, q, r):
    if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and 
           (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
        return True
    return False
  
def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise
      
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/ 
    # for details of below formula. 
      
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
    if (val > 0):
          
        # Clockwise orientation
        return 1
    elif (val < 0):
          
        # Counterclockwise orientation
        return 2
    else:
          
        # Collinear orientation
        return 0
  
# The main function that returns true if 
# the line segment 'p1q1' and 'p2q2' intersect.
def doIntersect(p1,q1,p2,q2):
      
    # Find the 4 orientations required for 
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
  
    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True
  
    # Special Cases
  
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True
  
    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True
  
    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True
  
    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True
  
    # If none of the cases
    return False

class Simulation(Controller):
    def __init__(self, args, cfg, port: int = 1071, check_version: bool = False, launch_build: bool = False, restart=False):
    
        super().__init__(port=port, check_version=check_version, launch_build=launch_build, restart=restart)
         
        self.keys_set = []
        self.local = args.local
        self.options = args
        self.cfg = cfg
        
        self.reset = False
        self.reset_message = False
        
        self.timer = 0
        self.real_timer = time.time()
        self.timer_start = self.timer
        self.reset_number = 0
        self.payment = 7
        self.timer_limit = float(self.cfg['timer'])
            
        self.ai_skip_frames = int(self.cfg['ai_skip_frames'])
        
        self.scenario_size = 20
        self.wall_length = 6
        
        self.scenario = self.options.scenario
        
        if self.options.seed > -1:
            self.seed_value = self.options.seed
        else:
            self.seed_value = random.randrange(sys.maxsize)
            
        random.seed(self.seed_value)
        print("SEED:", self.seed_value)

        # Creating occupancy map
        self.static_occupancy_map = OccupancyMap()       

        logger = Logger(path="tdw_log.txt", overwrite=True)

        #Add-ons
        self.add_ons.extend([self.static_occupancy_map, logger])

        # Create the scene.
        commands = self.generate_create_scene_commands()
        print("Creating scene")
        self.communicate(commands)
        print("Created scene")

        self.static_occupancy_map.generate(cell_size=self.cfg['cell_size']) #Get occupancy map only with walls
        self.communicate([])
        out_of_bounds = np.where(self.static_occupancy_map.occupancy_map == 2)
        self.static_occupancy_map.occupancy_map[out_of_bounds[0],out_of_bounds[1]] = -1

        self.ai_original_spawn_positions = [{"x": -2, "y": 0, "z": 1.1},{"x": -2, "y": 0, "z": 2.1}, {"x": -2, "y": 0, "z": 3.1}, {"x": -3, "y": 0, "z": 0.1}, {"x": -2, "y": 0, "z": 0.1},{"x": -2, "y": 0, "z": -1.1}, {"x": -2, "y": 0, "z": -2.1},{"x": -2, "y": 0, "z": -3.1},{"x": -3, "y": 0, "z": -1.1},{"x": -3, "y": 0, "z": -2.1}, {"x": -3, "y": 0, "z": 1.1}, {"x": -3, "y": 0, "z": 2.1}, {"x": -3.5, "y": 0, "z": 0.5}, {"x": -3.5, "y": 0, "z": 1.5}, {"x": -3.5, "y": 0, "z": 2.5}, {"x": -3.5, "y": 0, "z": 3.5}, {"x": -3.5, "y": 0, "z": -2.5}, {"x": -3.5, "y": 0, "z": -3.5}]
                
        self.create_agents()
        print("Created agents")
        print("There are " + str(num_ais) + " AI agents!")

        self.object_manager: ObjectManager = ObjectManager()
        self.add_ons.append(self.object_manager)
        
        commands = self.generate_populate_world_commands()
        self.communicate(commands)
        print("Populated world")
        
        self.communicate({"$type": "set_post_process", "value": False})

        self.send_init_data()

        self.object_type_coords_map = np.copy(self.static_occupancy_map.occupancy_map)
        self.object_attributes_id = {}

        self.update_environment_object_positions()    
        self.update_graspable_object_positions()
        self.update_magnebot_positions()

        global game_finished

        self.frame_num = 0
        self.disabled_robots = []
        self.goal_counter = 0

        self.events = [{
            "a": False,
            "b": False, 
            "c": False, 
            "d": False, 
            "e": False, 
            "f": False, 
            "g": False, 
            }, {
            "a": False,
            "b": False, 
            "c": False, 
            "d": False, 
            "e": False, 
            "f": False, 
            "g": False, 
        }]

    def send_init_data(self):
        # Occupancy map info
        extra_config = {}

        extra_config['edge_coordinate'] = [float(self.static_occupancy_map.positions[0,0,0]),float(self.static_occupancy_map.positions[0,0,1])]
        extra_config['cell_size'] = self.cfg['cell_size']
        extra_config['num_cells'] = self.static_occupancy_map.occupancy_map.shape
        extra_config['num_objects'] = len(self.graspable_objects)
        extra_config['all_robots'] = [(self.robot_names_translate[str(um.robot_id)],um.controlled_by) for um in self.ai_magnebots]
        extra_config['timer_limit'] = self.timer_limit
        extra_config['strength_distance_limit'] = self.cfg['strength_distance_limit']
        extra_config['communication_distance_limit'] = self.cfg['communication_distance_limit']
        extra_config["goal_radius"] = self.cfg["goal_radius"]

        translated_ai_magnebots_ids = [self.robot_names_translate[robot_id] for robot_id in self.ai_magnebots_ids]
        #print(*translated_ai_magnebots_ids, sep = ", ") 
        self.initial_config_map = extra_config

    def create_agents(self):
        self.robot_names_translate = {}
        self.ai_magnebots = []
        
        self.ai_spawn_positions = self.ai_original_spawn_positions.copy()
        random.shuffle(self.ai_spawn_positions)

        # Create AI magnebots
        for ai_idx in range(num_ais):  
            robot_id = self.get_unique_id()                                 
            self.ai_magnebots.append(Enhanced_Magnebot(robot_id=robot_id, position=self.ai_spawn_positions[ai_idx],image_frequency=ImageFrequency.never, controlled_by='ai'))
            self.ai_magnebots[ai_idx].view_radius = 5
            self.robot_names_translate[str(robot_id)] = chr(ord('A') + ai_idx)

        self.add_ons.extend(self.ai_magnebots)

        self.ai_magnebots_ids = [str(um.robot_id) for um in self.ai_magnebots]

        #print(*self.ai_magnebots_ids, sep=", ")
        
    # Create the TDW scene environment
    def generate_create_scene_commands(self):
        fps = int(self.cfg['fps'])     
        if fps:    
            commands.append({"$type": "set_target_framerate", "framerate": fps})
            
        if self.scenario == 1:
            self.scenario_size = 20
            self.wall_length = 6
            cell_size = self.cfg['cell_size']
            wall_width = 0.5
            
            
            wall1_1 = [{"x": self.wall_length, "y": idx+1} for idx in range(self.wall_length-2)]
            wall1_2 = [{"x": idx+1, "y": self.wall_length} for idx in range(self.wall_length-2)]
            
            wall2_1 = [{"x": self.scenario_size-(self.wall_length), "y": idx+1} for idx in range(self.wall_length-2)]
            wall2_2 = [{"x": self.scenario_size-(idx+1), "y": self.wall_length} for idx in range(self.wall_length-2)]
            
            wall3_1 = [{"x": self.wall_length, "y": self.scenario_size-(idx+1)} for idx in range(self.wall_length-2)]
            wall3_2 = [{"x": idx+1, "y": self.scenario_size-(self.wall_length)} for idx in range(self.wall_length-2)]
            
            wall4_1 = [{"x": self.scenario_size-(self.wall_length), "y": self.scenario_size-(idx+1)} for idx in range(self.wall_length-2)]
            wall4_2 = [{"x": self.scenario_size-(idx+1), "y": self.scenario_size-(self.wall_length)} for idx in range(self.wall_length-2)]
            
            self.walls = [[[wall[0]['x']+wall_width-self.scenario_size/2,wall[0]['y']+wall_width-self.scenario_size/2],[wall[-1]['x']+wall_width-self.scenario_size/2,wall[-1]['y']+wall_width-self.scenario_size/2]] for wall in [wall1_1,wall1_2,wall2_1,wall2_2,wall3_1,wall3_2,wall4_1,wall4_2]]

            
            commands = [
                        {"$type": "load_scene", "scene_name": "ProcGenScene"},
                        TDWUtils.create_empty_room(self.scenario_size, self.scenario_size),
                        self.get_add_material("parquet_long_horizontal_clean",
                                              library="materials_high.json"),
                        {"$type": "set_screen_size",
                         "width": width,
                         "height": height},
                        {"$type": "rotate_directional_light_by",
                         "angle": 30,
                         "axis": "pitch"},
                         {"$type": "create_interior_walls", "walls": [*wall1_1,*wall1_2]},
                         {"$type": "create_interior_walls", "walls": [*wall2_1,*wall2_2]},
                         {"$type": "create_interior_walls", "walls": [*wall3_1,*wall3_2]},
                         {"$type": "create_interior_walls", "walls": [*wall4_1,*wall4_2]},
                        {"$type": "set_floor_color", "color": {"r": 1, "g": 1, "b": 1, "a": 1}},
                        {"$type": "set_proc_gen_walls_color", "color": {"r": 1, "g": 1, "b": 0, "a": 1.0}}]
            
        number_angles = int(float(self.cfg["goal_radius"])*2*np.pi)
        
        for n in range(number_angles):
            angle_side = 2*n*np.pi/number_angles
            xn = float(self.cfg["goal_radius"])*np.cos(angle_side)
            zn = float(self.cfg["goal_radius"])*np.sin(angle_side)
        
            commands.append({"$type": "add_position_marker",
                                     "position": {"x": xn, "y": 0.01, "z": zn},
                                     "scale": 0.2,
                                     "shape":"circle"})
                  
        commands.append({"$type": "send_framerate", "frequency": "always"})                             
                                     
        return commands

    #Used to create all objects
    def generate_populate_world_commands(self):
        self.graspable_objects = []
        self.object_dropping = []
        self.occupancy_map_request = []
        self.objects_held_status_request = []
        self.danger_sensor_request = []
        self.ai_status_request = []
        self.raycast_request = []
        self.queue_perception_action = []
        self.extra_keys_pressed = []
        
        self.object_names_translate = {}
        
        self.already_collected = []
        
        self.timer = 0 
        self.real_timer = time.time()
        self.timer_start = self.timer

        self.terminate = False
    
        self.required_strength = {}
        self.danger_level = {} 
        self.dangerous_objects = []
        self.env_objects = []

        commands = []
        
        #Instantiate and locate objects
        if self.scenario == 1:        
            cell_size = self.cfg['cell_size']
            
            max_coord = int(self.scenario_size/2)-1
            object_models = {'iron_box':5}

            possible_ranges = [np.arange(self.scenario_size/2-self.wall_length+cell_size*1.5,self.scenario_size/2-cell_size*0.5,cell_size),np.arange(self.scenario_size/2-self.wall_length+cell_size*1.5,self.scenario_size/2-cell_size*0.5,cell_size)]
            
            possible_locations = [[i, j] for i in possible_ranges[0] for j in possible_ranges[1]]
            
            modifications = [[1.0,1.0],[-1.0,1.0],[1.0,-1.0],[-1.0,-1.0]]
            
            #danger_prob = self.cfg['danger_prob']*100
            danger_prob = 50 # 50%

            final_coords = {objm: [] for objm in object_models.keys()}

            for m in modifications:
                possible_locations_temp = possible_locations.copy()
                for fc in final_coords.keys():
                    for n_obj in range(object_models[fc]):
                        location = random.choice(possible_locations_temp)
                        possible_locations_temp.remove(location)
      
                        final_coords[fc].append(np.array(location)*m)

            object_index = 0
            for fc in final_coords.keys():
                for c in final_coords[fc]:
                    
                    possible_weights = list(range(1,num_users+num_ais+1))
                    weights_probs = [100]*len(possible_weights)
                    
                    for p_idx in range(len(possible_weights)):
                        if not p_idx:
                            weights_probs[p_idx] /= 2
                        elif p_idx == len(possible_weights)-1:
                            weights_probs[p_idx] = weights_probs[p_idx-1]
                        else:
                            weights_probs[p_idx] = weights_probs[p_idx-1]/2
                    
                    if len(possible_weights) == 1:
                        weight = 1
                    else:
                        weight = int(random.choices(possible_weights,weights=weights_probs)[0])
                    danger_level = random.choices([1,2],weights=[100-danger_prob,danger_prob])[0]
                    
                    try:
                        commands.extend(self.instantiate_object(fc,{"x": c[0], "y": 0, "z": c[1]},{"x": 0, "y": 0, "z": 0},1000,danger_level,weight, object_index))
                    except:
                        pdb.set_trace()
                    object_index += 1

            #Create environment objects
            self.env_objects.append(self.get_unique_id())
            
            commands.extend(self.get_add_physics_object(model_name="zenblocks",
                                             object_id=self.env_objects[-1],
                                             position={"x": max_coord-self.wall_length+cell_size/2, "y": 0, "z": max_coord-cell_size*1.5},
                                             default_physics_values=False,
                                             mass=1000,
                                             scale_mass=False,
                                             rotation={"x": 0, "y": 0, "z": 0}))

            self.env_objects.append(self.get_unique_id())

            commands.extend(self.get_add_physics_object(model_name="amphora_jar_vase",
                                             object_id=self.env_objects[-1],
                                             position={"x": 2.5*cell_size-max_coord, "y": 0, "z": self.wall_length-max_coord+cell_size/2},
                                             default_physics_values=False,
                                             mass=1000,
                                             scale_mass=False,
                                             rotation={"x": 0, "y": 0, "z": 0}))
                                             
            self.env_objects.append(self.get_unique_id())
            
            commands.extend(self.get_add_physics_object(model_name="linen_dining_chair",
                                             object_id=self.env_objects[-1],
                                             position={"x": 2.5*cell_size-max_coord, "y": 0, "z": max_coord-self.wall_length+cell_size/2},
                                             default_physics_values=False,
                                             mass=1000,
                                             scale_mass=False,
                                             rotation={"x": 0, "y": 0, "z": 0}))
                                             
            self.env_objects.append(self.get_unique_id())
            
            commands.extend(self.get_add_physics_object(model_name="cgaxis_models_50_12_vray",
                                             object_id=self.env_objects[-1],
                                             position={"x": max_coord-self.wall_length+cell_size/2, "y": 0, "z": 2.5*cell_size-max_coord},
                                             default_physics_values=False,
                                             mass=1000,
                                             scale_mass=False,
                                             rotation={"x": 0, "y": 0, "z": 0}))
        
        # Add post-processing.
        commands.extend(get_default_post_processing_commands())     
        
        self.target = {}
        
        commands.append({"$type": "add_compass_rose"})
        
        return commands

    #Function to instantiate objects
    def instantiate_object(self, model_name, position, rotation, mass, danger_level, required_strength, object_index):
        if self.options.single_weight:
            required_strength = self.options.single_weight
        #required_strength = 2
        if self.options.single_danger:
            danger_level = 2
        #danger_level = 2

        object_id = self.get_unique_id()
        self.graspable_objects.append(object_id)
        self.object_names_translate[object_id] = str(object_index)
        self.required_strength[object_id] = required_strength
        self.danger_level[object_id] = danger_level
        command = self.get_add_physics_object(model_name=model_name,
                                         object_id=object_id,
                                         position=position,
                                         rotation=rotation,
                                         default_physics_values=False,
                                         mass=mass,
                                         scale_mass=False)
        if self.danger_level[object_id] == 2:
            self.dangerous_objects.append(object_id)
            print("Dangerous object:\t", object_index, "\tweight: ", required_strength)
        else:
            print("Benign object:\t\t", object_index, "\tweight: ", required_strength)

        return command
    
    def get_involved_teammates(self, current_teammates, object_id): #Assign contributions to each teammate
        robot_ids = []
        total_time_spent = []
    
        for robot in current_teammates[object_id].keys():
            robot_ids.append(robot)
            total_time_spent.append(current_teammates[object_id][robot])
            
        sort_indices = np.argsort(np.array(total_time_spent)).tolist()
        sort_indices.reverse()
        
        return robot_ids, sort_indices

    def danger_sensor_reading(self, robot_id):
        all_ids = self.ai_magnebots_ids
        all_magnebots = self.ai_magnebots
        idx = all_ids.index(str(robot_id))
        ego_magnebot = all_magnebots[idx]
    
        near_items_pos = []
        near_items_idx = []
        danger_estimates = {}
        possible_danger_levels = [1,2]
        
        if ego_magnebot.refresh_sensor >= global_refresh_sensor: #Check if our sensor is refreshed
            ego_magnebot.refresh_sensor = 0
            
            ego_magnebot.stats.sensor_activation += 1
            
            for o_idx,o in enumerate(self.graspable_objects): #Sensor only actuates over objects that are in a certain radius
                if np.linalg.norm(self.object_manager.transforms[o].position -
                        ego_magnebot.dynamic.transform.position) < int(self.cfg['sensing_radius']) and not any(doIntersect([self.object_manager.transforms[o].position[0],self.object_manager.transforms[o].position[2]],[ego_magnebot.dynamic.transform.position[0],ego_magnebot.dynamic.transform.position[2]],[self.walls[w_idx][0][0],self.walls[w_idx][0][1]],[self.walls[w_idx][-1][0],self.walls[w_idx][-1][1]]) for w_idx in range(len(self.walls))):
                        
                    near_items_idx.append(len(all_ids)+o_idx)
                    near_items_pos.append(TDWUtils.array_to_vector3(self.object_manager.transforms[o].position))
                    actual_danger_level = self.danger_level[o]
                    
                    o_translated = self.object_names_translate[o]
                    
                    if o_translated not in ego_magnebot.item_info:
                        ego_magnebot.item_info[o_translated] = {}
                        
                    ego_magnebot.item_info[o_translated]['weight'] = int(self.required_strength[o])
                    
                    if 'sensor' not in ego_magnebot.item_info[o_translated]:
                        ego_magnebot.item_info[o_translated]['sensor'] = {}
                    
                    
                    robot_id_translated = self.robot_names_translate[str(ego_magnebot.robot_id)]
                    
                    #Get danger estimation, value and confidence level
                    if robot_id_translated not in ego_magnebot.item_info[o_translated]['sensor']:
                        possible_danger_levels_tmp = possible_danger_levels.copy()
                        possible_danger_levels_tmp.remove(actual_danger_level)
                    
                        if actual_danger_level == 1:
                            true_accuracy = ego_magnebot.p11
                        elif actual_danger_level == 2:
                            true_accuracy = ego_magnebot.p22
                        #estimate_accuracy = float(np.random.uniform(0.5, 1))
                    
                        danger_estimate = random.choices([actual_danger_level,*possible_danger_levels_tmp],weights=[true_accuracy*100,(1-true_accuracy)*100])
                        danger_estimates[o_translated] = danger_estimate[0]
                        
                        if danger_estimate[0] == 1:
                            estimate_accuracy = ego_magnebot.p11
                            other_estimate_accuracy = 1-ego_magnebot.p22
                        elif danger_estimate[0] == 2:
                            estimate_accuracy = ego_magnebot.p22
                            other_estimate_accuracy = 1-ego_magnebot.p11
                        
                        ego_magnebot.item_info[o_translated]['sensor'][robot_id_translated] = {}
                        ego_magnebot.item_info[o_translated]['sensor'][robot_id_translated]['value'] = int(danger_estimate[0])
                        ego_magnebot.item_info[o_translated]['sensor'][robot_id_translated]['confidence'] = (estimate_accuracy*0.5)/(0.5*estimate_accuracy + 0.5*other_estimate_accuracy)

                        
                    else: #If we already have a danger estimation reuse that one
                        danger_estimates[o_translated] = ego_magnebot.item_info[o_translated]['sensor'][robot_id_translated]['value']
                        
                        
                    ego_magnebot.item_info[o_translated]['time'] = self.timer
                    ego_magnebot.item_info[o_translated]['location'] = self.object_manager.transforms[o].position.tolist()
                        
            #If objects were detected
            if near_items_pos:
                ego_magnebot.danger_estimates = danger_estimates

        return idx, ego_magnebot.item_info
            
    def send_occupancy_map(self, magnebot_id):
        self.occupancy_map_request.append(magnebot_id)
        
        
    def send_objects_held_status(self,magnebot_id):
        self.objects_held_status_request.append(magnebot_id)
        
    def send_danger_sensor_reading(self, magnebot_id):
        self.danger_sensor_request.append(magnebot_id)
            
    def get_occupancy_map(self, magnebot_id):
        m_idx = self.ai_magnebots_ids.index(magnebot_id)
        locations_magnebot_map = {}
        
        magnebots_locations = np.where(self.object_type_coords_map == 3)
        locations_magnebot_map = {str(j):[magnebots_locations[0][i],magnebots_locations[1][i]] for i in range(len(magnebots_locations[0])) for j in self.object_attributes_id[str(magnebots_locations[0][i])+'_'+str(magnebots_locations[1][i])]}

        try:
            x = locations_magnebot_map[self.robot_names_translate[str(magnebot_id)]][0]
            y = locations_magnebot_map[self.robot_names_translate[str(magnebot_id)]][1]
        except:
            pdb.set_trace()
        
        
        if magnebot_id in self.occupancy_map_request and self.ai_magnebots[m_idx].view_radius:
            
            # print("Sending occupancy map!")
            self.occupancy_map_request.remove(magnebot_id)
        

            view_radius = self.ai_magnebots[m_idx].view_radius
            
            x_min = max(0,x-view_radius)
            y_min = max(0,y-view_radius)
            x_max = min(self.object_type_coords_map.shape[0]-1,x+view_radius)
            y_max = min(self.object_type_coords_map.shape[1]-1,y+view_radius)
            #limited_map = np.zeros_like(self.static_occupancy_map.occupancy_map)
            
            #Magnebot is at the center of the occupancy  map always or not
            if self.ai_magnebots[m_idx].centered_view:
                limited_map = np.zeros((view_radius*2+1,view_radius*2+1)) #+1 as we separately count the row/column where the magnebot is currently in
                #limited_map[:,:] = self.static_occupancy_map.occupancy_map[x_min:x_max+1,y_min:y_max+1]
                limited_map[:,:] = self.object_type_coords_map[x_min:x_max+1,y_min:y_max+1]
                objects_locations = np.where(limited_map > 1)
                reduced_metadata = {}
                limited_map[x-x_min,y-y_min] = 5

                for ol in range(len(objects_locations[0])):
                    rkey = str(objects_locations[0][ol]+x_min)+'_'+str(objects_locations[1][ol]+y_min)
                    rkey2 = str(objects_locations[0][ol])+'_'+str(objects_locations[1][ol])
                    reduced_metadata[rkey2] = self.object_attributes_id[rkey]
            else:
                limited_map = np.ones_like(self.object_type_coords_map)*(-2)
                limited_map[[0,limited_map.shape[0]-1],:] = -1
                limited_map[:,[0,limited_map.shape[1]-1]] = -1
                limited_map[x_min:x_max+1,y_min:y_max+1] = self.object_type_coords_map[x_min:x_max+1,y_min:y_max+1]
                objects_locations = np.where(limited_map > 1)
                reduced_metadata = {}
                limited_map[x,y] = 5
                
                for ol in range(len(objects_locations[0])):
                    rkey = str(objects_locations[0][ol])+'_'+str(objects_locations[1][ol])
                    reduced_metadata[rkey] = self.object_attributes_id[rkey]

        else: #If only location of robot
            limited_map = np.ones_like(self.object_type_coords_map)*(-2)
            limited_map[x,y] = 5
            reduced_metadata = {}
            
            #objects_locations = np.where(self.object_type_coords_map[x-1:x+2:y-1:y+2] == 2) #Object metadata only for surrounding objects
            objects_locations = np.where(self.object_type_coords_map > 1)
            for ol in range(len(objects_locations[0])):
                rkey = str(objects_locations[0][ol])+'_'+str(objects_locations[1][ol])
                reduced_metadata[rkey] = self.object_attributes_id[rkey]

        return limited_map, reduced_metadata
        
    def get_objects_held_state(self, magnebot_id):
        # Check the objects held in each arm
        objects_held = [0,0]
        m_idx = self.ai_magnebots_ids.index(magnebot_id)
        
        if magnebot_id in self.objects_held_status_request:
            self.objects_held_status_request.remove(magnebot_id)
            
        for arm_idx, arm in enumerate([Arm.left,Arm.right]):
            
            if self.ai_magnebots[m_idx].dynamic.held[arm].size > 0:
                objects_held[arm_idx] = self.object_names_translate[int(self.ai_magnebots[m_idx].dynamic.held[arm][0])]

        return objects_held
    
    def track_objects_carried(self, all_magnebots, all_idx, item_info):
        for arm in [Arm.left,Arm.right]:
            if all_magnebots[all_idx].dynamic.held[arm].size > 0:
                object_id = all_magnebots[all_idx].dynamic.held[arm][0]
                object_id_translated = self.object_names_translate[object_id]
                
                if object_id_translated not in all_magnebots[all_idx].item_info:
                    all_magnebots[all_idx].item_info[object_id_translated] = {}
                
                all_magnebots[all_idx].item_info[object_id_translated]["time"] = self.timer
                all_magnebots[all_idx].item_info[object_id_translated]["location"] = self.object_manager.transforms[object_id].position.tolist()
                all_magnebots[all_idx].item_info[object_id_translated]["weight"] = int(self.required_strength[object_id])
                

                if object_id not in all_magnebots[all_idx].stats.objects_in_goal and np.linalg.norm(self.object_manager.transforms[object_id].position[[0,2]]) < float(self.cfg["goal_radius"]):
                    if "sensor" not in all_magnebots[all_idx].item_info[object_id_translated]:
                        all_magnebots[all_idx].item_info[object_id_translated]["sensor"] = {}
                    
                    robot_id_translated = self.robot_names_translate[str(all_magnebots[all_idx].robot_id)]
                    
                    if robot_id_translated not in all_magnebots[all_idx].item_info[object_id_translated]["sensor"]:
                        all_magnebots[all_idx].item_info[object_id_translated]["sensor"][robot_id_translated] = {}
                        
                    all_magnebots[all_idx].item_info[object_id_translated]["sensor"][robot_id_translated]['value'] = int(self.danger_level[object_id])
                    all_magnebots[all_idx].item_info[object_id_translated]["sensor"][robot_id_translated]['confidence'] = 1
                        
                    if self.object_names_translate[object_id] not in self.already_collected and self.object_names_translate[object_id] not in all_magnebots[all_idx].stats.objects_in_goal:
                    
                        all_magnebots[all_idx].stats.objects_in_goal.append(self.object_names_translate[object_id])
                        self.already_collected.append(self.object_names_translate[object_id])
                        
                        if int(self.required_strength[object_id]) > 1: #Add teammates contribution
                        
                            robot_ids,sort_indices = self.get_involved_teammates(all_magnebots[all_idx].current_teammates, self.object_names_translate[object_id])
                            
                            for sidx in range(int(self.required_strength[object_id])-1):
                                all_magnebots[robot_ids[sort_indices[sidx]]].stats.objects_in_goal.append(self.object_names_translate[object_id])
                    
                        if self.danger_level[object_id] == 2 and self.object_names_translate[object_id] not in all_magnebots[all_idx].stats.dangerous_objects_in_goal:
                            all_magnebots[all_idx].stats.dangerous_objects_in_goal.append(self.object_names_translate[object_id])
                            
                            
                            if int(self.required_strength[object_id]) > 1: # Add teammates contribution
                            
                                for sidx in range(int(self.required_strength[object_id])-1):
                                    all_magnebots[robot_ids[sort_indices[sidx]]].stats.dangerous_objects_in_goal.append(self.object_names_translate[object_id])
                          
                item_info[self.object_names_translate[object_id]] = all_magnebots[all_idx].item_info[self.object_names_translate[object_id]]
    
    def update_environment_object_positions(self):
        min_pos = float(self.static_occupancy_map.positions[0,0,0])
        for o in self.env_objects:
            pos = self.object_manager.transforms[o].position
            pos_new = [round((pos[0]+abs(min_pos))/self.cfg['cell_size']), round((pos[2]+abs(min_pos))/self.cfg['cell_size'])]
            self.object_type_coords_map[pos_new[0],pos_new[1]] = 1

    def update_graspable_object_positions(self):
        min_pos = float(self.static_occupancy_map.positions[0,0,0])
        held_objects = []
        for um in self.ai_magnebots:
            for arm in [Arm.left,Arm.right]:
                if um.dynamic.held[arm].size > 0:
                    held_objects.append(um.dynamic.held[arm][0])

        for o in self.graspable_objects:
            pos = self.object_manager.transforms[o].position
            pos_new = [round((pos[0]+abs(min_pos))/self.cfg['cell_size']), round((pos[2]+abs(min_pos))/self.cfg['cell_size'])]
            # 2 is for objects
            
            ob_type = 2
            
            if o in held_objects:
                ob_type = 4
                
            try:
                if not (self.object_type_coords_map[pos_new[0],pos_new[1]] and ob_type == 4): # If robot carrying object, do not substitute cell with held object
                    self.object_type_coords_map[pos_new[0],pos_new[1]] = ob_type
            except:
                pdb.set_trace()

            if str(pos_new[0])+'_'+str(pos_new[1]) not in self.object_attributes_id:
                self.object_attributes_id[str(pos_new[0])+'_'+str(pos_new[1])] = []
            self.object_attributes_id[str(pos_new[0])+'_'+str(pos_new[1])].append((self.object_names_translate[o],self.required_strength[o],int(self.danger_level[o])))

    def update_magnebot_positions(self):
        min_pos = float(self.static_occupancy_map.positions[0,0,0])
        for o in self.ai_magnebots:
            pos = o.dynamic.transform.position
            pos_new = [round((pos[0]+abs(min_pos))/self.cfg['cell_size']), round((pos[2]+abs(min_pos))/self.cfg['cell_size'])]
            
            if o.disabled: #3 if active, 1 if not active
                self.object_type_coords_map[pos_new[0],pos_new[1]] = 1
            else:
                self.object_type_coords_map[pos_new[0],pos_new[1]] = 3
                
            if str(pos_new[0])+'_'+str(pos_new[1]) not in self.object_attributes_id:
                self.object_attributes_id[str(pos_new[0])+'_'+str(pos_new[1])] = []
            self.object_attributes_id[str(pos_new[0])+'_'+str(pos_new[1])].append((self.robot_names_translate[str(o.robot_id)]))
        
    def run(self, robot_id, action_message):
        # Re initialize event flags
        self.events = [{
            "a": False,
            "b": False, 
            "c": False, 
            "d": False, 
            "e": False, 
            "f": False, 
            "g": False, 
            }, {
            "a": False,
            "b": False, 
            "c": False, 
            "d": False, 
            "e": False, 
            "f": False, 
            "g": False, 
        }]

        print('New command:', action_message, robot_id)
        agent_id = list(self.robot_names_translate.keys())[list(self.robot_names_translate.values()).index(robot_id)]
        ai_agent_idx = self.ai_magnebots_ids.index(agent_id)
        ai_agent = self.ai_magnebots[ai_agent_idx]

        for actions in action_message:
            if actions[0] == 'send_occupancy_map':
                function = self.send_occupancy_map
            elif actions[0] == 'send_objects_held_status':
                function = self.send_objects_held_status
            elif actions[0] == 'send_danger_sensor_reading':
                function = self.send_danger_sensor_reading
            elif actions[0] == 'turn_by':
                ai_agent.turn_by(float(actions[1]), aligned_at=float(actions[2]))
            elif actions[0] == 'turn_to':
                object_id = list(self.object_names_translate.keys())[list(self.object_names_translate.values()).index(actions[1])]
                ai_agent.turn_to(object_id, aligned_at=float(actions[2]))
            elif actions[0] == 'move_by':
                ai_agent.move_by(float(actions[1]), arrived_at=float(actions[2]))
            elif actions[0] == 'move_to':
                ai_agent.move_to(json.loads(actions[1]), arrived_at=float(actions[2]), aligned_at=float(actions[3]), arrived_offset=float(actions[4]))
            elif actions[0] == 'reach_for':
                object_id = list(self.object_names_translate.keys())[list(self.object_names_translate.values()).index(actions[1])]
                ai_agent.reach_for(object_id, Arm(int(actions[2])))
            elif actions[0] == 'grasp':
                object_id = list(self.object_names_translate.keys())[list(self.object_names_translate.values()).index(actions[1])]
                if ai_agent.strength < self.required_strength[object_id]:
                    pass
                else:
                    if all(object_id not in um.dynamic.held[arm] for um in [*self.user_magnebots,*self.ai_magnebots] for arm in [Arm.left,Arm.right]):
                        extra_commands.append({"$type": "set_mass", "mass": 1, "id": object_id})
                        duration.append(1)
                        ai_agent.grasp(object_id, Arm(int(actions[2])))
                        ai_agent.stats.grab_attempts += 1
                        ai_agent.resetting_arm = True
            elif actions[0] == 'drop':
                object_id = list(self.object_names_translate.keys())[list(self.object_names_translate.values()).index(actions[1])]
                arm = Arm(int(actions[2]))
                ai_agent.drop(object_id, arm)
                self.object_dropping.append([int(ai_agent.dynamic.held[arm][0]),time.time(),ai_agent,arm])
            elif actions[0] == 'reset_arm':
                ai_agent.reset_arm(Arm(int(actions[1])))
            elif actions[0] == 'rotate_camera':
                ai_agent.rotate_camera(float(actions[1]), float(actions[2]), float(actions[3]))
            elif actions[0] == 'look_at':
                ai_agent.look_at(json.loads(actions[1]))
            elif actions[0] == 'move_camera':
                function = ai_agent.move_camera(json.loads(actions[1]))
            elif actions[0] == 'reset_camera':
                ai_agent.reset_camera()
            elif actions[0] == 'slide_torso':
                ai_agent.slide_torso(float(actions[1]))
            elif actions[0] == 'reset_position':
                ai_agent.reset_position()
            else:
                continue
            
            if 'send_' in actions[0]:
                self.queue_perception_action.append([function,[agent_id],self.ai_skip_frames])

        commands = []
        sim_elapsed_time = 0
        all_ids = self.ai_magnebots_ids
        all_magnebots = self.ai_magnebots
        start_time = time.time()
        
        screen_positions = {"position_ids":[], "positions":[]}
        
        # Prepare occupancy maps and associated metadata
        # object_attributes_id stores the ids of the objects and magnebots
        # object_type_coords_map creates a second occupancy map with objects and magnebots
        self.object_type_coords_map = np.copy(self.static_occupancy_map.occupancy_map)
        self.object_attributes_id = {}

        self.update_environment_object_positions()    
        self.update_graspable_object_positions()
        self.update_magnebot_positions()

        # Some extra commands to send and when to remove them
        to_eliminate = []
        for ex_idx in range(len(extra_commands)):
            duration[ex_idx] -= 1
            if not duration[ex_idx]:
                to_eliminate.append(ex_idx)
            commands.append(extra_commands[ex_idx])
        
        to_eliminate.reverse()
        for e in to_eliminate:
            del duration[e]
            del extra_commands[e]
            
        # We update timer
        mins = 0
        secs = 0

        mins, remainder = divmod(self.timer, 60)
        secs,millisecs = divmod(remainder,1)

        object_info_update = []

        #Update all stats related with closeness of magnebots, like strength factor
        #Iterate over all magnebots
        for idx in range(len(all_magnebots)):
            robot_id = all_magnebots[idx].robot_id
            all_magnebots[idx].strength = 1
            company = {}
            
            #print(all_magnebots[idx].screen_positions["position_ids"])
            
            pos1 = all_magnebots[idx].dynamic.transform.position[[0,2]] #Not interested in height coordinate
            
            if all_magnebots[idx].last_position.size > 0:
                all_magnebots[idx].stats.distance_traveled += float(np.linalg.norm(pos1 - all_magnebots[idx].last_position))
            
            all_magnebots[idx].last_position = pos1
            
            if all_magnebots[idx].robot_id not in self.disabled_robots and all_magnebots[idx].disabled:
            
                #start a fire
                commands.append(self.get_add_visual_effect(name="fire", 
                                    position=TDWUtils.array_to_vector3(all_magnebots[idx].dynamic.transform.position),
                                    effect_id=self.get_unique_id()))
            
                self.disabled_robots.append(all_magnebots[idx].robot_id)
                robot_id_translated = self.robot_names_translate[str(all_magnebots[idx].robot_id)]
                
                for object_id_translated in all_magnebots[idx].item_info.keys():
                    if "sensor" in all_magnebots[idx].item_info[object_id_translated] and robot_id_translated in all_magnebots[idx].item_info[object_id_translated]["sensor"]:
                        all_magnebots[idx].stats.objects_sensed += 1
                        
                        #len(list(all_magnebots[idx].item_info[object_id_translated]["sensor"].keys()))
                        
                number_dangerous_objects_in_goal = 0
                number_benign_objects_in_goal = 0
                number_dropped_objects = 0
                
                for ob in all_magnebots[idx].stats.objects_in_goal:
                
                    object_id = list(self.object_names_translate.keys())[list(self.object_names_translate.values()).index(ob)]
                
                    weight = self.required_strength[object_id]
                
                    if ob in all_magnebots[idx].stats.dangerous_objects_in_goal:
                        number_dangerous_objects_in_goal += 1/weight                            
                    else:
                        number_benign_objects_in_goal += 1/weight
                        
                        
                for ob in all_magnebots[idx].stats.dropped_outside_goal:
                    object_id = list(self.object_names_translate.keys())[list(self.object_names_translate.values()).index(ob)]
                    weight = self.required_strength[object_id]
                    number_dropped_objects += 1/weight
                    
                
                number_dangerous_objects_in_goal = len(all_magnebots[idx].stats.dangerous_objects_in_goal)
                all_magnebots[idx].stats.quality_work = max(0,(number_dangerous_objects_in_goal - number_benign_objects_in_goal - number_dropped_objects)/len(self.dangerous_objects))
                
                #Last magnebot to be sent stats. We also send to everyone the stats related to team performance
                if len(self.disabled_robots) == len(all_magnebots):
                
                    game_finished += 1
                
                    failure_reasons = {self.robot_names_translate[str(am.robot_id)]:am.stats.failed for am in all_magnebots}
                    
                    team_quality_work = sum([am.stats.quality_work for am in all_magnebots])

                    num_sensed = {}
                    
                    for go in self.graspable_objects:
                        o = self.object_names_translate[go]
                        
                        for am in all_magnebots:
                            robot_id_translated2 = self.robot_names_translate[str(am.robot_id)]
                            if o in am.item_info and "sensor" in am.item_info[o] and robot_id_translated2 in am.item_info[o]["sensor"]:
                                if o not in num_sensed.keys():
                                    num_sensed[o] = []
                                num_sensed[o].append(robot_id_translated2)
                        
                    all_stats = []
                    
                    end_time = all_magnebots[idx].stats.end_time
                        
                    if end_time > self.timer_limit:
                        end_time = self.timer_limit
                    
                    team_speed_work = self.timer_limit/(max(self.timer_limit/10,min(self.timer_limit,end_time)))
                    team_achievement = team_speed_work * team_quality_work
                    
                    for idx2 in range(len(all_magnebots)):
                    
                        if team_quality_work+human_team_effort > 0:
                            individual_contribution = (all_magnebots[idx2].stats.quality_work + all_magnebots[idx2].stats.effort)/(team_quality_work+human_team_effort) #human quality work
                        else:
                            individual_contribution = 0
                            
                        individual_payment = actual_human_team_payment*individual_contribution
                    
                        all_magnebots[idx2].stats.team_dangerous_objects_in_goal = self.self.goal_counter
                        all_magnebots[idx2].stats.total_dangerous_objects = len(self.dangerous_objects)
                        all_magnebots[idx2].stats.team_end_time = end_time
                        all_magnebots[idx2].stats.team_failure_reasons = failure_reasons
                        all_magnebots[idx2].stats.team_quality_work = team_quality_work
                        all_magnebots[idx2].stats.team_speed_work = team_speed_work
                        all_magnebots[idx2].stats.team_achievement = team_achievement
                        all_magnebots[idx2].stats.human_team_effort = human_team_effort
                        all_magnebots[idx2].stats.team_payment = actual_human_team_payment
                        all_magnebots[idx2].stats.individual_payment = individual_payment
                        
                        all_stats.append(all_magnebots[idx2].stats.__dict__)
                        
                        
                    if self.options.log_results:
                        if not os.path.exists(self.options.log_results):
                            os.makedirs(self.options.log_results)
                          
                        log_results_f = open(self.options.log_results + "/" + dateTime + "_" + str(game_finished) + '_results.txt', "w")
                        json.dump({"results": all_stats, "seed": self.seed_value}, log_results_f)
                        log_results_f.close()
                        
            for idx2 in range(len(all_magnebots)):
                if idx == idx2:
                    continue
                        
                pos2 = all_magnebots[idx2].dynamic.transform.position[[0,2]]
                distance = np.linalg.norm(pos1 - pos2)
                
                if not all_magnebots[idx2].disabled and distance < int(self.cfg['view_radius']) and not any(doIntersect([pos2[0],pos2[1]],[pos1[0],pos1[1]],[self.walls[w_idx][0][0],self.walls[w_idx][0][1]],[self.walls[w_idx][-1][0],self.walls[w_idx][-1][1]]) for w_idx in range(len(self.walls))): #Check if robot is visible    
                    # Event C:
                    self.events[0]["c"] = True

                if not all_magnebots[idx2].disabled and distance < int(self.cfg['strength_distance_limit']) and not any(doIntersect([pos2[0],pos2[1]],[pos1[0],pos1[1]],[self.walls[w_idx][0][0],self.walls[w_idx][0][1]],[self.walls[w_idx][-1][0],self.walls[w_idx][-1][1]]) for w_idx in range(len(self.walls))): #Check if robot is close enough to influence strength
                    # Event D:
                    self.events[0]["d"] = True
                    
                    all_magnebots[idx].strength += 1 #Increase strength
                    
                    robot2 = self.robot_names_translate[str(all_magnebots[idx2].robot_id)]
                    if robot2 not in all_magnebots[idx].stats.time_with_teammates:
                        all_magnebots[idx].stats.time_with_teammates[robot2] = 0
                    all_magnebots[idx].stats.time_with_teammates[robot2] += float(sim_elapsed_time)
                    
                    
                    for arm in [Arm.left,Arm.right]:
                        if all_magnebots[idx].dynamic.held[arm].size > 0:
                            object_id = self.object_names_translate[all_magnebots[idx].dynamic.held[arm][0]]
                            if object_id not in all_magnebots[idx].current_teammates:
                                all_magnebots[idx].current_teammates[object_id] = {}
                                
                            if idx2 not in all_magnebots[idx].current_teammates[object_id]:
                                all_magnebots[idx].current_teammates[object_id][idx2] = 0
                            all_magnebots[idx].current_teammates[object_id][idx2] += float(sim_elapsed_time)
                    
                    
                if distance < int(self.cfg['communication_distance_limit']): #Check if robot is close enough to communicate
                
                    company[self.robot_names_translate[str(all_magnebots[idx2].robot_id)]] = (all_magnebots[idx2].controlled_by, pos2.tolist(), float(distance), all_magnebots[idx2].disabled) #Add information about neighbors
                    
            all_magnebots[idx].company = company 
            
            #Refresh danger level sensor             
            if all_magnebots[idx].refresh_sensor < global_refresh_sensor:
                all_magnebots[idx].refresh_sensor += 1              

            for arm in [Arm.left,Arm.right]:
            
                if all_magnebots[idx].action.status == ActionStatus.cannot_reach or all_magnebots[idx].action.status == ActionStatus.failed_to_grasp:
                    all_magnebots[idx].grasping = False
                    all_magnebots[idx].resetting_arm = False
                    all_magnebots[idx].reset_arm(arm)
            
                if all_magnebots[idx].dynamic.held[arm].size > 0:
                    
                    
                    if all_magnebots[idx].resetting_arm and all_magnebots[idx].action.status != ActionStatus.ongoing:
                        all_magnebots[idx].resetting_arm = False
                        
                        if str(all_magnebots[idx].robot_id) in self.user_magnebots_ids:
                            all_magnebots[idx].reset_arm(arm)
            
                        print("Resetting arm")
                        
                        all_magnebots[idx].stats.grabbed_objects += 1
                        
                    #Drop object if strength decreases
                    if self.required_strength[all_magnebots[idx].dynamic.held[arm][0]] > all_magnebots[idx].strength:
                        
                        self.object_dropping.append([int(all_magnebots[idx].dynamic.held[arm][0]),time.time(),all_magnebots[idx], arm])
                    
                        grasped_object = all_magnebots[idx].dynamic.held[arm][0]
                        all_magnebots[idx].drop(target=grasped_object, arm=arm)
                        all_magnebots[idx].grasping = False
                        
                        if np.linalg.norm(self.object_manager.transforms[grasped_object].position[[0,2]]) >= float(self.cfg["goal_radius"]):
                            robot_ids,sort_indices = self.get_involved_teammates(all_magnebots[idx].current_teammates, self.object_names_translate[grasped_object])
                        
                            for sidx in range(int(self.required_strength[grasped_object])-1):
                                all_magnebots[robot_ids[sort_indices[sidx]]].stats.dropped_outside_goal.append(self.object_names_translate[grasped_object])
                        
                            all_magnebots[idx].stats.dropped_outside_goal.append(self.object_names_translate[grasped_object])
                        
            #Transmit ai controlled robots status
            if not self.local and all_magnebots[idx] in self.ai_magnebots:
                if all_magnebots[idx].action.status != all_magnebots[idx].past_status:
                    all_magnebots[idx].past_status = all_magnebots[idx].action.status
            
        #If all robots have been disabled
        if len(self.disabled_robots) == len(all_magnebots):
            pass # Do something when game finishes
        
        try:
            resp = self.communicate(commands)
        except Exception as e:
            print("Error communication")
            pdb.set_trace()
            #self.sio.emit("sim_crash", self.timer)
            if hasattr(e, 'message'):
                print(e.message)
            else:
                print(e)
                #if not self.local:
                #    self.reset_agents()
                return -1
        
        self.initializing = False

        commands.clear()

        # Process output data from build
        for i in range(len(resp) - 1):
            
            r_id = OutputData.get_data_type_id(resp[i])

            if r_id == "quit":
                print("Error quitting")
                #self.sio.emit("sim_crash", self.timer)
                return -1
                            
        if self.object_dropping:
            to_remove = []
            for o_idx,od in enumerate(self.object_dropping): # Increase mass of objects when dropped
                if not od[0] in od[2].dynamic.held[od[3]]:
                    if time.time() - od[1] > 1:
                        try:
                            commands.append({"$type": "set_mass", "mass": 1000, "id": od[0]})
                            to_remove.append(o_idx)
                        except:
                            print("grasped object2", od[0])
                else:
                    print("Can't drop object")
                    self.object_dropping[o_idx][1] = time.time()
                
            if to_remove:
                to_remove.reverse()
                for tr in to_remove:
                    del self.object_dropping[tr]    

        # Game ends when all dangerous objects are left in the rug
        self.goal_counter = 0
        
        for sd in self.dangerous_objects:
            if np.linalg.norm(self.object_manager.transforms[sd].position[[0,2]]) < float(self.cfg["goal_radius"]):
                self.goal_counter += 1

        if self.goal_counter == len(self.dangerous_objects):
            # Event G:
            self.events[0]["g"] = True
        
        to_remove = []
        #Execute delayed actions
        for qa_idx in range(len(self.queue_perception_action)):
            if not self.queue_perception_action[qa_idx][2]:
                self.queue_perception_action[qa_idx][2] -= 1
            else:
                self.queue_perception_action[qa_idx][0](*self.queue_perception_action[qa_idx][1])
                to_remove.append(qa_idx)
                
        to_remove.reverse()
        for tr in to_remove:
            del self.queue_perception_action[tr]
        
        #Send frames to virtual cameras in system and occupancy maps if required, and all outputs needed
        for m_idx, magnebot_id in enumerate(self.ai_magnebots_ids):
            all_idx = all_ids.index(str(magnebot_id)) 
            if not all_magnebots[all_idx].skip_frames:
                #Occupancy maps
                extra_status = [0]*3
                
                if magnebot_id in self.occupancy_map_request:
                        extra_status[0] = 1
                
                if not all_magnebots[all_idx].disabled:
                    limited_map, reduced_metadata = self.get_occupancy_map(magnebot_id)
                else:
                    limited_map = []
                    reduce_metadata = {}
                
                
                if magnebot_id in self.objects_held_status_request:
                    extra_status[2] = 1
                
                objects_held = self.get_objects_held_state(magnebot_id)
                
                # Event F:
                if objects_held[0] or objects_held[1]:
                    self.events[0]["f"] = True
                
                if magnebot_id in self.danger_sensor_request:
                    _,item_info = self.danger_sensor_reading(magnebot_id)
                    self.danger_sensor_request.remove(magnebot_id)
                    extra_status[1] = 1
                else:
                    item_info = {}
                    
                #Track objects being carried
                if not item_info:
                    self.track_objects_carried(all_magnebots, all_idx, item_info)
            
                #if all_idx in self.ai_status_request:
                ai_status = all_magnebots[all_idx].past_status.value
                    
                #print(limited_map)
                if not self.local and not all_magnebots[all_idx].last_output:
                    if any(extra_status):
                        print("Sending extra status")
                    self.this_action_output = (all_idx, limited_map, reduced_metadata, objects_held, item_info, ai_status, extra_status, all_magnebots[all_idx].strength, self.timer, all_magnebots[all_idx].disabled)
                    if all_magnebots[all_idx].disabled:
                        all_magnebots[all_idx].last_output = True

            idx += 1
        
        #If timer expires end simulation, else keep going
        if self.timer_limit and self.timer_limit-self.timer <= 0:
            for idx,_ in enumerate(all_magnebots):
                all_magnebots[idx].disabled = True
                all_magnebots[idx].stats.end_time = self.timer
                
        else:
            new_time = time.time()
            self.timer += new_time-self.real_timer
            self.real_timer = new_time

        #Reset world
        if self.reset:
            print("Resetting...")
            
            #resp = self.communicate({"$type": "destroy_all_objects"})
            self.reset_world()
            
            self.reset_number += 1
            
            if not self.local:
                self.reset_agents()
            
            self.reset = False
            #pdb.set_trace()
            print("Reset complete")
            sim_elapsed_time = 0
            self.disabled_robots = []

        self.frame_num +=1
            
        #self.communicate({"$type": "terminate"})
        
        ego_magnebot = ai_agent
        # high level events
        # Event A:  
        if np.linalg.norm(ego_magnebot.dynamic.transform.position[[0,2]]) <= float(self.cfg["goal_radius"]):
            self.events[0]["a"] = True
        
        # Event B:
        for o_idx,o in enumerate(self.graspable_objects): #Sensor only actuates over objects that are in a certain radius
            if np.linalg.norm(self.object_manager.transforms[o].position -
                ego_magnebot.dynamic.transform.position) < int(self.cfg['sensing_radius']) and np.linalg.norm(self.object_manager.transforms[object_id].position[[0,2]]) < float(self.cfg["goal_radius"]):
                self.events[0]["b"] = True
                break
        
        # Event E
        for o_idx,o in enumerate(self.graspable_objects): #Sensor only actuates over objects that are in a certain radius
            if np.linalg.norm(self.object_manager.transforms[o].position -
                ego_magnebot.dynamic.transform.position) < int(self.cfg['viewing_radius']) and np.linalg.norm(self.object_manager.transforms[object_id].position[[0,2]]) < float(self.cfg["goal_radius"]):
                self.events[0]["e"] = True
                break

        return self.this_action_output

    def reset_agents(self):
        for m in self.ai_magnebots:
            pass
            #self.sio.emit("agent_reset", (self.robot_names_translate[str(m.robot_id)],self.timer))

    def reset_world(self):
        
        if self.options.seed > -1:
            self.seed_value = self.options.seed
        else:
            self.seed_value = random.randrange(sys.maxsize)
        random.seed(self.seed_value)
        print("SEED:", self.seed_value)
        
        commands = []
        
        for go in self.graspable_objects:
            commands.append({"$type": "destroy_object", "id": go})
            
        for env_obj in self.env_objects:
            commands.append({"$type": "destroy_object", "id": env_obj})
            
        self.communicate(commands)
        
        commands = []
        
        self.ai_spawn_positions = self.ai_original_spawn_positions.copy()
        random.shuffle(self.ai_spawn_positions)
        
        for u_idx in range(len(self.ai_magnebots)):
            self.ai_magnebots[u_idx].reset(position=self.ai_spawn_positions[u_idx])    

        self.object_manager.initialized = False
        commands = []
        commands.extend(self.generate_create_scene_commands())
            
        commands.extend(self.generate_populate_world_commands())
        self.communicate(commands)