import argparse
import yaml
import copy
import gym_collab
import json_numpy
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

from simulation import Simulation
from rm_environment import RewardMachineEnv
from deepq_control import DeepQControl

from gym_collab.envs.action import Action

# holds the high level event flags for the agents, pulled directly from the environment
event_flags = [{
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

# for training 
class RobotState:
    def __init__(self, latest_map, object_held, meta_robots):
        self.latest_map = latest_map
        self.object_held = object_held
        self.items = []
        self.item_estimates = {}
        self.robots = [{"neighbor_type": meta_robots[n][1], "neighbor_location": [-1,-1], "last_seen_location": []} for n in range(len(meta_robots))]
        self.strength = 1
        self.map_metadata = {}
        
    def update_items(self,item_output, item_idx, robot_idx): #Updates items
        if item_idx not in self.item_estimates:
            self.item_estimates[item_idx] = [{"item_danger_level": 0, "item_danger_confidence": 0, "item_location": [-1,-1], "item_time": 0} for n in range(len(self.robots)+1)]
            
        self.item_estimates[item_idx][robot_idx]["item_location"] = [int(item_output["item_location"][0]),int(item_output["item_location"][1])]
        self.item_estimates[item_idx][robot_idx]["item_time"] = item_output["item_time"]
        self.item_estimates[item_idx][robot_idx]["item_danger_level"] = item_output["item_danger_level"]
        self.item_estimates[item_idx][robot_idx]["item_danger_confidence"] = item_output["item_danger_confidence"]
        print(item_output)
        if not self.items[item_idx]["item_danger_level"] or  (item_output["item_danger_level"] and round(self.items[item_idx]["item_danger_confidence"][0],3) == round(item_output["item_danger_confidence"][0],3) and self.items[item_idx]["item_time"][0] < item_output["item_time"][0]) or (item_output["item_danger_level"] and self.items[item_idx]["item_danger_confidence"][0] < item_output["item_danger_confidence"][0]):
            
            self.items[item_idx] = item_output
            self.items[item_idx]["item_location"] = [int(item_output["item_location"][0]),int(item_output["item_location"][1])]
            
        elif self.items[item_idx]["item_time"][0] < item_output["item_time"][0]:
        
            self.items[item_idx]["item_location"] = [int(item_output["item_location"][0]),int(item_output["item_location"][1])]
            self.items[item_idx]["item_time"] = item_output["item_time"]

def print_map(occupancy_map): # Occupancy maps require special printing so that the orientation is correct
    new_occupancy_map = occupancy_map.copy()
    for row_id in range(occupancy_map.shape[0]):
        new_occupancy_map[row_id,:] = occupancy_map[occupancy_map.shape[0]-row_id-1,:]

    new_new_occupancy_map = new_occupancy_map.copy()
    for row_id in range(occupancy_map.shape[1]): 
        new_new_occupancy_map[:,row_id] = new_occupancy_map[:,occupancy_map.shape[1]-row_id-1]
    print(new_new_occupancy_map)

# create wrapper for rewardmachineenv
class AICollabEnvWrapper(gym.Wrapper):
    def __init__(self, env, event_id):
        super().__init__(env)
        self.event_id = event_id

    def step(self, action):
        # executing the action in the environment
        next_obs, original_reward, env_done, info = self.env.step(action)
        self.info = info
        return next_obs, original_reward, env_done, info
    
    def get_events(self):
        events = ''
        if event_flags[self.event_id]["a"]:
            events+='a'
        if event_flags[self.event_id]["b"]:
            events+='b'
        if event_flags[self.event_id]["c"]:
            events+='c'
        if event_flags[self.event_id]["d"]:
            events+='d'
        if event_flags[self.event_id]["e"]:
            events+='e'
        if event_flags[self.event_id]["f"]:
            events+='f'
        if event_flags[self.event_id]["g"]:
            events+='g'
        print("Events:", events, " for agent", str(self.event_id))
        return events

if __name__ == "__main__": 
  # Initialize simulation
  parser = argparse.ArgumentParser()
  parser.add_argument('--local', action='store_true', help='run locally only')
  parser.add_argument('--log-state', action='store_true', help='Log occupancy maps')
  parser.add_argument('--scenario', type=int, default=1, help='Choose scenario')
  parser.add_argument('--log', action='store_true', help="Log occupancy maps + create videos")
  parser.add_argument('--showall', action='store_true', help="Show everything in the top view")
  parser.add_argument('--single-weight', type=int, default=0, help="Make all objects of the specified weight")
  parser.add_argument('--single-danger', action='store_true', help="Make all objects dangerous")
  parser.add_argument('--seed', type=int, default=-1, help="Input seed value")
  parser.add_argument('--log-results', type=str, default='', help='Directory where to log results')
  parser.add_argument('--save-map', action='store_true', help="Save the occupancy map")
  
  args = parser.parse_args()
  
  with open('config.yaml', 'r') as file:
      cfg = yaml.safe_load(file)
  
  print("Initializing simulation...")
  s = Simulation(args, cfg, launch_build=True, port=1071)

  print("Initializing training environments...")
  #gym.pprint_registry()
  env_1 = gym.make('gym_collab/AICollabWorld-v0', use_occupancy=True,
                  view_radius=5, robot_id="A", config=copy.deepcopy(s.initial_config_map))
    
  #env_2 = gym.make('gym_collab/AICollabWorld-v0', use_occupancy=True,
  #                view_radius=5, robot_id="B", config=copy.deepcopy(s.initial_config_map))

  rm_files = ["./t1.txt"]
  #rm_files = ["./t2.txt"]

  wrapped_env_1 = RewardMachineEnv(AICollabEnvWrapper(env_1, 0), rm_files)
  #wrapped_env_2 = RewardMachineEnv(AICollabEnvWrapper(env_2, 1), rm_files)

  print("Environments initialized!")

  for i in range(2):
    print()
    print("Initial occupancy map for agent", str(i + 1))
    s.occupancy_map_request.append(s.ai_magnebots_ids[i])
    m, _ = s.get_occupancy_map(s.ai_magnebots_ids[0])
    print_map(m)

  # Training loop
  current_episode = 0
  num_episodes = 3000
  device = "cpu"
  next_observation = []

  # Generate an action for each agent
  # ai_actions = [a.control(  ) for a in agents]

  # Run simulation
  # 1. Run actions of each agent
  # 2. Update outputs for agents
  # s.run(ai_actions)

  # Update global variables containing event data
  # Using outputs of simulation, update gym enviroment (rewards, etc.)
  # Increment step (this means each action will be one step regardless of real time)

  env = wrapped_env_1
  env.observation_space.shape
  env = FlattenObservation(env)
  env.observation_space.shape
  just_starting = True

  for e in range(num_episodes):
    current_step = 0
    done = False
    #for env in [wrapped_env_1, wrapped_env_2]:
    s.reset_world()
    #env.take_action(26)
    output = s.run(env.robot_id, env.action_message)
    env.map = output[0]
    env.new_output = output
    observation, info = env.reset()
    env.new_output = ()
    robotState = RobotState(observation['frame'].copy(), 0, env.neighbors_info)

    if just_starting:
      deepq_control = DeepQControl(observation,device,1000)
      just_starting = False
      #deepq_control_2 = DeepQControl(observation,device,1000)

    action_issued = [False,False]
    last_action = [0,0]

    messages = []
    
    action = env.action_space.sample()
    action["action"] = Action.get_occupancy_map.value

    action['num_cells_move'] = 1

    high_level_action_finished = True

    action_function = ""
    function_output = []

    step_count = 0
    
    process_reward = 0
    
    disabled = False

    action["action"] = deepq_control.start(observation)

    while current_step < 1000 and not done:
      for j in s.events[0]:
        event_flags[0][j] = s.events[0][j]

      #Make sure to issue concurrent actions but not of the same type. Else, wait.
      if action["action"] < Action.danger_sensing.value and not action_issued[0]:
        action_issued[0] = True
        last_action[0] = action["action"]
      elif action["action"] != Action.wait.value and action["action"] >= Action.danger_sensing.value and not action_issued[1]:
        last_action_arguments = [action["item"],action["robot"],action["message"]]
        action_issued[1] = True
        last_action[1] = action["action"]
      else:
        action["action"] = Action.wait.value

      env.take_action(action)
      output = s.run(env.robot_id, env.action_message)
      env.old_output = output
      next_observation, reward, terminated, truncated, info = env.step(action)

      if reward != 0:
        print('Reward', reward)
        process_reward += reward
          
      if next_observation["num_items"] > len(robotState.items):
        diff_len = next_observation["num_items"] - len(robotState.items)
        robotState.items.extend([{'item_weight': 0, 'item_danger_level': 0, 'item_danger_confidence': np.array([0.]), 'item_location': np.array([-1, -1], dtype=np.int16), 'item_time': np.array([0], dtype=np.int16)} for d in range(diff_len)])
          
      robotState.strength = next_observation["strength"]
          
      #When any action has completed
      if next_observation and any(next_observation['action_status']) and not disabled:
        ego_location = np.where(next_observation['frame'] == 5)
        previous_ego_location = np.where(robotState.latest_map == 5)
        robotState.latest_map[previous_ego_location[0][0],previous_ego_location[1][0]] = 0
          
        if next_observation['action_status'][2]: #If sensing action was succesful
          if Action(last_action[1]) == Action.get_occupancy_map: #Maintain the state of the occupancy map and update it whenever needed
            view_radius = 5
                  
            max_x = ego_location[0][0] + view_radius
            max_y = ego_location[1][0] + view_radius
            min_x = max(ego_location[0][0] - view_radius, 0)
            min_y = max(ego_location[1][0] - view_radius, 0)
            robotState.latest_map[min_x:max_x+1,min_y:max_y+1]= next_observation["frame"][min_x:max_x+1,min_y:max_y+1]
                
            robotState.map_metadata = info['map_metadata']

            for m_key in info['map_metadata'].keys(): #We get the ids of objects/robots present in the current view and update locations
              for map_object in info['map_metadata'][m_key]:
                m_key_xy = m_key.split('_')
                if isinstance(map_object, list): #Object information
                  try:
                    ob_key = info["object_key_to_index"][map_object[0]]
                  except:
                    pdb.set_trace()                                
                    robotState.items[ob_key]["item_location"] = [int(m_key_xy[0]), int(m_key_xy[1])]
                              
                elif map_object not in info['map_metadata'][str(ego_location[0][0])+'_'+str(ego_location[1][0])]: #Robot information
                  ob_key = info["robot_key_to_index"][map_object]
                  robotState.robots[ob_key]["neighbor_location"] = [int(m_key_xy[0]), int(m_key_xy[1])]

          elif Action(last_action[1]) == Action.get_objects_held:
            robotState.object_held = next_observation['objects_held']
      
          elif Action(last_action[1]) == Action.check_item:
            robotState.update_items(next_observation["item_output"],last_action_arguments[0], -1)
              
          elif Action(last_action[1]) == Action.check_robot: #Make sure to update estimates and take the one with the highest confidence
            item_idx = last_action_arguments[1]-1
            robotState.robots[item_idx]["neighbor_location"] = next_observation["neighbors_output"]["neighbor_location"]
            robotState.robots[item_idx]["neighbor_type"] = next_observation["neighbors_output"]["neighbor_type"]
                      
          robotState.latest_map[ego_location[0][0],ego_location[1][0]] = 5 #Set ego robot in map
          
          for ob_key in range(len(robotState.robots)): #If the agent is not where it was last seen, mark it
              robo_location = robotState.latest_map[robotState.robots[ob_key]["neighbor_location"][0],robotState.robots[ob_key]["neighbor_location"][1]]
              if robo_location != 5 and robo_location != 3:
                  robotState.robots[ob_key]["neighbor_location"] = [-1,-1]
                  
          for ob_key in range(len(robotState.items)): #If the agent is not where it was last seen, mark it
              item_location = robotState.latest_map[robotState.items[ob_key]["item_location"][0],robotState.items[ob_key]["item_location"][1]]
              if item_location == 0:
                  robotState.items[ob_key]["item_location"] = [-1,-1]

          if high_level_action_finished: #When a high level action finishes, we sense the environment
            if last_action[1] == Action.get_messages.value: #Action.get_occupancy_map.value:
              
              print_map(robotState.latest_map)
              print("Held:",robotState.object_held)
              
              last_action[1] = 0 #Reset last sensing action
              step_count += 1

              last_high_action = action["action"]
              action["action"] = deepq_control.control(reward, terminated, truncated, robotState, action, step_count, ego_location)

              if action["action"] < 0:
                break
                      
              messages = []
              process_reward = 0
                  
            elif last_action[1] == Action.get_occupancy_map.value:
              action["action"] = Action.get_objects_held.value

            elif last_action[1] == Action.get_objects_held.value:
              action["action"] = Action.get_messages.value

            else:       
              action["action"] = Action.get_occupancy_map.value

          if any(next_observation['action_status'][:2]):
            action_issued[0] = False
          if any(next_observation['action_status'][2:4]):
            action_issued[1] = False
              
      if terminated or truncated:
        done = True

      current_step += 1
    current_episode += 1

  env.close()
  #wrapped_env_2.close()
