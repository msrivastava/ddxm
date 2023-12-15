import numpy as np
import random
from magnebot import Magnebot, ActionStatus, ImageFrequency

from stats import Stats

class Enhanced_Magnebot(Magnebot):
    def __init__(self,robot_id, position, controlled_by, key_set=None,image_frequency=ImageFrequency.never,pass_masks=['_img'],strength=1, check_version=False):
        super().__init__(robot_id=robot_id, position=position,image_frequency=image_frequency,check_version=check_version)
        self.key_set = key_set
        self.strength = strength
        self.danger_estimates = []
        self.company = {}
        self.controlled_by = controlled_by
        self.focus_object = ""
        self.item_info = {}
        self.estimate_confidence = 0.9
        self.screen_positions = {"position_ids":[],"positions":[],"duration":[]}
        self.refresh_sensor = 0
        self.messages = []
        self.grasping = False
        self.grasping_time = 0
        self.past_status = ActionStatus.ongoing
        self.view_radius = 0
        self.centered_view = 0
        self.resetting_arm = False
        self.key_pressed = ''
        self.disabled = False
        self.last_output = False
        self.last_position = np.array([])
        self.stats = Stats()
        self.skip_frames = 0
        self.p11 = float(random.uniform(0.5, 0.9)) # Sensor readings are accurate with some probability
        self.p22 = float(random.uniform(0.5, 0.9))
        self.current_teammates = {}
        
    def reset(self, position):
        super().reset(position=position)
        self.resetting_arm = False
        self.past_status = ActionStatus.ongoing
        self.messages = []
        self.grasping = False
        self.grasping_time = 0
        self.screen_positions = {"position_ids":[],"positions":[],"duration":[]}
        self.focus_object = ""
        self.item_info = {}
        self.company = {}
        self.disabled = False
        self.last_output = False
        self.stats = Stats()
        self.last_position = np.array([])
        self.p11 = float(random.uniform(0.5, 0.9)) # Binary channel
        self.p22 = float(random.uniform(0.5, 0.9))
        self.current_teammates = {}
