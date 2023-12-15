import math

class RewardFunction:
    def __init__(self):
        pass

    # To implement...
    def get_reward(self, s_info):
        raise NotImplementedError("To be implemented")

    def get_type(self):
        raise NotImplementedError("To be implemented")

class RewardHoldingObject(RewardFunction):
    def __init__(self):
        super().__init__()
    
    def get_type(self):
        return "constant"
    
    def get_reward(self, s_info):
        return 5

class RewardNearAgent(RewardFunction):
    def __init__(self):
        super().__init__()
    
    def get_type(self):
        return 2

class RewardNearObject(RewardFunction):
    def __init__(self):
        super().__init__()
    
    def get_type(self):
        return 2

class RewardDone(RewardFunction):
    def __init__(self):
        super().__init__()
    
    def get_type(self):
        return 50 # big reward for completing task

class RewardDefault(RewardFunction):
    def __init__(self):
        super().__init__()
    
    def get_type(self):
        return -0.01
