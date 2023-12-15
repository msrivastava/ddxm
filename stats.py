class Stats():
    def __init__(self):
        self.distance_traveled = 0
        self.grabbed_objects = 0
        self.grab_attempts = 0
        self.dropped_outside_goal = []
        self.objects_sensed = 0
        self.sensor_activation = 0
        self.objects_in_goal = []
        self.dangerous_objects_in_goal = []
        self.failed = 0
        self.time_with_teammates = {}
        self.end_time = 0
        self.team_dangerous_objects_in_goal = 0
        self.total_dangerous_objects = 0
        self.quality_work = 0
        self.effort = 0
        self.team_end_time = 0
        self.team_failure_reasons = {}
        self.team_quality_work = 0
        self.team_speed_work = 0
        self.team_achievement = 0
        self.team_payment = 0
        self.individual_payment = 0