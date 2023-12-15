from enum import Enum

#Possible actions
class Action(Enum):
    move_up = 0
    move_down = 1
    move_left = 2
    move_right = 3
    move_up_right = 4
    move_up_left = 5
    move_down_right = 6
    move_down_left = 7
    grab_up = 8
    grab_right = 9
    grab_down = 10
    grab_left = 11
    grab_up_right = 12
    grab_up_left = 13
    grab_down_right = 14
    grab_down_left = 15
    drop_object = 16
    danger_sensing = 17
    get_occupancy_map = 18
    get_objects_held = 19
    check_item = 20
    check_robot = 21
    get_messages = 22
    send_message = 23
    request_item_info = 24
    request_agent_info = 25
    wait = 26
    communicate = 27
    