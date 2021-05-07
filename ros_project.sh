#!/bin/bash

# robot movement
gnome-terminal --window --tab -t "bring up the robot" -e 'bash -c "roslaunch mrobot_bringup fake_mrobot_with_laser.launch;exec bash"' \
			--tab -t 'robot movement move base' -e 'bash -c "sleep 5; roslaunch mrobot_navigation fake_nav_demo.launch;exec bash"'\
			--tab -t 'robot movement step by step' -e 'bash -c "sleep 5; roslaunch mrobot_navigation sbs_movement.launch;exec bash"'\
			--tab -t 'run navigation' -e 'bash -c "sleep 5; rosrun mrobot_navigation random_navigation.py;exec bash"'

# state machine
gnome-terminal --window --tab -t "state machine" -e 'bash -c "rosrun state_machine state.py;exec bash"'

# hand sign recognition & obj detection
gnome-terminal --window --tab -t "launch logi webcam" -e 'bash -c "roslaunch usb_cam usb_cam-test.launch;exec bash"' \
			--tab -t 'launch hand sign recognition' -e 'bash -c "sleep 5; roslaunch sign_recognition sign_recognition.launch;exec bash"'\
			--tab -t 'object detection' -e 'bash -c "sleep 5; roslaunch object_detect ros_obj_detect.launch;exec bash"'




eval $BASH_POST_RC
