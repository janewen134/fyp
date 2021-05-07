#!/usr/bin/env python
# -*- coding: utf-8 -*-

#############################################
##ask user to specify a mode to execute 
## 0: visit rooms on a given map (default) 
## 1: step by step control 
#############################################

import rospy
from state_machine.srv import State, StateResponse

def stateCallback(req):
	# return the mode to nodes 

    return StateResponse(state)


def state_server():
	# init this node
    rospy.init_node('state_server')

    # ask user to specify the mode
    global state
    choice = raw_input('Do you want to choose mode 0 visit rooms on a given map (default) \n or mode 1 step by step control ? ')
    if len(choice)!=1:
        state = 0
    elif choice.find('0') != -1:
        state = 0
    else:
        state = 1

	# create a server named /state，register the callback function stateCallback
    s = rospy.Service('/state', State, stateCallback)


    # this node could shut down after mode specification delivered to nodes
    rospy.spin()


if __name__ == "__main__":
    state_server()
