#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from geometry_msgs.msg import Twist
import sys, select, termios, tty
from mrobot_navigation.srv import Sbs, SbsRequest
from state_machine.srv import State, StateRequest

msg = """
Control mrobot!
---------------------------
This code is for testing the hand gesture part 
"""

moveBindings = {
        0:(1,0),
        # 1:(1,-1),
        1:(0,1),
        2:(0,-1),
        # 4:(1,1),
        3:(-1,0),
        # 6:(-1,1),
        # 7:(-1,-1),
           }

speedBindings={
        4:(1.1,1),
        6:(.9,1),
        # 'w':(1.1,1),
        # 'x':(.9,1),
        # 'e':(1,1.1),
        # 'c':(1,.9),
          }

def getSign():
    request = SbsRequest()
    response = sign_client(request)
    return response.result

def getState():
    request = StateRequest()   
    response = state_client(request)
    return response.state

speed = .2
turn = 1

def vels(speed,turn):
    return "currently:\tspeed %s\tturn %s " % (speed,turn)

def myhook():
    print "step by step mode is not selected"
    print "shut down this node"

if __name__=="__main__":
    # init node
    rospy.init_node('sign_controlled_movements')
    # get the selected state by user 
    rospy.wait_for_service('/state')
    global state_client
    state_client = rospy.ServiceProxy('/state', State)
    mode = getState()
    # shut itself down if this state is not selected
    if mode == 0:
        print "should shut down"
        rospy.on_shutdown(myhook)
        print "shut down process finished"
    else:
        # create a client to connect to the service /get_sign
        rospy.wait_for_service('/get_sign')
        global sign_client
        sign_client = rospy.ServiceProxy('/get_sign', Sbs)     
        # publish twist   
        pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        
        x = 0
        th = 0
        status = 0
        count = 0
        acc = 0.1
        target_speed = 0
        target_turn = 0
        control_speed = 0
        control_turn = 0
        try:
            print msg
            print vels(speed,turn)
            while(1):
                sign = getSign()
                # use sign to change movement directions（1：positive direction，-1: negative）
                if sign in moveBindings.keys():
                    x = moveBindings[sign][0]
                    th = moveBindings[sign][1]
                    count = 0
                #  use sign to change speed
                elif sign in speedBindings.keys():
                    speed = speed * speedBindings[sign][0]  # linear speed
                    turn = turn * speedBindings[sign][1]    # angular speed
                    count = 0

                    print vels(sign,turn)
                    if (status == 14):
                        print msg
                    status = (status + 1) % 15
                # stop
                elif sign == 5:
                    x = 0
                    th = 0
                    control_speed = 0
                    control_turn = 0
                

                # calculate speed and direction
                target_speed = speed * x
                target_turn = turn * th

                # set limits to speed
                if target_speed > control_speed:
                    control_speed = min( target_speed, control_speed + 0.02 )
                elif target_speed < control_speed:
                    control_speed = max( target_speed, control_speed - 0.02 )
                else:
                    control_speed = target_speed

                if target_turn > control_turn:
                    control_turn = min( target_turn, control_turn + 0.1 )
                elif target_turn < control_turn:
                    control_turn = max( target_turn, control_turn - 0.1 )
                else:
                    control_turn = target_turn

                # create and publish twist 
                twist = Twist()
                twist.linear.x = control_speed; 
                twist.linear.y = 0; 
                twist.linear.z = 0
                twist.angular.x = 0; 
                twist.angular.y = 0; 
                twist.angular.z = control_turn
                pub.publish(twist)

        except:
            print e

        finally:
            twist = Twist()
            twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
            twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
            pub.publish(twist)

