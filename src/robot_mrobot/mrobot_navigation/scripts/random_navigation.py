#!/usr/bin/env python 
# -*- coding: utf-8 -*-
 
import roslib;
import rospy  
import actionlib  
from actionlib_msgs.msg import *  
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, Point, Quaternion, Twist  
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal  
from random import sample  
from math import pow, sqrt  
from mrobot_navigation.srv import Sbs, SbsRequest
from state_machine.srv import State, StateRequest

class NavTest():  
    def __init__(self):  
        rospy.on_shutdown(self.shutdown)  
 
        # create a client to connect to the service /get_sign
        rospy.wait_for_service('/get_sign')
        global sign_client
        sign_client = rospy.ServiceProxy('/get_sign', Sbs) 

        # set the interval between each task
        self.rest_time = rospy.get_param("~rest_time", 2)  
 
 
        # set rooms' positions 
        location0 = Pose(Point(1.150, 5.461, 0.000), Quaternion(0.000, 0.000, -0.013, 1.000))  
        location1 = Pose(Point(6.388, 2.66, 0.000), Quaternion(0.000, 0.000, 0.063, 0.998))  
        location2 = Pose(Point(8.089, -1.657, 0.000), Quaternion(0.000, 0.000, 0.946, -0.324))  
        location3 = Pose(Point(9.767, 5.171, 0.000), Quaternion(0.000, 0.000, 0.139, 0.990))  
        location4 = Pose(Point(0.502, 1.270, 0.000), Quaternion(0.000, 0.000, 0.919, -0.392)) 
        location5 = Pose(Point(4.557, 1.234, 0.000), Quaternion(0.000, 0.000, 0.627, 0.779)) 

        locations = {
            0:location0,
            1:location1,
            2:location2,
            3:location3,
            4:location4,
            6:location5,
        }
        # publish twist
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=5)  
 
        # subscribe move_base server 
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)  

        rospy.loginfo("Waiting for move_base action server...")  

        # wait for the move_base server for 60s
        self.move_base.wait_for_server(rospy.Duration(60))  
        rospy.loginfo("Connected to move base server")  

        # keep the initial pose of the robot
        initial_pose = PoseWithCovarianceStamped()  

        # save some parameters: rate of success, total distance travelled 
        n_locations = len(locations)  
        n_goals = 0  
        n_successes = 0  
        i = n_locations  
        distance_traveled = 0  
        start_time = rospy.Time.now()  
        running_time = 0  
        location = ""  
        last_location = ""  

        # should give an initial position  
        while initial_pose.header.stamp == "":  
            rospy.sleep(1)  

        rospy.loginfo("Starting navigation test")  

        wait_times = 0

        # navigation loop 
        while not rospy.is_shutdown():   
            # get the hand sign
            print "please make a sign"
            rospy.sleep(2) # give user some time to react
            sign = getSign()
            rospy.sleep(2)
            
            # if no sign ("nothing") is given then check three times before quit the program
            if sign==5 & wait_times < 4:
                print "sign received"
                wait_times = wait_times + 1
                rospy.sleep(10)
                continue
            elif sign==5 & wait_times = 4:
                print "no sign received"
                print "I am going to shut down. Byebye"
                break
            print "I am on my way"
            # get the next goal position
            next_goal = locations[sign]   

            # set move base goal 
            self.goal = MoveBaseGoal()  
            self.goal.target_pose.pose = locations[sign]  
            self.goal.target_pose.header.frame_id = 'map'  
            self.goal.target_pose.header.stamp = rospy.Time.now()  

            # print the goal info 
            rospy.loginfo("Going to: " + str(next_goal))  
 
            # send goal to move base server  
            self.move_base.send_goal(self.goal)  

            # set a time limit to complete the task
            finished_within_time = self.move_base.wait_for_result(rospy.Duration(300))   

            # check the feedback  
            if not finished_within_time:  
                self.move_base.cancel_goal()  
                rospy.loginfo("Timed out achieving goal")  
            else:  
                state = self.move_base.get_state()  
                if state == GoalStatus.SUCCEEDED:  
                    rospy.loginfo("Goal succeeded!")  
                else:  
                  rospy.loginfo("Goal failed")     

            rospy.sleep(self.rest_time)  

    def update_initial_pose(self, initial_pose):  
        self.initial_pose = initial_pose  

    def shutdown(self):  
        rospy.loginfo("Stopping the robot...")  
        self.move_base.cancel_goal()  
        rospy.sleep(2)  
        self.cmd_vel_pub.publish(Twist())  
        rospy.sleep(1)  

def trunc(f, n):   
    slen = len('%.*f' % (n, f))  

    return float(str(f)[:slen])  

def getSign():
    request = SbsRequest()
    response = sign_client(request)
    return response.result    

def myhook():
    print "step by step mode is not selected"
    print "shut down this node"

def getState():
    request = StateRequest()   
    response = state_client(request)
    return response.state    

if __name__ == '__main__':  
    rospy.init_node('random_navigation', anonymous=True)
    # get the selected state by user 
    rospy.wait_for_service('/state')
    global state_client
    state_client = rospy.ServiceProxy('/state', State)
    mode = getState()
    # shut itself down if this mode is not selected
    if mode == 1:
        print "should shut down"
        rospy.on_shutdown(myhook)
        print "shut down process finished"  
        return  
    else:    
        try:
            NavTest()  
            rospy.spin()  

        except rospy.ROSInterruptException:  
            rospy.loginfo("Random navigation finished.")
