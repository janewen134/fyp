; Auto-generated. Do not edit!


(cl:in-package state_machine-srv)


;//! \htmlinclude State-request.msg.html

(cl:defclass <State-request> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass State-request (<State-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <State-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'State-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name state_machine-srv:<State-request> is deprecated: use state_machine-srv:State-request instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <State-request>) ostream)
  "Serializes a message object of type '<State-request>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <State-request>) istream)
  "Deserializes a message object of type '<State-request>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<State-request>)))
  "Returns string type for a service object of type '<State-request>"
  "state_machine/StateRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'State-request)))
  "Returns string type for a service object of type 'State-request"
  "state_machine/StateRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<State-request>)))
  "Returns md5sum for a message object of type '<State-request>"
  "800f34bc468def1d86e2d42bea5648c0")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'State-request)))
  "Returns md5sum for a message object of type 'State-request"
  "800f34bc468def1d86e2d42bea5648c0")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<State-request>)))
  "Returns full string definition for message of type '<State-request>"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'State-request)))
  "Returns full string definition for message of type 'State-request"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <State-request>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <State-request>))
  "Converts a ROS message object to a list"
  (cl:list 'State-request
))
;//! \htmlinclude State-response.msg.html

(cl:defclass <State-response> (roslisp-msg-protocol:ros-message)
  ((state
    :reader state
    :initarg :state
    :type cl:fixnum
    :initform 0))
)

(cl:defclass State-response (<State-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <State-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'State-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name state_machine-srv:<State-response> is deprecated: use state_machine-srv:State-response instead.")))

(cl:ensure-generic-function 'state-val :lambda-list '(m))
(cl:defmethod state-val ((m <State-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader state_machine-srv:state-val is deprecated.  Use state_machine-srv:state instead.")
  (state m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <State-response>) ostream)
  "Serializes a message object of type '<State-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'state)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <State-response>) istream)
  "Deserializes a message object of type '<State-response>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'state)) (cl:read-byte istream))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<State-response>)))
  "Returns string type for a service object of type '<State-response>"
  "state_machine/StateResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'State-response)))
  "Returns string type for a service object of type 'State-response"
  "state_machine/StateResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<State-response>)))
  "Returns md5sum for a message object of type '<State-response>"
  "800f34bc468def1d86e2d42bea5648c0")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'State-response)))
  "Returns md5sum for a message object of type 'State-response"
  "800f34bc468def1d86e2d42bea5648c0")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<State-response>)))
  "Returns full string definition for message of type '<State-response>"
  (cl:format cl:nil "uint8 state~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'State-response)))
  "Returns full string definition for message of type 'State-response"
  (cl:format cl:nil "uint8 state~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <State-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <State-response>))
  "Converts a ROS message object to a list"
  (cl:list 'State-response
    (cl:cons ':state (state msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'State)))
  'State-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'State)))
  'State-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'State)))
  "Returns string type for a service object of type '<State>"
  "state_machine/State")