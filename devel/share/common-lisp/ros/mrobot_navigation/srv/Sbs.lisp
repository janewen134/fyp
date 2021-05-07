; Auto-generated. Do not edit!


(cl:in-package mrobot_navigation-srv)


;//! \htmlinclude Sbs-request.msg.html

(cl:defclass <Sbs-request> (roslisp-msg-protocol:ros-message)
  ()
)

(cl:defclass Sbs-request (<Sbs-request>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Sbs-request>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Sbs-request)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name mrobot_navigation-srv:<Sbs-request> is deprecated: use mrobot_navigation-srv:Sbs-request instead.")))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Sbs-request>) ostream)
  "Serializes a message object of type '<Sbs-request>"
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Sbs-request>) istream)
  "Deserializes a message object of type '<Sbs-request>"
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Sbs-request>)))
  "Returns string type for a service object of type '<Sbs-request>"
  "mrobot_navigation/SbsRequest")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Sbs-request)))
  "Returns string type for a service object of type 'Sbs-request"
  "mrobot_navigation/SbsRequest")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Sbs-request>)))
  "Returns md5sum for a message object of type '<Sbs-request>"
  "25458147911545c320c4c0a299eff763")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Sbs-request)))
  "Returns md5sum for a message object of type 'Sbs-request"
  "25458147911545c320c4c0a299eff763")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Sbs-request>)))
  "Returns full string definition for message of type '<Sbs-request>"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Sbs-request)))
  "Returns full string definition for message of type 'Sbs-request"
  (cl:format cl:nil "~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Sbs-request>))
  (cl:+ 0
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Sbs-request>))
  "Converts a ROS message object to a list"
  (cl:list 'Sbs-request
))
;//! \htmlinclude Sbs-response.msg.html

(cl:defclass <Sbs-response> (roslisp-msg-protocol:ros-message)
  ((result
    :reader result
    :initarg :result
    :type cl:fixnum
    :initform 0))
)

(cl:defclass Sbs-response (<Sbs-response>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Sbs-response>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Sbs-response)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name mrobot_navigation-srv:<Sbs-response> is deprecated: use mrobot_navigation-srv:Sbs-response instead.")))

(cl:ensure-generic-function 'result-val :lambda-list '(m))
(cl:defmethod result-val ((m <Sbs-response>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader mrobot_navigation-srv:result-val is deprecated.  Use mrobot_navigation-srv:result instead.")
  (result m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Sbs-response>) ostream)
  "Serializes a message object of type '<Sbs-response>"
  (cl:write-byte (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'result)) ostream)
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Sbs-response>) istream)
  "Deserializes a message object of type '<Sbs-response>"
    (cl:setf (cl:ldb (cl:byte 8 0) (cl:slot-value msg 'result)) (cl:read-byte istream))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Sbs-response>)))
  "Returns string type for a service object of type '<Sbs-response>"
  "mrobot_navigation/SbsResponse")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Sbs-response)))
  "Returns string type for a service object of type 'Sbs-response"
  "mrobot_navigation/SbsResponse")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Sbs-response>)))
  "Returns md5sum for a message object of type '<Sbs-response>"
  "25458147911545c320c4c0a299eff763")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Sbs-response)))
  "Returns md5sum for a message object of type 'Sbs-response"
  "25458147911545c320c4c0a299eff763")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Sbs-response>)))
  "Returns full string definition for message of type '<Sbs-response>"
  (cl:format cl:nil "uint8 result~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Sbs-response)))
  "Returns full string definition for message of type 'Sbs-response"
  (cl:format cl:nil "uint8 result~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Sbs-response>))
  (cl:+ 0
     1
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Sbs-response>))
  "Converts a ROS message object to a list"
  (cl:list 'Sbs-response
    (cl:cons ':result (result msg))
))
(cl:defmethod roslisp-msg-protocol:service-request-type ((msg (cl:eql 'Sbs)))
  'Sbs-request)
(cl:defmethod roslisp-msg-protocol:service-response-type ((msg (cl:eql 'Sbs)))
  'Sbs-response)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Sbs)))
  "Returns string type for a service object of type '<Sbs>"
  "mrobot_navigation/Sbs")