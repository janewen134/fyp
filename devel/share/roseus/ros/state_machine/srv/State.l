;; Auto-generated. Do not edit!


(when (boundp 'state_machine::State)
  (if (not (find-package "STATE_MACHINE"))
    (make-package "STATE_MACHINE"))
  (shadow 'State (find-package "STATE_MACHINE")))
(unless (find-package "STATE_MACHINE::STATE")
  (make-package "STATE_MACHINE::STATE"))
(unless (find-package "STATE_MACHINE::STATEREQUEST")
  (make-package "STATE_MACHINE::STATEREQUEST"))
(unless (find-package "STATE_MACHINE::STATERESPONSE")
  (make-package "STATE_MACHINE::STATERESPONSE"))

(in-package "ROS")





(defclass state_machine::StateRequest
  :super ros::object
  :slots ())

(defmethod state_machine::StateRequest
  (:init
   (&key
    )
   (send-super :init)
   self)
  (:serialization-length
   ()
   (+
    0
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;;
   self)
  )

(defclass state_machine::StateResponse
  :super ros::object
  :slots (_state ))

(defmethod state_machine::StateResponse
  (:init
   (&key
    ((:state __state) 0)
    )
   (send-super :init)
   (setq _state (round __state))
   self)
  (:state
   (&optional __state)
   (if __state (setq _state __state)) _state)
  (:serialization-length
   ()
   (+
    ;; uint8 _state
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8 _state
       (write-byte _state s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8 _state
     (setq _state (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;;
   self)
  )

(defclass state_machine::State
  :super ros::object
  :slots ())

(setf (get state_machine::State :md5sum-) "800f34bc468def1d86e2d42bea5648c0")
(setf (get state_machine::State :datatype-) "state_machine/State")
(setf (get state_machine::State :request) state_machine::StateRequest)
(setf (get state_machine::State :response) state_machine::StateResponse)

(defmethod state_machine::StateRequest
  (:response () (instance state_machine::StateResponse :init)))

(setf (get state_machine::StateRequest :md5sum-) "800f34bc468def1d86e2d42bea5648c0")
(setf (get state_machine::StateRequest :datatype-) "state_machine/StateRequest")
(setf (get state_machine::StateRequest :definition-)
      "---
uint8 state
")

(setf (get state_machine::StateResponse :md5sum-) "800f34bc468def1d86e2d42bea5648c0")
(setf (get state_machine::StateResponse :datatype-) "state_machine/StateResponse")
(setf (get state_machine::StateResponse :definition-)
      "---
uint8 state
")



(provide :state_machine/State "800f34bc468def1d86e2d42bea5648c0")


