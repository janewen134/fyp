;; Auto-generated. Do not edit!


(when (boundp 'mrobot_navigation::Sbs)
  (if (not (find-package "MROBOT_NAVIGATION"))
    (make-package "MROBOT_NAVIGATION"))
  (shadow 'Sbs (find-package "MROBOT_NAVIGATION")))
(unless (find-package "MROBOT_NAVIGATION::SBS")
  (make-package "MROBOT_NAVIGATION::SBS"))
(unless (find-package "MROBOT_NAVIGATION::SBSREQUEST")
  (make-package "MROBOT_NAVIGATION::SBSREQUEST"))
(unless (find-package "MROBOT_NAVIGATION::SBSRESPONSE")
  (make-package "MROBOT_NAVIGATION::SBSRESPONSE"))

(in-package "ROS")





(defclass mrobot_navigation::SbsRequest
  :super ros::object
  :slots ())

(defmethod mrobot_navigation::SbsRequest
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

(defclass mrobot_navigation::SbsResponse
  :super ros::object
  :slots (_result ))

(defmethod mrobot_navigation::SbsResponse
  (:init
   (&key
    ((:result __result) 0)
    )
   (send-super :init)
   (setq _result (round __result))
   self)
  (:result
   (&optional __result)
   (if __result (setq _result __result)) _result)
  (:serialization-length
   ()
   (+
    ;; uint8 _result
    1
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; uint8 _result
       (write-byte _result s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; uint8 _result
     (setq _result (sys::peek buf ptr- :char)) (incf ptr- 1)
   ;;
   self)
  )

(defclass mrobot_navigation::Sbs
  :super ros::object
  :slots ())

(setf (get mrobot_navigation::Sbs :md5sum-) "25458147911545c320c4c0a299eff763")
(setf (get mrobot_navigation::Sbs :datatype-) "mrobot_navigation/Sbs")
(setf (get mrobot_navigation::Sbs :request) mrobot_navigation::SbsRequest)
(setf (get mrobot_navigation::Sbs :response) mrobot_navigation::SbsResponse)

(defmethod mrobot_navigation::SbsRequest
  (:response () (instance mrobot_navigation::SbsResponse :init)))

(setf (get mrobot_navigation::SbsRequest :md5sum-) "25458147911545c320c4c0a299eff763")
(setf (get mrobot_navigation::SbsRequest :datatype-) "mrobot_navigation/SbsRequest")
(setf (get mrobot_navigation::SbsRequest :definition-)
      "---
uint8 result
")

(setf (get mrobot_navigation::SbsResponse :md5sum-) "25458147911545c320c4c0a299eff763")
(setf (get mrobot_navigation::SbsResponse :datatype-) "mrobot_navigation/SbsResponse")
(setf (get mrobot_navigation::SbsResponse :definition-)
      "---
uint8 result
")



(provide :mrobot_navigation/Sbs "25458147911545c320c4c0a299eff763")


