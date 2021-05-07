
(cl:in-package :asdf)

(defsystem "mrobot_navigation-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "Sbs" :depends-on ("_package_Sbs"))
    (:file "_package_Sbs" :depends-on ("_package"))
  ))