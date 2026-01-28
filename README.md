This repository contains examples for inverse dynamics (model-based)  control. MuJoCo's contact-implicit inverse dynamics is used to obtain the joint torques. The examples presented are simplistic yet demonstrate the idea of utilising contact model for inverse dynamics control of legged robots. The desired acceleration of the floating base is set. Then, a non-linear least-squares minimization problem is solved to obtain the joint accelerations. The residuals vector mainly constitutes the data.qfrc_inverse[0:6] the base wrench. The obtained joint accelerations are considered valid if this residual is zero. A few more reguleriser terms are added to the residuals vector to maintain closeness of contact forces to the previous state and no-slip (soft) constraint on contact points.

[Video](https://youtu.be/VKUG-WIPbF4) for satellite set point orientation control.

[Video](https://youtu.be/KBQ-ML4luko) for pogo stick trajectory control.

[Video](https://youtu.be/96bVIeJwN5s) for spot quadruped set point floating base position + orientation control.
