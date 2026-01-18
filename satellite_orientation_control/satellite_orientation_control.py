import mujoco
import mujoco.viewer    
import time 
from time import sleep
import numpy as np
from scipy.optimize import minimize

# Load the kondo model
model_path = 'my_satellite.xml'
model = mujoco.MjModel.from_xml_path(model_path)
nact = model.nu


# make data
data = mujoco.MjData(model)

# data for inverse dynamics
data_ID = mujoco.MjData(model)


# Desired orientation of the floating base
desired_quat = np.array([1., 0, 0, 0])
# Set the axis and rotation angle
mujoco.mju_axisAngle2Quat(desired_quat, [0, 1, 1], np.deg2rad(90))
print("Desired quaternion:", desired_quat)

# Global pause flag for simulation
ppause = False

def keyboard_func(keycode):
    """
    Keyboard callback to toggle simulation pause.
    Spacebar toggles the pause state.
    """
    global ppause
    if keycode == ord(' '):
        ppause = not ppause

def controller(m:mujoco.MjModel, d:mujoco.MjData):
    """
    Controller for the robot.

    @param m: MuJoCo model
    @param d: MuJoCo data
    """
    global data_ID
    
    nState = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_PHYSICS)
    statevec = np.zeros(nState)
    
    mujoco.mj_getState(m, d, statevec, mujoco.mjtState.mjSTATE_PHYSICS)
    mujoco.mj_setState(m, data_ID, statevec, mujoco.mjtState.mjSTATE_PHYSICS)
    
    mujoco.mj_inverse(m, data_ID)
    
    # Control law to correct the orientation of the floating base
    global desired_quat
    current_quat = d.qpos[3:7]

    ########################################################################################
    #   Method 1: Iterative inverse dynamics optimisation to compute joint torques
    ########################################################################################

    # Compute the error vector
    err_quat_diff = np.zeros(3)
    mujoco.mju_subQuat(err_quat_diff, desired_quat, current_quat)
    print(f"err_quat_diff: {err_quat_diff}")
    # Control law
    desired_qb_acc = np.zeros(6)
    Kp=0.2
    Kv=0.5
    desired_qb_acc[3:6] = Kp * err_quat_diff - Kv * d.qvel[3:6]
    # print(f"desired_qb_acc: {desired_qb_acc[3:6]}")

    # Solve the optimization problem to get joint torques to achieve
    # the desired floating base acceleration
    res = minimize(cost_func, 
                    np.zeros(nact), 
                    args=(m, data_ID),
                    jac=grad_cost_func, 
                    method='L-BFGS-B', 
                    options={'maxiter': 10})
    desired_dqddot = res.x
    print("Optimization success:", res.success, " Cost:", res.fun, "Iterations:", res.nit)

    # joint torques vector
    data_ID.qacc[0:6] = desired_qb_acc
    data_ID.qacc[6:] = desired_dqddot
    mujoco.mj_inverse(m, data_ID)
    print(f"Physics violation: {data_ID.qfrc_inverse[0:6]}")

    jt = data_ID.qfrc_inverse[6:].copy()
    d.ctrl = jt

    ########################################################################################
    # Method 2: Direct computation of joint torques using equation of motion
    ########################################################################################
    # # Compute the error vector
    # err_quat_diff = np.zeros(3)
    # mujoco.mju_subQuat(err_quat_diff, desired_quat, current_quat)
    # print(f"err_quat_diff: {err_quat_diff}")

    # # Control law
    # desired_qb_acc = np.zeros(6)
    # Kp=0.2
    # Kv=0.5
    # desired_qb_acc[3:6] = Kp * err_quat_diff - Kv * d.qvel[3:6]
    # # print(f"desired_qb_acc: {desired_qb_acc[3:6]}")

    # data_ID.qacc[0:6] = desired_qb_acc

    # M = np.zeros((model.nv, model.nv))
    # mujoco.mj_fullM(model, M, data_ID.qM)
    # Mbb = M[0:6, 0:6]
    # Mbj = M[0:6, 6:]
    # Mjj = M[6:, 6:]
    # Cb = data_ID.qfrc_bias[0:6]
    # Cj = data_ID.qfrc_bias[6:]
    # # find the required joint torques to achieve the desired floating base acceleration
    # # Solve A qddot_j = b
    # A = Mbj
    # b = - Cb - Mbb @ data_ID.qacc[0:6]
    # qj_ddot_sol = np.linalg.pinv(A) @ b + (np.eye(model.nv - 6) - np.linalg.pinv(A) @ A) @ np.ones(model.nv - 6) * 0.
    # sol_tauJ_1 = Mbj.T @ data_ID.qacc[0:6] + Mjj @ qj_ddot_sol + Cj
    
    # d.ctrl = sol_tauJ_1.copy()
    ########################################################################################

def cost_func(dv, model: mujoco.MjModel, data: mujoco.MjData):
    """
    dv: design variable: joint accelerations in this case
    """
    # Compute the FRB_wrench norm sqr using the inverse dynamics
    # Minimal implementation: apply joint accelerations, run inverse dynamics, and
    # return the L2 norm of the floating-base wrench.
    data.qacc[6:] = dv
    mujoco.mj_inverse(model, data)
    tau_base = data.qfrc_inverse[0:6]
    return np.dot(tau_base, tau_base)

def grad_cost_func(dv, model: mujoco.MjModel, data: mujoco.MjData):
    """
    Gradient of the cost function
    dv: design variable: joint accelerations in this case
    """
    # Compute the gradient of the cost function using the inverse dynamics derivatives
    data.qacc[6:] = dv
    mujoco.mj_inverse(model, data)
    tau_base = data.qfrc_inverse[0:6]
    
    # Get the inverse dynamics derivatives (transposed w.r.t. control theory notation)
    DfDaT = np.zeros((model.nv, model.nv))
    
    # backup_intergrator = model.opt.integrator
    # model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    
    mujoco.mjd_inverseFD(model, data, eps=1e-6, flg_actuation=True, 
                            DfDq=None, DfDv=None, DfDa=DfDaT, DsDq=None, DsDv=None, DsDa=None, DmDq=None)
    
    # Get the derivaties in the control theory notation by transposing
    DfDa = DfDaT.T
    
    grad = 2 * DfDa[0:6, 6:].T @ tau_base
    # model.opt.integrator = backup_intergrator

    return grad

# define main and setup the viewer
def main():

    # set numpy print precision to 3 digits
    np.set_printoptions(precision=3, suppress=True)

    # set the control callback
    mujoco.set_mjcb_control(controller)

    RENDER_HZ = 60.0
    render_loop_time = 1.0 / RENDER_HZ
    SLOWDOWN = 1. # times slower than real-time

    # Shrink the simulation timestep if needed
    advance_sim_by = ((1.0 / RENDER_HZ) / SLOWDOWN)
    if model.opt.timestep > (advance_sim_by / 5):
        model.opt.timestep = advance_sim_by / 5

    # Set the joint angles accordingly
    global q_des_list, dq_des_list
    q_des_list = data.qpos[7:].copy()
    dq_des_list = data.qvel[6:].copy()

    global ppause
    idx_geom = 0
    with mujoco.viewer.launch_passive(model, data, key_callback=keyboard_func) as viewer:

        with viewer.lock():
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

        # Simulation loop
        while viewer.is_running():
            simstart = data.time
            wallclock = time.time()
            if not ppause:
                while (data.time-simstart) < advance_sim_by:                
                    # Control callback is set already, just step the simulation
                    mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)

            # Time to render
            viewer.sync()
            viewer.user_scn.ngeom = 0

            # Sleep
            computation_time = time.time() - wallclock
            # print(computation_time)
            # print(d.time)
            if computation_time < render_loop_time:
                sleep((render_loop_time - computation_time))
        
        viewer.close()


if __name__ == "__main__":
    main()
