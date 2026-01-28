#############################################################################
# Set point controller for 12-DOF spot quadruped
# Set desired desired base acceleration
# Press 1 for joint PD control
# Press 2 for model based control
#############################################################################

import mujoco
import mujoco.viewer    
import time 
from time import sleep
import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import qr

# Load the spot quadruped model
model_path = 'boston_dynamics_spot/scene.xml'
model = mujoco.MjModel.from_xml_path(model_path)
nv = model.nv                    # number of generalized velocities
nu = model.nu                    # number of actuators
nsensordata = model.nsensordata  # number of sensor data

# Body IDs in contact with the ground
FL_LEG_BODY = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"fl_lleg")
FR_LEG_BODY = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"fr_lleg")
HL_LEG_BODY = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"hl_lleg")
HR_LEG_BODY = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"hr_lleg")
listConSensBodies = [FL_LEG_BODY, FR_LEG_BODY, HL_LEG_BODY, HR_LEG_BODY]

# Sensor IDs for contact sensors
FL_CS_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "fl_con_sens")
FR_CS_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "fr_con_sens")
HL_CS_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "hl_con_sens")
HR_CS_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "hr_con_sens")
listConSensIDs = [FL_CS_ID, FR_CS_ID, HL_CS_ID, HR_CS_ID]
nconSens = len(listConSensIDs)
lenSensData = 7 # found | force | pos

# contact sensor data: found | force3x1 | pos3x1  --> total 7 values per contact sensor
cfs_idxs = np.r_[
                1:4,                 # 1,2,3
                1+7 : 1+7+3,         # 8,9,10
                1+7*2 : 1+7*2+3,     # 15,16,17
                1+7*3 : 1+7*3+3      # 22,23,24
                ]

# make data
data = mujoco.MjData(model)
data_ID = mujoco.MjData(model)  # data for controller dynamics computations

# set the standing keyframe
mujoco.mj_resetDataKeyframe(model, data, 0)

ppause = False
choose_controller = 1  # 1 for PD, 2 for Model Based

# Joint PD control gains
KpLeg = 500.
KvLeg = 10. 

# Global variables for desired joint position and velocity control
mujoco.mj_resetDataKeyframe(model, data, 0)
des_qjpos = data.qpos[7:].copy()
des_qjvel = data.qvel[6:].copy()

# Desired location and orientation of the floating base
des_fb_pos = data.qpos[0:3].copy()
des_fb_quat = np.array([1., 0, 0, 0])
mujoco.mju_axisAngle2Quat(des_fb_quat, [0, 1, 0.], np.deg2rad(0.))

def keyboard_func(keycode):
    """
    Keyboard callback to toggle simulation pause.
    Spacebar toggles the pause state.
    """
    global ppause, choose_controller, data
    if keycode == ord(' '):
        ppause = not ppause
    if keycode == ord('1'):
        choose_controller = 1
        print("Switched to PD controller")
    if keycode == ord('2'):
        choose_controller = 2
        print("Switched to Model Based controller")
    if keycode == ord('3'):
        choose_controller = 3
        print("Switched to no controller")

def remove_dependent_constraints(A, b, tol=1e-10):
    """
    Remove linearly dependent rows from constraint matrix A and corresponding RHS b
    
    Returns: A_reduced, b_reduced with only independent constraints
    """
    if A.shape[0] == 0:
        return A, b
    
    # QR decomposition with column pivoting
    Q, R, P = qr(A.T, mode='economic', pivoting=True)
    
    # Find independent columns (which correspond to independent rows in A)
    rank = np.sum(np.abs(np.diag(R)) > tol)
    independent_indices = P[:rank]
    
    return A[independent_indices, :], b[independent_indices]

W_tauB = np.eye(6) * 1.             # Weights for physics violation
W_cfs = 1.0                          # Weight for contact force reguleriser
W_pc_acc = 1.0                      # Weight for contact point acceleration reguleriser

def residual_vec(dv, model:mujoco.MjModel, d_ID:mujoco.MjData, Amat:np.ndarray, d:mujoco.MjData):
    """
    Cost function written as least squares vector residual
    """
    global W_tauB, W_cfs, W_pc_acc, cfs_idxs

    # print(f"fb acc was set to: {d_ID.qacc[0:6]}")

    # set the joint accelerations for current iteration
    d_ID.qacc[6:] = dv.copy()
    
    # compute the inverse dynamics (full mj_inverse has been done once before entering optimisation)
    mujoco.mj_inverseSkip(model, 
                            d_ID, 
                            skipstage=mujoco.mjtStage.mjSTAGE_VEL,
                            skipsensor=False)

    # Physics violation residual: tauB 
    tauB_residual = d_ID.qfrc_inverse[0:6]

    # Change in contact forces from the previous integration step
    idxs = np.r_[1:4]
    cfs = d_ID.sensordata[cfs_idxs]
    cfs_robot = d.sensordata[cfs_idxs]
    cfs_residual = (cfs - cfs_robot)
    
    # Change in contact point acceleration: pacc = (Amat * dv - bvec)
    accel_residual = Amat @ (dv - d.qacc[6:]) # Jdot * qdot is unchanged hence, cancelled out
    
    # Line up all the residuals one below another
    cost_residual_vector = np.concatenate([
        np.sqrt(W_tauB) @ tauB_residual,
        np.sqrt(W_cfs) * cfs_residual,
        np.sqrt(W_pc_acc) * accel_residual
    ])
    
    return cost_residual_vector
    
def jac_residual(dv, model:mujoco.MjModel, d_ID:mujoco.MjData, Amat:np.ndarray, d:mujoco.MjData):
    """ 
    Jacobian of the cost function: nresiduals x ndv 
    """
    global nv, nsensordata, W_tauB, W_pc_acc, W_cfs, cfs_idxs

    # print(f"fb acc was set to: {d_ID.qacc[0:6]}")
    
    # set the joint accelerations for current iteration
    d_ID.qacc[6:] = dv.copy()
    
    # compute the inverse dynamics (full mj_inverse has been done once before entering optimisation)
    mujoco.mj_inverseSkip(model, 
                            d_ID, 
                            skipstage=mujoco.mjtStage.mjSTAGE_VEL,
                            skipsensor=False)
    
    # Compute ID gradients (they are transposed w.r.t. control theory convention)
    DtauDaT = np.zeros((nv, nv))            # the mass matrix!
    DsDaT = np.zeros((nv, nsensordata))     
    integrator_backup = model.opt.integrator
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    mujoco.mjd_inverseFD(model, d_ID, eps=1e-6, flg_actuation=True,
                         DfDq=None, DfDv=None, DfDa=DtauDaT,
                         DsDq=None, DsDv=None, DsDa=DsDaT,
                         DmDq=None)
    model.opt.integrator = integrator_backup
    
    # Get the matrices in control theory convention
    DtauDa = DtauDaT.T # shape (nv, nv)
    DsDa = DsDaT.T     # shape (nsensordata, nv)

    # Jacobian of physics violation residual vector (tauB)
    jac_tauB_residual = DtauDa[0:6, 6:]
    
    # Jacobian of the contact forces residual
    jac_cf_residual = DsDa[cfs_idxs, 6:] 
    
    # Jacobian of the acceleration residual
    grad_accel_cost = Amat

    jac_cost_residual = np.vstack([
        np.sqrt(W_tauB) @ jac_tauB_residual,
        np.sqrt(W_cfs) * jac_cf_residual,
        np.sqrt(W_pc_acc) * grad_accel_cost
    ])

    return jac_cost_residual

def controller(m, d):
    """
    Controller callback function
    """
    global choose_controller, nv, nu, nsensordata, nconSens, lenSensData
    global data_ID, des_qjpos, des_qjvel
    
    d.ctrl[:] = 0.0  # Clear the control inputs

    if choose_controller == 1:
        # Joint level P-D control
        jt = np.zeros(nu)
        for i in range(nu):
            err_pos = des_qjpos[i] - d.qpos[7+i]
            err_vel = des_qjvel[i] - d.qvel[6+i]
            jt[i] = KpLeg * (err_pos) + KvLeg * (err_vel)
        # Saturate the joint torques
        d.ctrl = np.clip(jt, -5000, 5000)
        # print(f"PD Joint torques: {d.ctrl} pos err: {des_qjpos - d.qpos[7:]} vel err: {des_qjvel - d.qvel[6:]}")
        return
    elif choose_controller == 2:
        if d.ncon == 0:
            print("no contacts. Using PD controller instead.")
            choose_controller = 1
            return  
        # Rest of the model based control is written below
    else:
        choose_controller = 1
        return

    # copy the robot's state the data_ID structure used for model based control
    # Get the robot's current qpos, and qvel
    # stateSpec = mujoco.mjtState.mjSTATE_PHYSICS
    stateSpec = mujoco.mjtState.mjSTATE_PHYSICS | mujoco.mjtState.mjSTATE_QFRC_APPLIED | mujoco.mjtState.mjSTATE_XFRC_APPLIED
    nState = mujoco.mj_stateSize(m, stateSpec)
    statevec = np.zeros(nState)
    # print(f"nState: {nState}")

    # Set the current robot qpos and qvel to data_ID
    mujoco.mj_getState(m, d, statevec, stateSpec)
    mujoco.mj_setState(m, data_ID, statevec, stateSpec)

    # Propagate everything in data_ID
    data_ID.qacc[:] = d.qacc.copy()  # start from current accelerations
    mujoco.mj_inverse(m, data_ID)

    des_fb_qacc = np.zeros(6) # For now, just stand steady
    cmd_qacc = np.zeros(nv)
    Kp_fb_qacc = 1.
    Kd_fb_qacc = .1
    Kp_ori_fb_qacc = 1.
    Kd_ori_fb_qacc = .1
    cmd_qacc[0:3] = des_fb_qacc[0:3] + Kp_fb_qacc * (des_fb_pos - d.qpos[0:3]) + Kd_fb_qacc * ( - d.qvel[0:3])
    err_quat = np.zeros(3)
    mujoco.mju_subQuat(err_quat, des_fb_quat, d.qpos[3:7])
    cmd_qacc[3:6] = des_fb_qacc[3:6] + Kp_ori_fb_qacc * (err_quat) + Kd_ori_fb_qacc * ( - d.qvel[3:6])

    diff_qacc_fb = cmd_qacc[0:6] - d.qacc[0:6]
    # print(f"Curr qacc:\n {d.qacc[0:6]} \n Cmd qacc:\n {cmd_qacc[0:6]} \n Diff:\n {diff_qacc_fb}")

    data_ID.qacc[0:6] += 1. * diff_qacc_fb  # for debugging in case of inconvergence of optimiser

    # Rest of the joint accelerations are solved for using optimisation problem

    # Compute the contact point acceleration: (JacP * qacc + JacPDot * qvel) = should be zero for sticking contact
    # But we will use it as a soft constraint
    # data_ID and main data will differ only in terms of accelerations and contact forces---> joint torques
    # Jacobian matrices remain same for main data and data_ID as they depend only on qpos
    JacPstack = np.zeros((3*nconSens, nv))
    JacPDotstack = np.zeros((3*nconSens, nv))
    for i in range(4): #listConSensIDs
        # Check if the contact is found
        found = int(d.sensordata[lenSensData*i])
        if found:
            # Compute the Jacobian for the contact point
            addr = lenSensData*i + 4
            pt_pos = d.sensordata[addr:addr+3]
            conBody = listConSensBodies[i]
            mujoco.mj_jac(m, d, JacPstack[3*i:3*(i+1),:], None, pt_pos, conBody)
            mujoco.mj_jacDot(m, d, JacPDotstack[3*i:3*(i+1),:], None, pt_pos, conBody)

    # Contact point acc constraints look like this: Jj * qj_acc = - Jb *qbacc - (Jdot * qvel)
    Jb_qacc = JacPstack[:, 0:6]
    Jj_qacc = JacPstack[:, 6:]
    RHS = - Jb_qacc @ cmd_qacc[0:6] - JacPDotstack @ d.qvel

    Amat, bvec = remove_dependent_constraints(Jj_qacc, RHS)    
    # Amat, bvec = JacPstack, RHS # Debugging
    
    # The cost function gradient uses mjd_inverseFD that does not support RK4 integrator
    integrator_backup = model.opt.integrator
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    res = least_squares(fun=residual_vec,
                        x0=d.qacc[6:],
                        jac=jac_residual,
                        args=(m, data_ID, Amat, d),
                        method='lm', # or trf
                        ftol=1e-3,
                        xtol=1e-3,
                        gtol=1e-3
                        )
    model.opt.integrator = integrator_backup                    
    if res.success:
        # print("Opt:", res.success, "Status", res.status)
        cmd_qacc[6:] =  res.x
    else:
        print("Opt failed:", res.status)
        # choose_controller = 1

    # Compute the final inverse dynamics with the optimized accelerations
    data_ID.qacc[6:] = cmd_qacc[6:].copy()
    mujoco.mj_inverse(m, data_ID)
    # print(f"tau_b: {data_ID.qfrc_inverse[0:6]} tau_j: {data_ID.qfrc_inverse[6:]} cfs: {data_ID.sensordata[1:4]}")
    
    # Set the control inputs (joint torques)
    ID_factor = 0.7
    d.ctrl[:] = ID_factor * np.clip(data_ID.qfrc_inverse[6:].copy(), -1000, 1000)
    
    factor = 0.3
    d.ctrl[:] += factor * (KpLeg * (des_qjpos - d.qpos[7:]) + KvLeg * (des_qjvel - d.qvel[6:]))
    
   
def main():
    """
    main function
    """
    # numpy print options
    np.set_printoptions(precision=3, suppress=True, linewidth=300)

    global model, data, nv, nu, nsensordata

    # set controller callback
    mujoco.set_mjcb_control(controller)

    RENDER_HZ = 60.0
    render_loop_time = 1.0 / RENDER_HZ
    SLOWDOWN = 1.0 # times slower than real-time

    # Shrink the simulation timestep if needed
    advance_sim_by = ((1.0 / RENDER_HZ) / SLOWDOWN)
    if model.opt.timestep > (advance_sim_by / 5):
        model.opt.timestep = advance_sim_by / 5


    global ppause
    with mujoco.viewer.launch_passive(model, data, key_callback=keyboard_func) as viewer:

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
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
