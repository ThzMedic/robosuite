import pinocchio as pin
import numpy as np
np.set_printoptions(precision=4, suppress=True, threshold=1e-4)
from numpy.linalg import norm, solve

def compute_fk(urdf_path, q, ee_frame):
    """
    Compute forward kinematics using Pinocchio.
    
    Args:
        urdf_path (str): Path to the robot's URDF.
        q (np.ndarray): Joint configuration.
        ee_frame (str): Name of the end-effector frame.
        
    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix of the end-effector.
    """
    
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    
    # Compute forward kinematics
    q_pin = standard_to_pinocchio(model, q)
    pin.forwardKinematics(model, data, q_pin)
    
    for i in range(0, len(model.frames)):
        pin.updateFramePlacement(model, data, i)
        oMact = data.oMf[i]
        frame_name = model.frames[i].name
        print(f"i: {i}, Frame Name: {frame_name}\noMact: \n{clean_and_print_matrix(oMact)}")
    # Get the end-effector frame ID
    tool_frame_id = model.getFrameId(ee_frame)
    
    # Get the end-effector pose
    T = get_end_effector_pose(model, data, tool_frame_id, q_pin)
    
    return fk

def compute_ik(urdf_path, ee_frame, target_pose, q0, max_iter=100, tol=1e-4):
    """
    Compute inverse kinematics using Pinocchio.
    
    Args:
        urdf_path (str): Path to the robot's URDF.
        ee_frame (str): Name of the end-effector frame.
        target_pose (np.ndarray): Desired 4x4 homogeneous transformation.
        target_pose (np.ndarray): Desired 3x1 position.
        q0 (np.ndarray): Initial configuration.
        max_iter (int): Maximum iterations.
        tol (float): Tolerance.
        
    Returns:
        np.ndarray: Joint configuration achieving target_pose.
    """

    oMdes = pin.SE3(target_pose[:3,:3], target_pose[:3,3])
    # print("oMdes: \n", clean_and_print_matrix(oMdes))

    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()# model = pin.buildModelFromUrdf(urdf_path)

    # Use the default Pinocchio solver (e.g., Levenberg-Marquardt) as a placeholder
    tool_frame_id = model.getFrameId(ee_frame)
    q = q0.copy()

    q_pin = standard_to_pinocchio(model, q)
    q_pin_orginal = q_pin.copy()
    # print("q_pin:", q_pin)
    
    pin.forwardKinematics(model,data,q_pin)

    # iterate through the transformation matricies of all frame ids
    # for i in range(0, len(model.frames)):
    #     pin.updateFramePlacement(model, data, i)
    #     oMact = data.oMf[i]
    #     frame_name = model.frames[i].name
    #     print(f"i: {i}, Frame Name: {frame_name}\noMact: \n{clean_and_print_matrix(oMact)}")
    #     T = get_end_effector_pose(model, data, i, q_pin)
    #     print("T: \n", clean_and_print_matrix(T))

    # q_pin      = pin.neutral(model)
    eps    = 1e-4
    IT_MAX = 1000
    DT     = 1e-1
    damp   = 1e-12

    J = pin.computeFrameJacobian(model, data, q_pin, tool_frame_id)


    i = 0
    while True:
        pin.forwardKinematics(model,data,q_pin)
        # pin.updateFramePlacement(model, data, tool_frame_id)
        # oMdes = data.oMi[7] # check that oMdes.actInv(data.oMi[joint_id]) works
        T = get_end_effector_pose(model, data, tool_frame_id, q_pin)
        # print("T: \n", clean_and_print_matrix(T))
        dMi = oMdes.actInv(T)        

        err = pin.log(dMi).vector

        if norm(err) < eps:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break
        J = pin.computeFrameJacobian(model, data, q_pin, tool_frame_id)
        v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
        q_pin = pin.integrate(model,q_pin,v*DT)
        if not i % 10:
            # print('%d: error = %s' % (i, err.T))
            pass
        i += 1
    
    
    if success:
        print("Convergence achieved! Iterations:", i)
    else:
        print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")
        # q_pin = q_pin_orginal
    
    # print("q0:", q0)
    # print("q_pin:", q_pin)
    q = pinocchio_to_standard(model, q_pin)
    print("q final:", q)
    print("J\n:", J)
    return q, J

## FUNCTIONS

def clean_and_print_matrix(matrix, threshold=1e-4):
    """Clean small values from matrix and return string representation"""
    if isinstance(matrix, pin.SE3):
        # Convert SE3 to 4x4 numpy array 
        matrix_array = np.eye(4)
        matrix_array[:3,:3] = matrix.rotation
        matrix_array[:3,3] = matrix.translation
    else:
        matrix_array = np.array(matrix)
        
    matrix_clean = matrix_array.copy()
    matrix_clean[np.abs(matrix_clean) < threshold] = 0
    return matrix_clean

def standard_to_pinocchio(model, q: np.ndarray) -> np.ndarray:
    """Convert standard joint angles (rad) to Pinocchio joint angles"""
    q_pin = np.zeros(model.nq)
    for i, j in enumerate(model.joints[1:]):
        if j.nq == 1:
            q_pin[j.idx_q] = q[j.idx_v]
        else:
            # cos(theta), sin(theta)
            q_pin[j.idx_q:j.idx_q+2] = np.array([np.cos(q[j.idx_v]), np.sin(q[j.idx_v])])
    return q_pin

def normalize_angle(angle):
    """Normalize angle to [-π, π]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def pinocchio_to_standard(model, q_pin: np.ndarray) -> np.ndarray:
    """Convert Pinocchio joint angles to standard joint angles (rad)"""
    q = np.zeros(model.nv)
    for i, j in enumerate(model.joints[1:]):
        if j.nq == 1:
            q[j.idx_v] = q_pin[j.idx_q]
            # if j.type == pin.JointType.JointModelRZ:  # For continuous revolute joints
            #     q[j.idx_v] = normalize_angle(q[j.idx_v])
        else:
            q_back = np.arctan2(q_pin[j.idx_q+1], q_pin[j.idx_q])
            q[j.idx_v] = normalize_angle(q_back)
    return q

def R_matrix_to_euler(R):
    """
    Convert a 3x3 rotation matrix to roll, pitch, and yaw angles (XYZ convention).
    
    Parameters:
        R (numpy.ndarray): A 3x3 rotation matrix.
    
    Returns:
        tuple: (roll, pitch, yaw) in radians.
    """
    if R.shape != (3, 3):
        raise ValueError("Input must be a 3x3 matrix")
    
    pitch = np.arcsin(-R[2, 0])
    
    if abs(R[2, 0]) < 0.99999:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:  # Handle Gimbal lock
        roll = np.arctan2(-R[0, 1], R[1, 1])
        yaw = 0
    
    return roll, pitch, yaw

def get_end_effector_pose(model, data, EE_frame_id, q: np.ndarray) -> np.ndarray:
    """Get current end-effector pose"""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacement(model, data, EE_frame_id)
    T = data.oMf[EE_frame_id]
    # position = T.translation
    # rotation = np.degrees(pin.rpy.matrixToRpy(T.rotation))
    # return np.concatenate([position, rotation])
    return T

    # print("oMdes.translation:", oMdes.translation[i])
    # for i in range(0, len(model.frames)):
    #     pin.updateFramePlacement(model, data, i)
    #     oMact = data.oMf[i]
    #     print( "i:", "frame_id", "\noMact:", oMact)


    # for joint_id in range(len(model.joints)):
    #     oMact = data.oMi[joint_id]
    #     joint_name = model.names[joint_id]
    #     print(f"Joint ID: {joint_id}, Joint Name: {joint_name}\noMact: \n{clean_and_print_matrix(oMact)}")