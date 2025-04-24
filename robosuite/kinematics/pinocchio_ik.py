import pinocchio as pin
import numpy as np
np.set_printoptions(precision=4, suppress=True, threshold=1e-4)
from numpy.linalg import norm, solve

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
    print("oMdes: \n", clean_and_print_matrix(oMdes))

    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()# model = pin.buildModelFromUrdf(urdf_path)

    # Use the default Pinocchio solver (e.g., Levenberg-Marquardt) as a placeholder
    tool_frame_id = model.getFrameId(ee_frame)
    q = q0.copy()

    print("len_q_pin", model.nq)
    print("len_q", model.nv)

    q_pin = standard_to_pinocchio(model, q)
    
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
            print('%d: error = %s' % (i, err.T))
        i += 1
    
    if success:
        print("Convergence achieved!")
        print("Iterations:", i)
    else:
        print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")
    
    print('\nresult: %s' % q_pin.flatten().tolist())
    print('\nfinal error: %s' % err.T)
    
    print("q0:", q0)
    print("q_pin:", q_pin)
    q = pinocchio_to_standard(model, q_pin)
    print("q final:", q)
    return q

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

# def T_matrix_to_euler(T):
#     """
#     Extracts roll, pitch, and yaw (RPY) angles from a 4x4 transformation matrix.

#     Parameters:
#         T (numpy.ndarray): A 4x4 transformation matrix.

#     Returns:
#         tuple: (roll, pitch, yaw) in radians.
#     """
#     R = T[:3, :3]  # Extract the rotation matrix

#     # Compute pitch
#     if abs(R[2, 0]) != 1:
#         pitch = -np.arcsin(R[2, 0])
#         cos_pitch = np.cos(pitch)
#         roll = np.arctan2(R[2, 1] / cos_pitch, R[2, 2] / cos_pitch)
#         yaw = np.arctan2(R[1, 0] / cos_pitch, R[0, 0] / cos_pitch)
#     else:
#         # Gimbal lock case
#         yaw = 0  # Can be set to any value
#         if R[2, 0] == -1:
#             pitch = np.pi / 2
#             roll = np.arctan2(R[0, 1], R[0, 2])
#         else:
#             pitch = -np.pi / 2
#             roll = np.arctan2(-R[0, 1], -R[0, 2])

#     return roll, pitch, yaw


# def parse_urdf_origin(origin_str):
#     """Parse URDF origin tag string into xyz and rpy values"""
#     # Parse origin tag like: <origin rpy="-1.5708 0 0" xyz="0 -0.058 0.232548"/>
#     xyz_str = origin_str[origin_str.find('xyz="')+5:].split('"')[0]
#     rpy_str = origin_str[origin_str.find('rpy="')+5:].split('"')[0]
    
#     # Convert strings to numpy arrays
#     xyz = np.array([float(x) for x in xyz_str.split()])
#     rpy = np.array([float(x) for x in rpy_str.split()])
    
#     return xyz, rpy[0], rpy[1], rpy[2]  # returns position and roll,pitch,yaw

# # Example usage:
# origin_tag = '<origin rpy="-1.5708 0 0" xyz="0 -0.058 0.232548"/>'
# xyz, roll, pitch, yaw = parse_urdf_origin(origin_tag)

# T_ee_1 = data.oMi[7]
# pin.forwardKinematics(model,data,q_pin)
# T_ee_2 = data.oMi[7]
# T_trans = T_ee_2.translation
# T_rot = T_ee_2.rotation
# var = T_ee_1 == T_ee_2
# T_rot = T_ee_2.rotation
# var = T_ee_1 == T_ee_2
# T_trans_base = data.oMi[0].translation
# T_rot_base = data.oMi[0].rotation

# """Creates the base transform from the URDF <origin> tag values"""
# # Values from URDF: <origin rpy="-1.5708 0 0" xyz="0 -0.058 0.232548"/>
# xyz_base = np.array([0, -0.058, 0.232548])
# roll = -1.5708  # -pi/2
# pitch = 0
# yaw = 0

# # Create rotation matrix from roll, pitch, yaw
# R_base = pin.rpy.rpyToMatrix(roll, pitch, yaw)

# # Create SE3 transform
# T_base = pin.SE3(R_base, xyz_base)
# print("Base joint origin transformation from URDF:")
# print("Translation:", T_base.translation)
# print("Rotation:\n", T_base.rotation)

# model = pin.buildModelFromUrdf(urdf_path)
    # data = model.createData()
    # # Use the default Pinocchio solver (e.g., Levenberg-Marquardt) as a placeholder
    # q = q0.copy()
    # q_pin = standard_to_pinocchio(model, q)
    # for _ in range(max_iter):
    #     pin.forwardKinematics(model, data, q_pin)
    #     pin.updateFramePlacements(model, data)
    #     current_pose = data.oMf[model.getFrameId(ee_frame)]
    #     error_mat = pin.log(current_pose.inverse() * pin.SE3(target_pose[:3,:3], target_pose[:3,3]))
    #     err = np.linalg.norm(error_mat)
    #     if err < tol:
    #         break
    #     # Compute Jacobian
    #     J = pin.computeFrameJacobian(model, data, q_pin, model.getFrameId(ee_frame))
    #     dq = np.linalg.lstsq(J, error_mat, rcond=None)[0]
    #     q += dq
    # return q

    # print("oMdes.translation:", oMdes.translation[i])
    # for i in range(0, len(model.frames)):
    #     pin.updateFramePlacement(model, data, i)
    #     oMact = data.oMf[i]
    #     print( "i:", "frame_id", "\noMact:", oMact)


    # for joint_id in range(len(model.joints)):
    #     oMact = data.oMi[joint_id]
    #     joint_name = model.names[joint_id]
    #     print(f"Joint ID: {joint_id}, Joint Name: {joint_name}\noMact: \n{clean_and_print_matrix(oMact)}")