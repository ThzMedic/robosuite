from collections import OrderedDict

import numpy as np
import sys
np.set_printoptions(precision=4, suppress=True)
np.set_printoptions(threshold=sys.maxsize)
import robosuite as suite
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import LemonObject, SphereObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat

from robosuite.controllers import load_part_controller_config
# control_config = load_part_controller_config(default_controller="JOINT_POSITION") # this doesn't even work...
from robosuite.controllers import load_composite_controller_config
from robosuite.kinematics.pinocchio_ik import compute_ik

from camera_path_6d import generate_upper_hemisphere_path_with_orientation
from ICP_tool_box import rotate_frame_on_ball

import mujoco
import time
import matplotlib.pyplot as plt

class Sphere(ManipulationEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        # self.table_offset = np.array((0, 0, 0.7))  # made changes
        self.table_offset = np.array((0, 0, 0.5))  # made changes
        # Omron LD-60 Mobile Base setting
        self.init_torso_height = 0.342

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
        )

    def reward(self, action):
        """
        Placeholder reward function
        Args:
            action (np.array): Action to execute within the environment
        Returns:
            float: Reward from environment
        """
        # For now, return a constant reward of 0 since we're just observing
        return 0.0

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        # Initialize sphere instead of lemon

        self.ball = SphereObject(
            name="sphere", # has to match the model="sphere" in the xml file
        )
        

        # No need to modify collision properties as they're set in the object initialization

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.ball)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.ball,
                x_range=[-0.1124, -0.1124],
                y_range=[0.3297, 0.3297],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=1.1-0.5,  # _ - 0.5 meter above the table
            )
            # left_ee_pos: array([-0.1124,  0.3297,  0.9375])

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.ball,
        )

    def _setup_references(self):
        """
        Sets up references to important components
        """
        super()._setup_references()

        # Additional object references from this env
        self.ball_body_id = self.sim.model.body_name2id(self.ball.root_body)

    def _setup_observables(self):
        """
        Sets up observables
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # define observables modality
            modality = "object"

            # ball-related observables
            @sensor(modality=modality)
            def ball_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.ball_body_id])

            @sensor(modality=modality)
            def ball_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.ball_body_id]), to="xyzw")

            sensors = [ball_pos, ball_quat]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        # set the mobilebase joint torso height if it exists
        self.deterministic_reset = True
        active_robot = self.robots[0]
        if active_robot.robot_model._torso_joints is not None:
            # dont need this since it's in super.reset()
            # torso_name = active_robot.robot_model._torso_joints[0]
            # self.sim.data.qpos[self.sim.model.get_joint_qpos_addr(torso_name)] = self.init_torso_height
            # # also set the initial torso height in the robot model
            active_robot.init_torso_qpos = np.array([self.init_torso_height,])

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
        else:
            # Deterministic reset -- set all objects to their specified positions
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                # new_pos = np.array([0.1, 1.0, 1.0])
                # new_quat = np.array([1, 0, 0, 0])  # Keep fixed orientation
                # self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([new_pos, new_quat]))
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

        super()._reset_internal()


    def _apply_gravity_compensation(self):
        """
        Computes the control needed to compensate for gravity to hold the arm in place.
        and applies it to the robot.
        """
        # Zero out accelerations for the full simulation (ensure the state is appropriate)
        # self.sim.data.qacc[:] = 0

        # Get the total number of degrees of freedom from the raw MuJoCo model
        # n_dof = self.sim.model._model.nv

        # # Preallocate a 2D column vector for the computed torques with shape (n_dof, 1)
        # gravity_torques_full = np.zeros((n_dof, 1), dtype=np.float64)

        # # Use the underlying raw model and data objects (successful but not needed)
        # mujoco.mj_rne(self.sim.model._model, self.sim.data._data, 0, gravity_torques_full)

        # For each robot, extract the relevant torques and assign them as control inputs
        for robot in self.robots:
            indices = robot._ref_joint_pos_indexes
            gravity_compensation = self.sim.data.qfrc_bias[indices]
            
            control_indices = robot._ref_arm_joint_actuator_indexes
            self.sim.data.ctrl[control_indices] = gravity_compensation

    def _jog_robot_to_pose(self, desired_arm_pos, desired_torso_height=0.342):
        """
        Jog the robot to a desired arm position and torso height.
        """
        active_robot = self.robots[0]

        # Preparing Input for the default_dual_kinova3 controller (HybridMobileBase)
        action_dict = {}
        for arm in active_robot.arms:
            # got the following syntex from demo_sensor_corruption.py
            if arm == "right":
                action_dict[arm] = desired_arm_pos[:7]
            if arm == "left":
                action_dict[arm] = desired_arm_pos[7:]
            action_dict[f"{arm}_gripper"] = np.zeros(active_robot.gripper[arm].dof)

        action_dict["torso"] = np.array([desired_torso_height,])
        action_dict["base"] = np.array([0.0, 0.0, 0.0])

        env_action = active_robot.create_action_vector(action_dict)

        left_arm_joints = env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes[7:14]]

        joint_error = desired_arm_pos[7:] - left_arm_joints
        target_reached_bool = np.all(np.abs(joint_error < 0.01))

        return env_action, target_reached_bool

    def _ik_left_arm(self, p_wd_target, R_wd_target, lq0):
        """
        Performs inverse kinematics for the left arm to reach a target position and orientation.
        """
        # Compute the left-most point (assuming positive x is right)
        # p_target = sphere_center + np.array([-sphere_radius, 0, 0])
        # p_target = sphere_center

        # HARD CODED for desired joints to be all 0
        # p_target = np.array([-0.56  ,  1.2454,  1.2994])
        # R_desired = np.array([[ 1., -0.,  0.],
        #                     [ 0.,  0.,  1.],
        #                     [-0., -1.,  0.]])

        # GET T_wd_base
        lbase_id = self.sim.model.body_name2id('robot0_left_arm_fixed_base_link')
        p_wd_lbase = self.sim.data.body_xpos[lbase_id]
        R_wd_lbase = self.sim.data.body_xmat[lbase_id].reshape(3, 3)
        # print("Left base position:", p_wd_lbase)
        # print("Left base rotation:\n", R_wd_lbase)
        T_wd_lbase = np.eye(4)
        T_wd_lbase[:3, :3] = R_wd_lbase
        T_wd_lbase[:3, 3] = p_wd_lbase
        
        # GET end effector position for testing
        lhand_id = self.sim.model.body_name2id('robot0_left_end_effector')
        R_wd_ee = self.sim.data.body_xmat[lhand_id].reshape(3, 3)
        p_wd_ee = data.xpos[lhand_id]

        # R_wd_target = R_wd_ee # alternative R_wd_ee for testing
        # p_wd_target = p_wd_ee # alternative p_wd_ee for testing
        # GET T_wd_target 
        T_wd_target = np.eye(4)
        T_wd_target[:3, :3] = R_wd_target
        T_wd_target[:3, 3] = p_wd_target
        # T_wd_target[:3, 3] = p_wd_ee

        T_lbase_target = np.linalg.inv(T_wd_lbase) @ T_wd_target

        # Path to the robot's URDF (update with your actual URDF file)
        urdf_path = "robosuite/models/assets/robots/dual_kinova3/leonardo.urdf"
        # Name of the left end-effector frame (adjust as needed)
        
        # Initial configuration for the left arm (using full robot qpos; adjust joint indices)
        # q0 = self.robots[0].init_qpos.copy()
        # lq0 = self.sim.data.qpos[self.robots[0]._ref_joint_pos_indexes[7:14]]

        # Name of the left end-effector frame
        left_ee = "end_effector"

        # Compute the inverse kinematics solution using Pinnocchio
        q_sol = compute_ik(urdf_path, left_ee, T_lbase_target, lq0)
        # print("initial qpos:", lq0)
        # print("IK solution:", q_sol)
        # print("", q_sol == lq0)

        # Set left arm joints angles which are indexed from 7 to 14.
        desired_arm_pos = self.robots[0].init_qpos.copy()
        desired_arm_pos[7:14] = q_sol
        
        return desired_arm_pos
    
    def generate_pose_matrices(self, center, radius, num_points):
        """
        Generate pose matrices for a set of points on the upper hemisphere around a sphere.
        """
        path_points = generate_upper_hemisphere_path_with_orientation(radius, num_points)
        pose_matrices = []

        print("\ncenter:", center)
        for i, point in enumerate(path_points):
            print("i: ", i)
            x, y, z, yaw = point
            print(f"point: {point}")
            position = center + np.array([x, y, z])
            print("position: ", position)

            direction = -(position - center)  # Direction vector from the flower center to camera
            print("direction: ", direction)
            pitch_angle = np.arctan2(direction[0], direction[2])
            # print("pitch angle: ", pitch_angle)
            # roll_angle = 0
            roll_angle = np.arctan2(direction[2], direction[1]) # - 3*np.pi/2
            print("roll angle: ", roll_angle)
            yaw_angle = yaw
            # print("yaw angle: ", yaw_angle)
            # Get a transformation matrix H that orients the camera to face the flower (center)
            # H = rotate_frame_on_ball(direction, roll=roll_angle, pitch=pitch_angle, yaw=yaw_angle)
            H = rotate_frame_on_ball(direction, roll=roll_angle, pitch=0, yaw=0)
            # print("H: \n", H)
            
            z_axis = direction / np.linalg.norm(direction)
            up = np.array([0, 0, 1])  # Assuming the camera's up direction is along the z-axis
            if np.allclose(z_axis, up) or np.allclose(z_axis, -up):
                up = np.array([0, 1, 0])

            x_axis = np.cross(up, z_axis)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            H[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
            H[:3, 3] = position

            pose_matrices.append(H)

        return pose_matrices
            

class TimeKeeper:
    def __init__(self, desired_freq=60):
        self.period = 1.0 / desired_freq
        self.last_time = time.perf_counter()
        self.time_accumulator = 0
        self.frame_count = 0
        self.start_time = self.last_time

    def should_step(self):
        current_time = time.perf_counter()
        frame_time = current_time - self.last_time
        self.last_time = current_time
        self.time_accumulator += frame_time
        return self.time_accumulator >= self.period

    def consume_step(self):
        self.time_accumulator -= self.period
        self.frame_count += 1

    def get_fps(self):
        elapsed = time.perf_counter() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0

if __name__ == "__main__":

    simulation_time = 100.0 # seconds
    env_step_size = 0.0001 # seconds
    horizon = int(simulation_time / env_step_size)
    # Create environment
    # note default controller is in "robosuite/controllers/config/robots/default_dualkinova3.json"
    # which uses JOINT_POSITION part_controller for both arm in the HYBRID_MOBILE_BASE type.
    env = suite.make(
        env_name="Sphere",
        robots="DualKinova3",
        # controller_configs=load_composite_controller_config(controller="BASIC"), 
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        horizon=horizon,
    )

    # Reset the environment
    env.reset()

    active_robot = env.robots[0]

    # Get initial joint positions for both arms
    # ways to retrieve joint positions
    # right_arm_joints = env.sim.data.qpos[active_robot._ref_arm_joint_pos_indexes[:7]]  # First 7 joints for right arm
    # left_arm_joints = env.sim.data.qpos[active_robot._ref_arm_joint_pos_indexes[7:]]   # Next 7 joints for left arm
    
    desired_arm_positions = active_robot.init_qpos
    # desired_arm_positions[7:14] = [1.5707963267948966, -1.5707963267948966, 1.5707963267948966, -1.5707963267948966, 0.0, -0.5235987755982988, -1.5707963267948966]
    desired_torso_height = env.init_torso_height

    # Preparing Input for the default_dual_kinova3 controller (HybridMobileBase)
    action_dict = {}
    for arm in active_robot.arms:
        # got the following syntex from demo_sensor_corruption.py
        if arm == "right":
            action_dict[arm] = desired_arm_positions[:7]
        if arm == "left":
            action_dict[arm] = desired_arm_positions[7:]
        action_dict[f"{arm}_gripper"] = np.zeros(active_robot.gripper[arm].dof)

    action_dict["torso"] = np.array([desired_torso_height,])
    action_dict["base"] = np.array([0.0, 0.0, 0.0])

    env_action = active_robot.create_action_vector(action_dict)
    # assess action dimension
    # to inspect use
    # print(active_robot.composite_controller._action_split_indexes)


    # Get model and data
    model = env.sim.model._model
    data = env.sim.data._data
    
    # Set smaller timestep for more accurate physics simulation
    model.opt.timestep = env_step_size  # commented out to use the default timestep


    # Lists to store time, force and position data
    times = []
    forces = []
    z_positions = []
    contact_object = 'sphere_g0'

    # ball_body_id = env.sim.model.body_name2id('sphere_main')

    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        # Set initial camera parameters
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -45
        viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])

        time_keeper = TimeKeeper(desired_freq=1/model.opt.timestep)
        
        # get the initial pose of the left end effector
        left_ee_body_id = env.sim.model.body_name2id('robot0_left_end_effector')
        left_ee_pos = data.xpos[left_ee_body_id]
        R_wd_lee = env.sim.data.body_xmat[left_ee_body_id].reshape(3, 3)
        
        ball_body_id = env.sim.model.body_name2id('sphere_main')
        p_wd_ball = env.sim.data.body_xpos[ball_body_id]
        R_wd_ball = env.sim.data.body_xmat[ball_body_id].reshape(3, 3)

        print("Ball position:", p_wd_ball)
        print("Ball rotation:\n", R_wd_ball)

        p_wd_lee = env.sim.data.body_xpos[left_ee_body_id]
        
        print("Bool: ", left_ee_pos == p_wd_lee)
        
        print("Left hand position:", p_wd_lee)
        print("Left hand rotation:\n", R_wd_lee)
        # sphere_center = data.xpos[env.sim.model.body_name2id('sphere_main')]
        
        # desired_joint = env._ik_left_arm(left_ee_pos, sphere_radius)
        
        # circle_radius = 0.1  # radius of the circle around the hand
        # num_points = 16      # number of points in the circle        
        
        # desired_pos_list = [] # store transformation matrix        
        
        # for i in range(num_points):
        #     angle = 2 * np.pi * i / num_points
        #     # Create circle in the XY plane around the hand
        #     x = p_wd_lee[0] + circle_radius * np.cos(angle)
        #     y = p_wd_lee[1] + circle_radius * np.sin(angle)
        #     z = p_wd_lee[2]  # keep same height as hand
        #     desired_pos_list.append(np.array([x, y, z]))        
        
        center = p_wd_ball
        radius = 0.15
        num_points = 12
        desired_list = env.generate_pose_matrices(center, radius, num_points)
        print("Desired list: ", desired_list)

        current_action_index = 0
        last_action_time = 0
        action_interval = 1.0  # Change action every n seconds
        target_reached_bool = False
        time_interval_start = False

        lq0 = env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes[7:14]]

        desired_joint = env._ik_left_arm(left_ee_pos, R_wd_lee, lq0)
        
        desired_joints = []
        # Generate desired poses vector
        print("center:", center)
        print("R_hand:", R_wd_lee)
        for i in range(len(desired_list)):            
            desired_pos = desired_list[i][:3, 3]
            desired_R = desired_list[i][:3, :3]
            # print("Desired position:", desired_pos)
            # print("Desired rotation:\n", desired_R)
            print("desired_Transform:\n", desired_list[i])
            
            R_desired = np.eye(3) # R_wd_lee
            R_desired = desired_R
            desired_joint = env._ik_left_arm(desired_pos, R_desired, lq0)
            lq0 = desired_joint
            desired_joints.append(desired_joint)

        desired_joints = [desired_joints[-1]]
        
        for i in range(len(desired_list) - 2, -1, -1):            
            print("i: ", i)
            desired_pos = desired_list[i][:3, 3]
            desired_R = desired_list[i][:3, :3]
            # print("Desired position:", desired_pos)
            # print("Desired rotation:\n", desired_R)
            print("desired_Transform:\n", desired_list[i])
            
            R_desired = np.eye(3) # R_wd_lee
            R_desired = desired_R
            desired_joint = env._ik_left_arm(desired_pos, R_desired, lq0)
            lq0 = desired_joint
            desired_joints.insert(0, desired_joint)
    
        print("Desired joint angles:")
        for i, joint in enumerate(desired_joints):
            print(f"Desired Rotation\n {i}: {desired_list[i][:3, :3]}")
            print(f"Pose \n{i}: {joint}")

        desired_joint = desired_joints[0]

        while viewer.is_running() and not env.done and data.time < simulation_time:
            if time_keeper.should_step():
                # Simulation step
                
                # Record data and update viewer
                
                # data.ctrl[:] = 0  # Disable controller
                
                # env._apply_gravity_compensation()

                ####Controlling the ball ######
                # Apply a force to the ball
                
                # Step the simulation
                # env.sim.step()
                # mujoco.mj_step(model, data)

                # # jog both arm to zero configuration 
                # zeros_config = np.zeros(14)
                # env_action = env._jog_robot_to_pose(zeros_config, desired_torso_height)
                
                # set last action time when the target is reached
                if target_reached_bool and not time_interval_start:
                    print(f"Target reached at time {data.time:.2f}")
                    last_action_time = data.time
                    time_interval_start = True

                if data.time - last_action_time >= action_interval:
                    current_action_index = (current_action_index + 1) % len(desired_list)
                    
                    desired_joint = desired_joints[current_action_index]
                    
                    time_interval_start = False
                    last_action_time = data.time
                    print(f"Switching to action {current_action_index} at time {data.time:.2f}")
                
                # Apply the current action
                
                # desired_joint = env._ik_left_arm(left_ee_pos, sphere_radius)
               
                # env_action = env._jog_robot_to_pose(desired_joint)
                # env.step(env_action)

                env_action, target_reached_bool = env._jog_robot_to_pose(desired_joint)
                # env_action = np.zeros_like(env_action)
                env.step(env_action)

                total_force = 0
                # Iterate over all detected contacts
                for i in range(data.ncon):
                    contact = data.contact[i]
                    # Check if contact involves the table and the object of interest
                    if ((contact.geom1 == env.sim.model.geom_name2id('table_collision') and 
                        contact.geom2 == env.sim.model.geom_name2id(contact_object)) or
                        (contact.geom2 == env.sim.model.geom_name2id('table_collision') and 
                        contact.geom1 == env.sim.model.geom_name2id(contact_object))):
                        
                        # Compute contact force (6D: 3D force + 3D torque)
                        force_vector = np.zeros(6)
                        mujoco.mj_contactForce(model, data, i, force_vector)
                        
                        # Extract normal force (first component in the contact frame)
                        normal_force = force_vector[0]
                        total_force += normal_force
                
                # Record positions, times, and forces
                ball_body_id = env.sim.model.body_name2id('sphere_main')
                z_positions.append(data.xpos[ball_body_id][2])
                times.append(data.time)
                forces.append(total_force)  # This now includes the spike
                
                # # Viewer updates (unchanged)
                # with viewer.lock():
                #     viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1

                viewer.sync()
                time_keeper.consume_step()
                
                # # Optional: Monitor performance
                # if time_keeper.frame_count % 60 == 0:
                #     print(f"Current FPS: {time_keeper.get_fps():.2f}")

    # # Create subplots for force and position
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # # Plot forces
    # ax1.plot(times, forces)
    # ax1.set_xlabel('Time (s)')
    # ax1.set_ylabel('Impact Force (N)')
    # ax1.set_title('Ball-Table Impact Force over Time')
    # ax1.grid(True)
    
    # # Plot z position
    # ax2.plot(times, z_positions)
    # ax2.set_xlabel('Time (s)')
    # ax2.set_ylabel('Z Position (m)')
    # ax2.set_title('Ball Z Position over Time')
    # ax2.grid(True)
    
    # plt.tight_layout()
    # plt.show()