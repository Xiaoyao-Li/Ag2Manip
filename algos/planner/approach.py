import os
import copy
import pickle
import imageio
import numpy as np
from tqdm import tqdm
from isaacgym import gymapi
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

class APPROACH:
    def __init__(self, vec_env, cfg, save_goal=False, save_video=False):
        self.vec_env = vec_env
        self.env_name = cfg['name'].rsplit('@', 1)[0]
        self.traj_path = cfg['traj_path']
        self.traj_basedir = os.path.dirname(self.traj_path)
        self.traj_name = os.path.basename(self.traj_path).split('.')[0][7:]
        self.dummy_traj = pickle.load(open(self.traj_path, 'rb'))['trajectory']
        self.save_goal = save_goal
        self.save_video = save_video
        self.FSM_STATE = 'FREE'  ## StateDict: {'FREE', 'APPROACH', 'ATTACH'}

    def transfer(self, env_id=0):
        if self.env_name in ['frankakitchen@open_hingecabinet', 'frankakitchen@close_hingecabinet', 
                             'frankakitchen@open_microwave', 'frankakitchen@close_microwave',  ]:
            quat_default = np.array([0., 1., 0., 0.])
        elif self.env_name in ['frankakitchen@open_slidecabinet', 'frankakitchen@close_slidecabinet',
                               'frankakitchen@move_kettle', 'frankakitchen@pickup_kettle',
                               'frankakitchen@turnon_switch', 'frankakitchen@turnoff_switch',
                               'partmanip@open_wooddrawer_middle', 'partmanip@close_wooddrawer_middle',
                               'partmanip@open_dishwasher', 'partmanip@close_dishwasher',
                               'partmanip@press_button', 'partmanip@lift_lid',
                               'maniskill@turn_leftfaucet', 'maniskill@turn_rightfaucet',
                               'maniskill@close_door', 'maniskill@open_door',
                               'maniskill@insert_peg', 'maniskill@pickup_cube',
                               'maniskill@stack_cube', 'maniskill@pickup_clutterycb']:
            quat_default = np.array([0.707, 0.707, 0., 0.])
        else:
            raise NotImplementedError
        self.FSM_STATE = 'FREE'
        related_tm = None
        self.execute_traj = []  ## pandahand-atrtactor pose, gripper effort (positive leads to open)
        for i_step, act in enumerate(self.dummy_traj):            # print(f"slidecabinet: {act['rigid_bodies'][env_id, act['attached_body_handle'][env_id], :7]}")
            if act['attached_info_indices'] == -1: 
                pos = act['panda_hand'][env_id, :3]
                # quat = act['panda_hand'][env_id, 3:7]
                quat = quat_default
                attractor_pose = gymapi.Transform()
                attractor_pose.p = gymapi.Vec3(*pos)

                # rot_mat = Rotation.from_quat(quat).as_matrix() @ Rotation.from_euler('xyz', [3.14, -1.57, 3.14]).as_matrix()
                rot_mat = Rotation.from_quat(quat).as_matrix()
                rot_quat = Rotation.from_matrix(rot_mat).as_quat()
                attractor_pose.r = gymapi.Quat(*rot_quat)
                gripper_effort = 100.
            else:
                attached_info = act['attach_info'][act['attached_info_indices'][env_id]]
                actor_trans = act['attach_info'][act['attached_info_indices'][env_id]]['object_trans']
                actor_rot = act['attach_info'][act['attached_info_indices'][env_id]]['object_rot']
                attach_trans = attached_info['attach_info']['translation']
                attach_rot = attached_info['attach_info']['rotation_matrix'] @ Rotation.from_euler('xyz', [0., 1.57, 0.]).as_matrix()

                attach_trans = actor_rot @ attach_trans + actor_trans
                attach_rot = actor_rot @ attach_rot
                attach_trans = attach_trans - attach_rot @ np.array([0., 0., 0.058])
                attach_tm = np.eye(4)
                attach_tm[:3, :3] = attach_rot
                attach_tm[:3, 3] = attach_trans 
                attach_rot = Rotation.from_matrix(attach_rot).as_quat()
                if self.FSM_STATE == 'FREE':
                    self.FSM_STATE = 'APPROACH'
                    body_trans = act['rigid_bodies'][env_id, act['attached_body_handle'][env_id], :3]
                    body_rot = act['rigid_bodies'][env_id, act['attached_body_handle'][env_id], 3:7]
                    body_rot = Rotation.from_quat(body_rot).as_matrix()
                    body_tm = np.eye(4)
                    body_tm[:3, :3] = body_rot
                    body_tm[:3, 3] = body_trans
                    related_tm = attach_tm @ np.linalg.inv(body_tm)

                    #! plan the approach motion
                    attach_trans_back = attach_trans - Rotation.from_quat(attach_rot).as_matrix() @ np.array([0., 0., 0.2])
                    attractor_pose = gymapi.Transform()
                    attractor_pose.p = gymapi.Vec3(*attach_trans_back)
                    attractor_pose.r = gymapi.Quat(*attach_rot)
                    gripper_effort = 100.
                    self.execute_traj.append([attractor_pose, gripper_effort, copy.deepcopy(self.FSM_STATE)])
                    self.FSM_STATE = 'ATTACH'
                    attractor_pose = gymapi.Transform()
                    attractor_pose.p = gymapi.Vec3(*attach_trans)
                    attractor_pose.r = gymapi.Quat(*attach_rot)
                    gripper_effort = 100.
                else:
                    body_trans = act['rigid_bodies'][env_id, act['attached_body_handle'][env_id], :3]
                    body_rot = act['rigid_bodies'][env_id, act['attached_body_handle'][env_id], 3:7]
                    body_rot = Rotation.from_quat(body_rot).as_matrix()
                    body_tm_t = np.eye(4)
                    body_tm_t[:3, :3] = body_rot
                    body_tm_t[:3, 3] = body_trans
                    body_tm_t = body_tm_t @ np.linalg.inv(body_tm)
                    attach_tm = body_tm_t @ related_tm @ body_tm
                    attach_trans = attach_tm[:3, 3]
                    attach_rot = Rotation.from_matrix(attach_tm[:3, :3]).as_quat()

                    attractor_pose = gymapi.Transform()
                    attractor_pose.p = gymapi.Vec3(*attach_trans)
                    attractor_pose.r = gymapi.Quat(*attach_rot)
                    # gripper_effort = -100.
                    gripper_effort = -10.
                
            self.execute_traj.append([attractor_pose, gripper_effort, copy.deepcopy(self.FSM_STATE)])
        
        # smooth the excute_traj on attractor_pose
        SMOOTH_STEPS = 10
        transition_flag = False
        attach_flag = False
        execute_traj_smooth = []
        for i_step in range(len(self.execute_traj) - 1):
            lower_pos = np.array([self.execute_traj[i_step][0].p.x, self.execute_traj[i_step][0].p.y, self.execute_traj[i_step][0].p.z])
            upper_pos = np.array([self.execute_traj[i_step + 1][0].p.x, self.execute_traj[i_step + 1][0].p.y, self.execute_traj[i_step + 1][0].p.z])
            lower_quat = np.array([self.execute_traj[i_step][0].r.x, self.execute_traj[i_step][0].r.y, self.execute_traj[i_step][0].r.z, self.execute_traj[i_step][0].r.w])
            upper_quat = np.array([self.execute_traj[i_step + 1][0].r.x, self.execute_traj[i_step + 1][0].r.y, self.execute_traj[i_step + 1][0].r.z, self.execute_traj[i_step + 1][0].r.w])
            interp_rot = Slerp([0, 1], Rotation.from_quat([lower_quat, upper_quat]))
            smooth_steps = SMOOTH_STEPS
            if self.execute_traj[i_step + 1][2] == 'APPROACH':
                smooth_steps = 100
            elif self.execute_traj[i_step + 1][2] == 'ATTACH' and transition_flag == False:
                smooth_steps = 100
            for i_smooth in range(smooth_steps):
                i_smooth_pos = lower_pos + (upper_pos - lower_pos) * i_smooth / smooth_steps
                i_smooth_quat = interp_rot(i_smooth / smooth_steps).as_quat()
                attractor_pose = gymapi.Transform()
                attractor_pose.p = gymapi.Vec3(*i_smooth_pos)
                attractor_pose.r = gymapi.Quat(*i_smooth_quat)
                gripper_effort = self.execute_traj[i_step + 1][1]
                execute_traj_smooth.append([attractor_pose, gripper_effort])
            if self.execute_traj[i_step + 1][2] == 'ATTACH' and transition_flag == False:
                for _ in range(20):
                    gripper_effort = self.execute_traj[i_step + 2][1]
                    execute_traj_smooth.append([attractor_pose, gripper_effort])
                transition_flag = True
        self.execute_traj = execute_traj_smooth

    def run(self):
        # transfer to get the excute_traj
        self.transfer()
        scores = []
        indicators = []
        frames = []
        pbar = tqdm(total=len(self.execute_traj), desc='processing')
        print(f'TPATH: {self.traj_path}')
        for i_step, act in enumerate(self.execute_traj):
            infos = self.vec_env.task.step_plan(act)
            # print(f'Step: {i_step} | Score: {infos["success_scores"].cpu().item()}')
            scores.append(infos["success_scores"].cpu().item())
            indicators.append(infos["indicator"].cpu().numpy())
            pbar.update(1)
            pbar.set_description(f'{i_step}/{len(self.execute_traj)} | indicator: {infos["indicator"].cpu().numpy()}')
        
        # raise Warning('Not saving the execres to pkl file!')
        execres = {
            'scores': scores,
            'indicators': indicators,
        }
        execres_path = os.path.join(self.traj_basedir, f'execres_{self.traj_name}.pkl')
        pickle.dump(execres, open(execres_path, 'wb'))
        print(f'Save execres to {execres_path}')