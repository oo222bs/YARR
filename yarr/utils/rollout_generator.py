from multiprocessing import Value
import copy
import numpy as np
import torch
from yarr.agents.agent import Agent, ActResult
from yarr.envs.env import Env
from yarr.utils.transition import ReplayTransition
from rlbench.backend import utils
from torch.utils.tensorboard import SummaryWriter
from yarr.utils.log_writer import LogWriter
import torch
class RolloutGenerator(object):

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, step_signal: Value, env: Env, agent: Agent, writer: LogWriter,
                  episode_length: int, timesteps: int,
                  eval: bool, init_arm_pose: list = None, eval_demo_seed: int = 0,
                  record_enabled: bool = False, depth: bool=False, train: bool=True, with_baseline: bool=True,
                  agent_old: Agent=None, rl_alg='REINFORCE'):

        if eval:
            obs = env.reset_to_demo(eval_demo_seed, train)#env.reset_to_demo(eval_demo_seed)
        else:
            obs = env.reset()

        env.env._pyrep.step()
        agent.reset()
        if type(init_arm_pose)==list and eval_demo_seed>=1000:
            ran_pose_flag = torch.randint(4, (1,))
            if ran_pose_flag==1:
                ran_pose = torch.randint(len(init_arm_pose), (1,))
                env.step(init_arm_pose[ran_pose])
        obs_s = env._task.get_observation()
        init_behavior = obs_s.gripper_pose
        init_behavior = np.append(init_behavior, [obs_s.gripper_open, 1])#obs_s.ignore_collisions])#1])#ignore_collisions])
        obs_s = env.extract_obs(obs_s)
        beh_history = []
        if depth:
            obs_s_w_depth_copy = obs_s['wrist_depth']
            obs_s['wrist_depth'] = utils.float_array_to_rgb_image(
                obs_s['wrist_depth'][0], scale_factor=2 ** 24 - 1).convert('L')
        if rl_alg == 'PPO':
            with torch.no_grad():
                hid = agent_old.initialise_hidden(obs_s, torch.Tensor(init_behavior).unsqueeze(0).unsqueeze(0).to('cuda'), env._lang_goal)
        elif rl_alg == 'DDPG':
            with torch.no_grad():
                hid = agent.initialise_hidden(obs_s, torch.Tensor(init_behavior).unsqueeze(0).unsqueeze(0).to('cuda'), env._lang_goal)
        else:
            hid = agent.initialise_hidden(obs_s, torch.Tensor(init_behavior).unsqueeze(0).unsqueeze(0).to('cuda'), env._lang_goal)
        if depth:
            obs_s['wrist_depth'] = obs_s_w_depth_copy
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs_s.items()}
        beh_history.append(torch.Tensor(init_behavior).unsqueeze(0).unsqueeze(0).to('cuda'))
        if train:
            log_probs = []
            if with_baseline:
                baseline_preds = []
                #if rl_alg == 'DDPG':
                    #critic_target_preds = []
                if rl_alg == 'PPO':
                    log_probs_old = []
                    baseline_preds_old = []
        if train == False and 'SlideBlockToColorTargetSimplified' in env._task._task.__str__():
            initial_block_pose = env._task._task.block.get_pose()
        for step in range(episode_length):
            prepped_data = {k:torch.from_numpy(np.array(v)) for k, v in obs_history.items()}
            if depth:
                prepped_data['wrist_depth'] = utils.float_array_to_rgb_image(
                    prepped_data['wrist_depth'].squeeze().detach().cpu().numpy(), scale_factor=2 ** 24 - 1).convert('L')
            if train:
                if with_baseline:
                    if rl_alg=='DDPG':
                        obs_frgb = prepped_data['front_rgb']
                        obs_lsrgb = prepped_data['left_shoulder_rgb']
                        obs_rsrgb = prepped_data['right_shoulder_rgb']
                        obs_wrgb = prepped_data['wrist_rgb']
                        with torch.no_grad():
                            act_result_joints = agent.get_action_ddpg(prepped_data, hid, beh_history[step])
                    elif rl_alg=='PPO':
                        observation_for_agent = prepped_data
                        with torch.no_grad():
                            act_result_joints, log_prob_old, baseline_out_old = agent_old.act_with_exploration(prepped_data, hid, beh_history[step], True, with_baseline, rl_alg, eval_demo_seed)
                            log_probs_old.append(log_prob_old)
                            baseline_preds_old.append(baseline_out_old)
                    elif rl_alg=='CACLA':
                        if step>0:
                            hid = agent.initialise_hidden(obs_s,torch.Tensor(init_behavior).unsqueeze(0).unsqueeze(0).to('cuda'), env._lang_goal)
                        act_result_joints, actor_out, baseline_inp = agent.act_with_exploration(prepped_data, hid, beh_history[step], True, with_baseline, rl_alg, eval_demo_seed)
                    else:
                        act_result_joints, log_prob, baseline_out = agent.act_with_exploration(prepped_data, hid, beh_history[step], True, with_baseline, rl_alg, eval_demo_seed)
                        baseline_preds.append(baseline_out)
                        log_probs.append(log_prob)
                else:
                    act_result_joints, log_prob = agent.act_with_exploration(prepped_data, hid, beh_history[step], True, with_baseline, rl_alg, eval_demo_seed)
                    log_probs.append(log_prob)
            else:
                act_result_joints = agent.act(prepped_data, hid, beh_history[step])

            act_result = ActResult(act_result_joints.squeeze().detach().cpu())
            if type(init_arm_pose)==list and step==0 and eval_demo_seed<1000:
                init_arm_pose.append(act_result)

            # Convert to np if not already
            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            extra_replay_elements = {k: np.array(v) for k, v in
                                     act_result.replay_elements.items()}

            transition = env.step(act_result)
            obs_tp1 = dict(transition.observation)

            obs_c = env._task.get_observation()
            behavior = obs_c.gripper_pose
            behavior = torch.Tensor(np.append(behavior, [obs_c.gripper_open])).unsqueeze(0).unsqueeze(0).to('cuda')
            behavior = torch.cat((behavior,act_result_joints[:,:,-1].unsqueeze(0)), -1)
            beh_history.append(behavior)

            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not transition.terminal
                if timeout:
                    transition.terminal = True
                    if "needs_reset" in transition.info:
                        transition.info["needs_reset"] = True

            obs_and_replay_elems = {}
            obs_and_replay_elems.update(obs)
            obs_and_replay_elems.update(agent_obs_elems)
            obs_and_replay_elems.update(extra_replay_elements)

            for k in obs_history.keys():
                obs_history[k].append(transition.observation[k])
                obs_history[k].pop(0)

            transition.info["active_task_id"] = env.active_task_id

            replay_transition = ReplayTransition(
                obs_and_replay_elems, act_result.action, transition.reward,
                transition.terminal, timeout, summaries=transition.summaries,
                info=transition.info)

            if transition.terminal or timeout:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                if len(act_result.observation_elements) > 0:
                    prepped_data = {k:torch.from_numpy(np.array(v)) for k, v in obs_history.items()}
                    act_result = agent.act(step_signal.value, prepped_data,
                                           deterministic=eval)
                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                           act_result.observation_elements.items()}
                    obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1
                if train==False:
                    if 'PushButtons' in env._task._task.__str__():
                        distance_to_target = np.sqrt(np.square(obs_c.gripper_pose[0]-env._task._task.target_buttons[0].get_pose()[0])+
                                                     np.square(obs_c.gripper_pose[1]-env._task._task.target_buttons[0].get_pose()[1])+
                                                     np.square(obs_c.gripper_pose[2]-env._task._task.target_buttons[0].get_pose()[2]))
                        distance_to_distractor1 = np.sqrt(np.square(obs_c.gripper_pose[0]-env._task._task.target_buttons[1].get_pose()[0])+
                                                     np.square(obs_c.gripper_pose[1]-env._task._task.target_buttons[1].get_pose()[1])+
                                                     np.square(obs_c.gripper_pose[2]-env._task._task.target_buttons[1].get_pose()[2]))
                        distance_to_distractor2 = np.sqrt(np.square(obs_c.gripper_pose[0]-env._task._task.target_buttons[2].get_pose()[0])+
                                                     np.square(obs_c.gripper_pose[1]-env._task._task.target_buttons[2].get_pose()[1])+
                                                     np.square(obs_c.gripper_pose[2]-env._task._task.target_buttons[2].get_pose()[2]))
                        if distance_to_target < np.minimum(distance_to_distractor1, distance_to_distractor2):
                            print('The gripper has gone to the correct button.')
                    elif 'PickUpCup' in env._task._task.__str__():
                        distance_to_target = np.sqrt(np.square(obs_c.gripper_pose[0]-env._task._task.cup1.get_pose()[0])+
                                                     np.square(obs_c.gripper_pose[1]-env._task._task.cup1.get_pose()[1])+
                                                     np.square(obs_c.gripper_pose[2]-env._task._task.cup1.get_pose()[2]))
                        distance_to_distractor = np.sqrt(np.square(obs_c.gripper_pose[0]-env._task._task.cup2.get_pose()[0])+
                                                     np.square(obs_c.gripper_pose[1]-env._task._task.cup2.get_pose()[1])+
                                                     np.square(obs_c.gripper_pose[2]-env._task._task.cup2.get_pose()[2]))
                        if distance_to_target < distance_to_distractor:
                            print('The gripper has gone to the correct cup.')
                    elif 'SlideBlockToColorTargetSimplified' in env._task._task.__str__():
                        movement_x = initial_block_pose[0]-env._task._task.block.get_pose()[0]
                        movement_y = initial_block_pose[1]-env._task._task.block.get_pose()[1]
                        initial_dist_x = initial_block_pose[0] - env._task._task._waypoint_paths[env._task._task._variation_index][1].get_pose()[0]
                        initial_dist_y = initial_block_pose[1] - env._task._task._waypoint_paths[env._task._task._variation_index][1].get_pose()[1]
                        if (np.absolute(movement_x) > np.absolute(movement_y)) and ((movement_x < 0 and initial_dist_x < 0) or (movement_x > 0 and initial_dist_x > 0)):
                            print('The gripper has moved the block in the right direction.')
                        elif (np.absolute(movement_x) < np.absolute(movement_y)) and (
                                (movement_y < 0 and initial_dist_y < 0) or (movement_y > 0 and initial_dist_y > 0)):
                            print('The gripper has moved the block in the right direction.')
                # update the agent based on the return and log probabilities (REINFORCE)
                if train:
                    if transition.reward != 100 and eval_demo_seed<100 and type(init_arm_pose)==list:
                        init_arm_pose.pop(-1)
                    if with_baseline:
                        if rl_alg=='CACLA':
                            state_value = agent.baseline(baseline_inp)
                            reward, critic_loss, delta_t, critic_target = agent.update_with_cacla(beh_history[-1], actor_out, transition.reward, state_value, step=step)
                            writer.add_scalar(1+eval_demo_seed*step, 'critic loss', critic_loss)
                            writer.add_scalar(eval_demo_seed, 'critic value', state_value)
                            writer.add_scalar(1+eval_demo_seed*step, 'delta T', delta_t)
                            writer.add_scalar(1 + eval_demo_seed * step, 'target', critic_target)
                        elif rl_alg=='REINFORCE':
                            reward, baseline_loss = agent.update_with_reinforce(log_probs, transition.reward, baseline_preds)
                            writer.add_scalar(eval_demo_seed, 'baseline loss', baseline_loss)
                        elif rl_alg == 'DDPG':
                            critic_target_out = torch.zeros([1,1,1]).to('cuda')
                        if rl_alg != 'PPO' and rl_alg!='DDPG':
                            writer.add_scalar(eval_demo_seed, 'reward', reward)
                            if reward == 1.00:
                                writer.add_scalar(eval_demo_seed, 'positive reward', 100.0)
                                writer.add_scalar(eval_demo_seed, 'null reward', 0.0)
                                writer.add_scalar(eval_demo_seed, 'negative reward', 0.0)
                            elif reward == 0.00:
                                writer.add_scalar(eval_demo_seed, 'null reward', 100.0)
                                writer.add_scalar(eval_demo_seed, 'positive reward', 0.0)
                                writer.add_scalar(eval_demo_seed, 'negative reward', 0.0)
                            else:
                                writer.add_scalar(eval_demo_seed, 'negative reward', 100.0)
                                writer.add_scalar(eval_demo_seed, 'positive reward', 0.0)
                                writer.add_scalar(eval_demo_seed, 'null reward', 0.0)
                    else:
                        if rl_alg == 'REINFORCE':
                            reward = agent.update_with_reinforce(log_probs, transition.reward)
                            writer.add_scalar(eval_demo_seed, 'reward', reward)
                        if reward == 1.00:
                            writer.add_scalar(eval_demo_seed, 'positive reward', 100.0)
                            writer.add_scalar(eval_demo_seed, 'null reward', 0.0)
                            writer.add_scalar(eval_demo_seed, 'negative reward', 0.0)
                        elif reward == 0.00:
                            writer.add_scalar(eval_demo_seed, 'null reward', 100.0)
                            writer.add_scalar(eval_demo_seed, 'positive reward', 0.0)
                            writer.add_scalar(eval_demo_seed, 'negative reward', 0.0)
                        else:
                            writer.add_scalar(eval_demo_seed, 'negative reward', 100.0)
                            writer.add_scalar(eval_demo_seed, 'positive reward', 0.0)
                            writer.add_scalar(eval_demo_seed, 'null reward', 0.0)
            else:
                if train:
                    if with_baseline:
                        if rl_alg == 'CACLA':
                            prepped_data = {k:torch.from_numpy(np.array(v)) for k, v in obs_history.items()}
                            with torch.no_grad():
                                next_state_value = agent.evaluate_next_state(prepped_data, hid, beh_history[-1])
                            state_value = agent.baseline(baseline_inp)
                            reward, critic_loss, delta_t, critic_target = agent.update_with_cacla(beh_history[-1], actor_out, transition.reward, state_value, step, next_state_value)
                            writer.add_scalar(1+eval_demo_seed*step, 'critic loss', critic_loss)
                            writer.add_scalar(1+eval_demo_seed*step, 'delta T', delta_t)
                            writer.add_scalar(1 + eval_demo_seed * step, 'target', critic_target)
                        elif rl_alg == 'DDPG':
                            prepped_data = {k:torch.from_numpy(np.array(v)) for k, v in obs_history.items()}
                            with torch.no_grad():
                                critic_target_out = agent_old.evaluate_next_state_ddpg(prepped_data, hid, beh_history[step + 1])

            if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                env.env._action_mode.arm_action_mode.record_end(env.env._scene,
                                                                steps=60, step_scene=True)

            obs = dict(transition.observation)
            if rl_alg == 'PPO' and train==True:
                yield replay_transition, log_prob_old, transition.reward, baseline_out_old, observation_for_agent, obs_s, init_behavior, env._lang_goal, beh_history[step]
            elif rl_alg == 'DDPG' and train==True:
                init_obs_rgb = obs_s['front_rgb']
                init_obs_lsrgb = obs_s['left_shoulder_rgb']
                init_obs_rsrgb = obs_s['right_shoulder_rgb']
                init_obs_wrgb = obs_s['wrist_rgb']
                yield replay_transition, [act_result_joints, transition.reward, env._lang_goal, init_obs_rgb, init_obs_lsrgb, init_obs_rsrgb,
                                          init_obs_wrgb, obs_frgb, obs_lsrgb, obs_rsrgb, obs_wrgb, init_behavior, beh_history[step],
                                          critic_target_out, transition.terminal, step]
            else:
                if type(init_arm_pose)==list:
                    yield replay_transition, init_arm_pose
                else:
                    yield replay_transition

            if transition.info.get("needs_reset", transition.terminal):
                return
