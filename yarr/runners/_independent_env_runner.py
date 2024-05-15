import copy
import logging
import os
import time
import pandas as pd

from multiprocessing import Process, Manager
from multiprocessing import get_start_method, set_start_method
from typing import Any

import numpy as np
import torch
from yarr.agents.agent import Agent
from yarr.agents.agent import ScalarSummary
from yarr.agents.agent import Summary
from yarr.envs.env import Env
from yarr.utils.rollout_generator import RolloutGenerator
from yarr.utils.log_writer import LogWriter
from yarr.utils.process_str import change_case
from yarr.utils.video_utils import CircleCameraMotion, TaskRecorder

from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from yarr.runners._env_runner import _EnvRunner
from yarr.utils.stat_accumulator import StatAccumulator, SimpleAccumulator
#from peract.agents.xbit.xbit_agent import DDPGCritic
import random

class _IndependentEnvRunner(_EnvRunner):

    def __init__(self,
                 train_env: Env,
                 eval_env: Env,
                 agent: Agent,
                 timesteps: int,
                 train_envs: int,
                 eval_envs: int,
                 rollout_episodes: int,
                 eval_episodes: int,
                 training_iterations: int,
                 eval_from_eps_number: int,
                 episode_length: int,
                 kill_signal: Any,
                 step_signal: Any,
                 num_eval_episodes_signal: Any,
                 eval_epochs_signal: Any,
                 eval_report_signal: Any,
                 log_freq: int,
                 rollout_generator: RolloutGenerator,
                 save_load_lock,
                 current_replay_ratio,
                 target_replay_ratio,
                 weightsdir: str = None,
                 logdir: str = None,
                 env_device: torch.device = None,
                 previous_loaded_weight_folder: str = '',
                 num_eval_runs: int = 1,
                 ):

            super().__init__(train_env, eval_env, agent, timesteps,
                             train_envs, eval_envs, rollout_episodes, eval_episodes,
                             training_iterations, eval_from_eps_number, episode_length,
                             kill_signal, step_signal, num_eval_episodes_signal,
                             eval_epochs_signal, eval_report_signal, log_freq,
                             rollout_generator, save_load_lock, current_replay_ratio,
                             target_replay_ratio, weightsdir, logdir, env_device,
                             previous_loaded_weight_folder, num_eval_runs)

    def _load_save(self):
        if self._weightsdir is None:
            logging.info("'weightsdir' was None, so not loading weights.")
            return
        while True:
            weight_folders = []
            with self._save_load_lock:
                if os.path.exists(self._weightsdir):
                    weight_folders = os.listdir(self._weightsdir)
                if len(weight_folders) > 0:
                    weight_folders = sorted(map(int, weight_folders))
                    # only load if there has been a new weight saving
                    if self._previous_loaded_weight_folder != weight_folders[-1]:
                        self._previous_loaded_weight_folder = weight_folders[-1]
                        d = os.path.join(self._weightsdir, str(weight_folders[-1]))
                        try:
                            self._agent.load_weights(d)
                        except FileNotFoundError:
                            # rare case when agent hasn't finished writing.
                            time.sleep(1)
                            self._agent.load_weights(d)
                        logging.info('Agent %s: Loaded weights: %s' % (self._name, d))
                        self._new_weights = True
                    else:
                        self._new_weights = False
                    break
            logging.info('Waiting for weights to become available.')
            time.sleep(1)

    def _get_task_name(self):
        if hasattr(self._eval_env, '_task_class'):
            eval_task_name = change_case(self._eval_env._task_class.__name__)
            multi_task = False
        elif hasattr(self._eval_env, '_task_classes'):
            if self._eval_env.active_task_id != -1:
                task_id = (self._eval_env.active_task_id) % len(self._eval_env._task_classes)
                eval_task_name = change_case(self._eval_env._task_classes[task_id].__name__)
            else:
                eval_task_name = ''
            multi_task = True
        else:
            raise Exception('Neither task_class nor task_classes found in eval env')
        return eval_task_name, multi_task

    def _run_eval_independent(self, name: str,
                              stats_accumulator,
                              weight,
                              writer_lock,
                              eval=True,
                              device_idx=0,
                              save_metrics=True,
                              cinematic_recorder_cfg=None, train=False, rl_alg='REINFORCE', init_diff=False):

        self._name = name
        self._save_metrics = save_metrics
        self._is_test_set = type(weight) == dict

        self._agent = copy.deepcopy(self._agent)
        if rl_alg == 'PPO':
            self._agent_old = copy.deepcopy(self._agent)
            K = 10 #K epochs per rollout
            gamma = 0.95
            mini_batch_size = 16
            step = 0
            per_eps_upd_freq = 25
            anneal_lr = False
            norm_adv = False
            warmup_actor_lr = False
            decay_critic_lr = False
        elif rl_alg == 'DDPG':
            self._agent_old = copy.deepcopy(self._agent)
            for target_param, param in zip(self._agent_old.parameters(), self._agent.parameters()):
                target_param.data.copy_(param.data)
            self._agent_old.train()
            batch_size = 128
            tau = 0.01
            max_buffer_size = 2000
            from collections import deque
            buffer = deque(maxlen=max_buffer_size)
        else:
            self._agent_old = None

        if init_diff:
            init_arm_pose = []
        else:
            init_arm_pose = None
        device = torch.device('cuda:%d' % device_idx) if torch.cuda.device_count() > 1 else torch.device('cuda:0')
        #with writer_lock: # hack to prevent multiple CLIP downloads ... argh should use a separate lock
        #    self._agent.build(training=False, device=device)

        logging.info('%s: Launching env.' % name)
        np.random.seed()

        logging.info('Agent information:')
        logging.info(self._agent)

        env = self._eval_env
        env.eval = eval
        env.launch()

        # initialize cinematic recorder if specified
        rec_cfg = cinematic_recorder_cfg
        if rec_cfg.enabled:
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            cam = VisionSensor.create(rec_cfg.camera_resolution)
            cam.set_pose(cam_placeholder.get_pose())
            cam.set_parent(cam_placeholder)

            cam_motion = CircleCameraMotion(cam, Dummy('cam_cinematic_base'), rec_cfg.rotate_speed)
            tr = TaskRecorder(env, cam_motion, fps=rec_cfg.fps)

            env.env._action_mode.arm_action_mode.set_callable_each_step(tr.take_snap)

        if not os.path.exists(self._weightsdir):
            raise Exception('No weights directory found.')

        # to save or not to save evaluation metrics (set as False for recording videos)
        if self._save_metrics:
            csv_file = 'eval_data.csv' if not self._is_test_set else 'test_data.csv'
            writer = LogWriter(self._logdir+'/logs/'+rl_alg+'/', True, True,
                               env_csv=csv_file)

        # one weight for all tasks (used for validation)
        if type(weight) == int:
            logging.info('Evaluating weight %s' % weight)
            weight_path = os.path.join(self._weightsdir, str(weight))
            seed_path = self._weightsdir.replace('/weights', '')
            #self._agent.load_weights(weight_path)
            weight_name = str(weight)

        new_transitions = {'train_envs': 0, 'eval_envs': 0}
        total_transitions = {'train_envs': 0, 'eval_envs': 0}
        current_task_id = -1

        for n_eval in range(self._num_eval_runs):#3
            if rec_cfg.enabled:
                tr._cam_motion.save_pose()

            # best weight for each task (used for test evaluation)
            if type(weight) == dict:
                task_name = list(weight.keys())[n_eval]
                task_weight = weight[task_name]
                weight_path = os.path.join(self._weightsdir, str(task_weight))
                seed_path = self._weightsdir.replace('/weights', '')
                self._agent.load_weights(weight_path)
                weight_name = str(task_weight)
                print('Evaluating weight %s for %s' % (weight_name, task_name))

            # evaluate on N tasks * M episodes per task = total eval episodes
            for ep in range(self._eval_episodes):
                eval_demo_seed = ep + self._eval_from_eps_number + self._eval_episodes * n_eval
                logging.info('%s: Starting episode %d, seed %d.' % (name, ep + self._eval_from_eps_number + self._eval_episodes * n_eval, eval_demo_seed))
                if rl_alg == 'PPO' and eval_demo_seed%per_eps_upd_freq==0:
                    if eval_demo_seed > 0:
                        #returns = torch.tensor(returns).to('cuda')  # convert to tensor
                        #baseline_preds_old = torch.cat(baseline_preds_old, dim=-1).squeeze(0).squeeze(0).to('cuda')
                        #advantages.append(returns - baseline_preds_old.detach())
                        shuffler = list(zip(observations,log_probs_old, returns, baseline_preds_old, init_observations, init_behaviors, instructions, poses))         # shuffle the rollout
                        random.shuffle(shuffler)
                        observations, log_probs_old, returns, baseline_preds_old, init_observations, init_behaviors, instructions, poses = zip(*shuffler)
                        observations = list(observations)
                        init_behaviors = list(init_behaviors)
                        instructions = list(instructions)
                        init_observations = list(init_observations)
                        poses = list(poses)
                        log_probs_old = list(log_probs_old)
                        returns = list(returns)
                        baseline_preds_old = list(baseline_preds_old)
                        for k in range(K): # Optimise for K epochs
                            for i in range(0,len(log_probs_old),mini_batch_size):                                # minibatch size 2
                                log_probs = []
                                baseline_preds = []
                                if len(log_probs_old) % mini_batch_size == 0 or len(log_probs_old) > i + mini_batch_size:
                                    for j in range(mini_batch_size):
                                        #if norm_adv:
                                            #advantages[i+j] = (advantages[i+j]-advantages[i+j].mean())/(advantages[i+j].std()+1e-8)
                                        hid = self._agent.initialise_hidden(init_observations[i+j],
                                                                                    torch.Tensor(init_behaviors[i+j]).unsqueeze(
                                                                                        0).unsqueeze(0).to('cuda'), instructions[i+j], True)
                                        _, log_prob, baseline_pred = self._agent.act_with_exploration(observations[i+j], hid,poses[i+j], True, True, rl_alg)
                                        log_probs.append(log_prob)
                                        baseline_preds.append(baseline_pred)
                                    baseline_loss=self._agent.update_with_ppo(log_probs, log_probs_old[i:i+mini_batch_size], returns[i:i+mini_batch_size],
                                                                              baseline_preds, baseline_preds_old[i:i+mini_batch_size])
                                else:
                                    for j in range(len(log_probs_old)%mini_batch_size):
                                        #if norm_adv:
                                            #advantages[i+j] = (advantages[i+j]-advantages[i+j].mean())/(advantages[i+j].std()+1e-8)
                                        hid = self._agent.initialise_hidden(init_observations[i+j],
                                                                            torch.Tensor(init_behaviors[i+j]).unsqueeze(
                                                                                0).unsqueeze(0).to('cuda'),
                                                                            instructions[i+j], True)
                                        _, log_prob, baseline_pred = self._agent.act_with_exploration(observations[i+j],
                                                                                                      hid,
                                                                                                      poses[i+j], True,
                                                                                                      True, rl_alg)
                                        log_probs.append(log_prob)
                                        baseline_preds.append(baseline_pred)
                                    baseline_loss = self._agent.update_with_ppo(log_probs, log_probs_old[i:i + len(log_probs_old) % mini_batch_size],
                                                                                returns[i:i + len(log_probs_old) % mini_batch_size],
                                                                                baseline_preds, baseline_preds_old[i:i + len(log_probs_old) % mini_batch_size])
                                writer.add_scalar(eval_demo_seed, 'baseline loss', baseline_loss)
                        self._agent_old = copy.deepcopy(self._agent)
                        if anneal_lr:       # Learning rate annealing
                            update = eval_demo_seed//per_eps_upd_freq
                            num_updates = (self._num_eval_runs * self._eval_episodes) // per_eps_upd_freq
                            frac = 1.0 - (update - 1.0) / num_updates
                            lrnow = frac * 1e-5
                            self._agent.optimiser.param_groups[0]["lr"] = lrnow
                            print("Current learning rate:", lrnow)
                        if eval_demo_seed // per_eps_upd_freq <= 10:
                            if warmup_actor_lr:
                                current_lr = self._agent.optimiser.param_groups[0]["lr"]
                                new_lr = current_lr + 1.5e-5/(100/per_eps_upd_freq)
                                self._agent.optimiser.param_groups[0]["lr"] = new_lr
                            if decay_critic_lr:
                                current_lr = self._agent.optimiser.param_groups[1]["lr"]
                                new_lr = current_lr - 1.5e-5/(100/per_eps_upd_freq)
                                self._agent.optimiser.param_groups[1]["lr"] = new_lr
                    observations = []
                    log_probs_old = []
                    init_observations = []
                    init_behaviors= []
                    instructions = []
                    baseline_preds_old = []
                    poses = []
                    rws = []
                    returns = []
                    #advantages = []

                # the current task gets reset after every M episodes
                episode_rollout = []
                generator = self._rollout_generator.generator(
                    self._step_signal, env, self._agent, writer,
                    self._episode_length, self._timesteps,
                    eval, init_arm_pose, eval_demo_seed=eval_demo_seed,
                    record_enabled=rec_cfg.enabled, train=train, agent_old=self._agent_old, rl_alg=rl_alg)

                try:
                    for replay_transition in generator:
                        if init_diff:
                            [replay_transition, init_arm_pose] = replay_transition
                        if rl_alg == 'PPO' and train==True:
                            step = step + 1
                            [replay_transition, log_prob_old, rw, value_old, observation, init_obs, init_beh, instruction, pose] = replay_transition
                            init_observations.append(init_obs)
                            init_behaviors.append(init_beh)
                            instructions.append(instruction)
                            log_probs_old.append(log_prob_old)
                            observations.append(observation)
                            baseline_preds_old.append(value_old)
                            poses.append(pose)
                            rws.append(rw)
                        elif rl_alg == 'DDPG' and train==True:
                            replay_transition, experience = replay_transition
                            terminal = experience[-2]
                            if terminal:
                                rw = experience[1]
                                step = experience[-1]
                                if rw == 0.0 and step + 1 < self._episode_length:
                                    experience[1] = -0.01
                                elif rw == 100.0:
                                    experience[1] = 1.00
                                writer.add_scalar(eval_demo_seed, 'reward', experience[1])
                                if experience[1] == 1.00:
                                    writer.add_scalar(eval_demo_seed, 'positive reward', 100.0)
                                    writer.add_scalar(eval_demo_seed, 'null reward', 0.0)
                                    writer.add_scalar(eval_demo_seed, 'negative reward', 0.0)
                                elif experience[1] == 0.00:
                                    writer.add_scalar(eval_demo_seed, 'null reward', 100.0)
                                    writer.add_scalar(eval_demo_seed, 'positive reward', 0.0)
                                    writer.add_scalar(eval_demo_seed, 'negative reward', 0.0)
                                else:
                                    writer.add_scalar(eval_demo_seed, 'negative reward', 100.0)
                                    writer.add_scalar(eval_demo_seed, 'positive reward', 0.0)
                                    writer.add_scalar(eval_demo_seed, 'null reward', 0.0)
                            buffer.append(experience)
                            if len(buffer) > batch_size:
                                actions = []
                                rws = []
                                lang_goals = []
                                init_observations_frgb = []
                                init_observations_lsrgb = []
                                init_observations_rsrgb = []
                                init_observations_wrgb = []
                                observations_frgb = []
                                observations_rsrgb = []
                                observations_lsrgb = []
                                observations_wrgb = []
                                init_behaviors = []
                                behaviors = []
                                critic_target_outs = []
                                batch = random.sample(buffer, batch_size)  # shuffle the rollout
                                for experience in batch:
                                    action, reward, lang_goal, init_obs_frgb, init_obs_lsrgb, init_obs_rsrgb, init_obs_wrgb, \
                                    obs_frgb, obs_lsrgb, obs_rsrgb, obs_wrgb, init_behavior, behavior, critic_target_out, _, _ = experience
                                    actions.append(action)
                                    rws.append(reward)
                                    lang_goals.append(lang_goal)
                                    init_observations_frgb.append(init_obs_frgb)
                                    init_observations_lsrgb.append(init_obs_lsrgb)
                                    init_observations_rsrgb.append(init_obs_rsrgb)
                                    init_observations_wrgb.append(init_obs_wrgb)
                                    observations_frgb.append(obs_frgb)
                                    observations_lsrgb.append(obs_lsrgb)
                                    observations_rsrgb.append(obs_rsrgb)
                                    observations_wrgb.append(obs_wrgb)
                                    init_behaviors.append(init_behavior)
                                    behaviors.append(behavior)
                                    critic_target_outs.append(critic_target_out)
                                hids = self._agent.initialise_hidden_ddpg(np.stack(init_observations_frgb),
                                                                          np.stack(init_observations_rsrgb),
                                                                          np.stack(init_observations_lsrgb),
                                                                          np.stack(init_observations_wrgb),
                                                                          torch.from_numpy(np.stack(init_behaviors,
                                                                                                    axis=0)).unsqueeze(
                                                                              1).to('cuda'), lang_goals, True)
                                policy_grads, critic_outs = self._agent.calculate_pol_grads_and_q_values(
                                    torch.cat(actions), torch.cat(observations_frgb),
                                    torch.cat(observations_rsrgb), torch.cat(observations_lsrgb),
                                    torch.cat(observations_wrgb), hids, torch.cat(behaviors), True)
                                baseline_loss = self._agent.update_with_ddpg(policy_grads, rws, critic_outs,
                                                                             critic_target_outs)
                                writer.add_scalar(eval_demo_seed, 'baseline loss', baseline_loss)
                                for target_param, param in zip(self._agent_old.parameters(),
                                                               self._agent.parameters()):
                                    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
                        while True:
                            if self._kill_signal.value:
                                env.shutdown()
                                return
                            if (eval or self._target_replay_ratio is None or
                                    self._step_signal.value <= 0 or (
                                            self._current_replay_ratio.value >
                                            self._target_replay_ratio)):
                                break
                            time.sleep(1)
                            logging.debug(
                                'Agent. Waiting for replay_ratio %f to be more than %f' %
                                (self._current_replay_ratio.value, self._target_replay_ratio))

                        with self.write_lock:
                            if len(self.agent_summaries) == 0:
                                # Only store new summaries if the previous ones
                                # have been popped by the main env runner.
                                if type(self._agent) == list:
                                    for s in self._agent[0].act_summaries():
                                        self.agent_summaries.append(s)
                                else:
                                    for s in self._agent.act_summaries():
                                        self.agent_summaries.append(s)
                        episode_rollout.append(replay_transition)
                except StopIteration as e:
                    continue
                except Exception as e:
                    env.shutdown()
                    raise e

                if rl_alg == 'PPO':
                    R = 0
                    rewards = [0.0 for i in range(step-1)]
                    if rws[-1] == 0.0:# and step == self._episode_length:
                        rewards.append(0.0)
                    else:
                        rewards.append(1.0)
                    insert_index = len(returns)
                    for r in rewards[::-1]:
                        R = r + gamma * R
                        returns.insert(insert_index, R)
                    step = 0
                    writer.add_scalar(eval_demo_seed, 'reward', rewards[-1])
                    if rewards[-1] == 1.00:
                        writer.add_scalar(eval_demo_seed, 'positive reward', 100.0)
                        writer.add_scalar(eval_demo_seed, 'null reward', 0.0)
                        writer.add_scalar(eval_demo_seed, 'negative reward', 0.0)
                    elif rewards[-1] == 0.00:
                        writer.add_scalar(eval_demo_seed, 'null reward', 100.0)
                        writer.add_scalar(eval_demo_seed, 'positive reward', 0.0)
                        writer.add_scalar(eval_demo_seed, 'negative reward', 0.0)
                    else:
                        writer.add_scalar(eval_demo_seed, 'negative reward', 100.0)
                        writer.add_scalar(eval_demo_seed, 'positive reward', 0.0)
                        writer.add_scalar(eval_demo_seed, 'null reward', 0.0)

                with self.write_lock:
                    for transition in episode_rollout:

                        new_transitions['eval_envs'] += 1
                        total_transitions['eval_envs'] += 1
                        stats_accumulator.step(transition, eval)
                        current_task_id = transition.info['active_task_id']

                self._num_eval_episodes_signal.value += 1

                task_name, _ = self._get_task_name()
                reward = episode_rollout[-1].reward
                lang_goal = env._lang_goal
                print(f"Evaluating {task_name} | Episode {ep + self._eval_episodes * n_eval} | Score: {reward} | Lang Goal: {lang_goal}")

                # save recording
                if rec_cfg.enabled:
                    success = reward > 0.99
                    record_file = os.path.join(seed_path, 'videos',
                                               '%s_w%s_s%s_%s.mp4' % (task_name,
                                                                      weight_name,
                                                                      eval_demo_seed,
                                                                      'succ' if success else 'fail'))

                    lang_goal = self._eval_env._lang_goal

                    tr.save(record_file, lang_goal, reward)
                    tr._cam_motion.restore_pose()

                if train and ep + self._eval_episodes * n_eval>0 and (ep + self._eval_episodes * n_eval)%1000==0:#200==0
                    if type(self._agent) == list:
                        self._agent[0].save_model(ep + self._eval_episodes * n_eval, rl_alg)
                    else:
                        self._agent.save_model(ep + self._eval_episodes * n_eval, rl_alg)
                    print(f"Saved the weights after {ep + self._eval_episodes * n_eval} epochs")
                    num_of_episodes = self._eval_episodes
                    self._eval_episodes = 150#25#
                    root_dir = env._task._dataset_root.split('/')[-1]
                    env._task._dataset_root = env._task._dataset_root.replace(root_dir, 'test_pick_cup')#'test_slide_block')#'test_button_green')#
                    eval_stats_accumulator = SimpleAccumulator(eval_video_fps=30)
                    if type(self._agent) == list:
                        self._agent[0].eval()
                    else:
                        self._agent.eval()

                    for ep_eval in range(self._eval_episodes):
                        eval_demo_seed = ep_eval + self._eval_from_eps_number
                        logging.info('%s: Starting episode %d, seed %d.' % (name, ep_eval, eval_demo_seed))

                        # the current task gets reset after every M episodes
                        episode_rollout = []
                        generator = self._rollout_generator.generator(
                            self._step_signal, env, self._agent, writer,
                            self._episode_length, self._timesteps,
                            eval, eval_demo_seed=eval_demo_seed,
                            record_enabled=rec_cfg.enabled, train=False)
                        try:
                            for replay_transition in generator:
                                while True:
                                    if self._kill_signal.value:
                                        env.shutdown()
                                        return
                                    if (eval or self._target_replay_ratio is None or
                                            self._step_signal.value <= 0 or (
                                                    self._current_replay_ratio.value >
                                                    self._target_replay_ratio)):
                                        break
                                    time.sleep(1)
                                    logging.debug(
                                        'Agent. Waiting for replay_ratio %f to be more than %f' %
                                        (self._current_replay_ratio.value, self._target_replay_ratio))

                                with self.write_lock:
                                    if len(self.agent_summaries) == 0:
                                        # Only store new summaries if the previous ones
                                        # have been popped by the main env runner.
                                        if type(self._agent) == list:
                                            for s in self._agent[0].act_summaries():
                                                self.agent_summaries.append(s)
                                        else:
                                            for s in self._agent.act_summaries():
                                                self.agent_summaries.append(s)
                                episode_rollout.append(replay_transition)
                        except StopIteration as e:
                            continue
                        except Exception as e:
                            env.shutdown()
                            raise e

                        with self.write_lock:
                            for transition in episode_rollout:
                                eval_stats_accumulator.step(transition, eval)
                                current_task_id = transition.info['active_task_id']

                        task_name, _ = self._get_task_name()
                        reward = episode_rollout[-1].reward
                        lang_goal = env._lang_goal
                        print(f"Evaluating {task_name} | Episode {ep_eval} | Score: {reward} | Lang Goal: {lang_goal}")

                        # save recording
                        if rec_cfg.enabled:
                            success = reward > 0.99
                            record_file = os.path.join(seed_path, 'videos',
                                                       '%s_w%s_s%s_%s.mp4' % (task_name,
                                                                              weight_name,
                                                                              eval_demo_seed,
                                                                              'succ' if success else 'fail'))

                            lang_goal = self._eval_env._lang_goal

                            tr.save(record_file, lang_goal, reward)
                            tr._cam_motion.restore_pose()
                    summaries = []
                    summaries.extend(eval_stats_accumulator.pop())
                    eval_task_name, multi_task = self._get_task_name()
                    if eval_task_name and multi_task:
                        for s in summaries:
                            if 'eval' in s.name:
                                s.name = '%s/%s' % (s.name, eval_task_name)

                    if len(summaries) > 0:
                        if multi_task:
                            success = [s.value for s in summaries if f'eval_envs/return/{eval_task_name}' in s.name][
                                0]
                        else:
                            success = [s.value for s in summaries if f'eval_envs/return' in s.name][0]
                    else:
                        success = "unknown"
                    writer.add_scalar(ep + self._eval_episodes * n_eval, 'test success rate', success)
                    env._task._dataset_root = env._task._dataset_root.replace('test_slide_block', root_dir)#env._task._dataset_root.replace('test_button_green', root_dir)#
                    self._eval_episodes = num_of_episodes
                    if type(self._agent) == list:
                        self._agent[0].train()
                    else:
                        self._agent.train()
                    summaries = []
                    eval_stats_accumulator = []

            # report summaries
            summaries = []
            summaries.extend(stats_accumulator.pop())

            eval_task_name, multi_task = self._get_task_name()

            if eval_task_name and multi_task:
                for s in summaries:
                    if 'eval' in s.name:
                        s.name = '%s/%s' % (s.name, eval_task_name)

            if len(summaries) > 0:
                if multi_task:
                    task_score = [s.value for s in summaries if f'eval_envs/return/{eval_task_name}' in s.name][0]
                else:
                    task_score = [s.value for s in summaries if f'eval_envs/return' in s.name][0]
            else:
                task_score = "unknown"

            print(f"Finished {eval_task_name} | Final Score: {task_score}\n")
            if train:
                if type(self._agent) == list:
                    self._agent[0].save_model((n_eval+1)*self._eval_episodes, rl_alg)
                else:
                    self._agent.save_model((n_eval + 1) * self._eval_episodes, rl_alg)
                print(f"Saved the weights after {(n_eval+1)*self._eval_episodes} epochs of {eval_task_name}")
            if self._save_metrics:
                with writer_lock:
                    writer.add_summaries(weight_name, summaries)

            self._new_transitions = {'train_envs': 0, 'eval_envs': 0}
            self.agent_summaries[:] = []
            self.stored_transitions[:] = []

        if self._save_metrics:
            with writer_lock:
                writer.end_iteration()

        logging.info('Finished evaluation.')
        env.shutdown()

    def kill(self):
        self._kill_signal.value = True
