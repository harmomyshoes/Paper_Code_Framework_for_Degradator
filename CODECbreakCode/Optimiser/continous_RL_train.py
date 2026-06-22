from datetime import datetime
import sys
import tensorflow as tf
cpus = tf.config.experimental.list_physical_devices('CPU') 
tf.config.experimental.set_visible_devices(cpus[0], 'CPU')
import tf_agents


#import driver
from tf_agents.drivers import py_driver
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse

#import environment
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment #allows parallel computing for generating experiences
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment

#import replay buffer
from tf_agents import replay_buffers as rb

#import agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.utils import value_ops
from tf_agents.trajectories import StepType,time_step


#other used packages
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from matplotlib.axes import Axes as ax
import numpy as np
import pandas as pd
import os,gc
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


from tf_agents.networks import actor_distribution_network
from tf_agents.specs import array_spec
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
from CODECbreakCode.Optimiser.config import get_config
from CODECbreakCode.Optimiser.config import denormalize_action
from CODECbreakCode.Optimiser.env import Env_Continue as Env
from CODECbreakCode.Optimiser.custom_normal_projection_network import NormalProjectionNetwork
from typing import Callable
from functools import partial

#To limit TensorFlow to a specific set of GPUs


#enable multiprocessing for parallel computing
tf_agents.system.multiprocessing.enable_interactive_mode()
gc.collect()

cfg = get_config()
train_cfg = cfg["training"]

class continous_RL_train: 
    def __init__(self,sub_episode_length=0, sub_episode_num_single_batch=0, env_num=0):

        if env_num == 0:
            self._env_num = train_cfg["env_num"]
        else:
            self._env_num = env_num
        if sub_episode_length == 0:
            self._sub_episode_length = train_cfg["sub_episode_length"] #number of time_steps in a sub-episode
        else:
            self._sub_episode_length = sub_episode_length

        if sub_episode_num_single_batch == 0:
            self._sub_episode_num_single_batch = train_cfg["sub_episode_num_single_batch"] #number of sub-episodes in each episodes(trajectories)
        else:
            self._sub_episode_num_single_batch = sub_episode_num_single_batch

        self._episode_length = int(self._sub_episode_length * self._sub_episode_num_single_batch) #number of time_steps in an single env of the trajectory
        self._sub_episode_num = int(self._env_num* self._sub_episode_num_single_batch) #number of all sub-episodes in the parallel environments
        # print(f"In current setting, each epoch includes env number :{self._env_num}, {self._sub_episode_num_single_batch} episodes "
        #       f"in each env(trajectory), {self._sub_episode_length} is the time_steps in each episode, "
        #       f"Therefore, total {self._episode_length} time_steps in each trajectory per env.")
        print("number of sub_episodes used for a single param update:", self._sub_episode_num)

        self._train_env = None
        self._eval_env = None
#        self._REINFORCE_agent = None
        self._train_step_num = 0 #number of training steps
        self._final_reward = train_cfg["final_reward"] #the best reward found so far
        self._final_solution = cfg["env"]["x0_reinforce"].copy() #inital solution
        self._reward_list = [] #the best reward found so far
        self._solution_list = [] #the best solution found so far
        self._REINFORCE_logs = [] #for logging the best objective value of the best solution among all the solutions used for one update of theta
        self._metrics_history = [] #NEW: for tracking RL metrics per update (avg_reward, std, entropy, lr, grad_norm, etc.)
        self._project_scores_history = [] #NEW: for tracking per-project scores if using multi-project reward
        #Learning Schedule = initial_lr * (C/(step+C))
        # Adam optimizer is better than SGD for RL - handles adaptive learning rates per parameter
        self._opt = tf.keras.optimizers.legacy.Adam(
            learning_rate=lr_schedule(initial_lr=0.007, C=train_cfg["lr_half_decay_steps"]),
            beta_1=0.9,     # Default momentum
            beta_2=0.999,   # Default RMSprop-like decay
            epsilon=1e-7    # Numerical stability
        )
        self._state_dim = cfg["env"]["state_dim"] #dimension of the state space

    def train(self, update_num=0, eval_intv=0, collect_data=True, save_every=100):
        """
        Main function for training the agent.
        It sets up the environment, agent, and runs the training loop.

        
        """
        
        if collect_data:
            collector = DataCollector(save_dir='rl_analysis_data')

        self.set_agent()
        #################
        #replay_buffer is used to store policy exploration data
        #################
        replay_buffer = rb.TFUniformReplayBuffer(
                        data_spec = self._REINFORCE_agent.collect_data_spec,  # describe spec for a single iterm in the buffer. A TensorSpec or a list/tuple/nest of TensorSpecs describing a single item that can be stored in this buffer.
                        batch_size = self._env_num,    # number of parallel worlds, where in each world there is an agent generating trajectories
                                                # One batch corresponds to one parallel environment
                        max_length = self._episode_length*5    # The maximum number of items that can be stored in a single batch segment of the buffer.     
                                                            # if exceeding this number previous trajectories will be dropped
        )



        #A driver uses an agent to perform its policy in the environment.
        #The trajectory is saved in replay_buffer
        collect_driver = DynamicEpisodeDriver(
                                                env = self._train_env, #train_env contains parallel environments (no.: env_num)
                                                policy = self._REINFORCE_agent.collect_policy,
                                                observers = [replay_buffer.add_batch],
                                                num_episodes = self._sub_episode_num   #SUM_i (number of episodes to be performed in the ith parallel environment)
                                            )




        ####Examine on the update number and interval
        if update_num == 0:
            update_num = train_cfg["generation_num"]

        if eval_intv == 0:
            eval_intv = train_cfg["eval_every"]
        

        print(f"update_num: {update_num}, eval_intv: {eval_intv}")
        # tf.random.set_seed(0)


        for n in range(0,update_num):
            #Generate Trajectories
            replay_buffer.clear()
            collect_driver.run()  #a batch of trajectories will be saved in replay_buffer
            
            experience = replay_buffer.gather_all() #get the batch of trajectories, shape=(batch_size, episode_length)
            rewards = self.extract_episode(traj_batch=experience,epi_length=self._sub_episode_length,attr_name = 'reward') #shape=(sub_episode_num, sub_episode_length)
            observations = self.extract_episode(traj_batch=experience,epi_length=self._sub_episode_length,attr_name = 'observation') #shape=(sub_episode_num, sub_episode_length, state_dim)
            actions = self.extract_episode(traj_batch=experience,epi_length=self._sub_episode_length,attr_name = 'action') #shape=(sub_episode_num, sub_episode_length, state_dim)
            step_types = self.extract_episode(traj_batch=experience,epi_length=self._sub_episode_length,attr_name = 'step_type')
            discounts = self.extract_episode(traj_batch=experience,epi_length=self._sub_episode_length,attr_name = 'discount')

            # Collect data for analysis
            if collect_data:
                collector.add(actions, rewards)
                if (n + 1) % save_every == 0:
                    collector.save(suffix=f'update_{n+1}')
                    print(f"Saved data at update {n+1}/{update_num}")

            time_steps = ts.TimeStep(step_types,
                                    tf.zeros_like(rewards),
                                    tf.zeros_like(discounts),
                                    observations
                                    )
            rewards_sum = tf.reduce_sum(rewards, axis=1) #shape=(sub_episode_num,) 
         

            print(f"\n=== STEP {n}/{update_num-1} ===", flush=True)
            sys.stdout.flush()
            
            with tf.GradientTape() as tape:
                #trainable parameters in the actor_network in REINFORCE_agent
                variables_to_train = self._REINFORCE_agent._actor_network.trainable_weights


                ###########Compute J_loss = -J
                actions_distribution = self._REINFORCE_agent.collect_policy.distribution(
                                    time_steps, policy_state=None).action
            
                
                ######Entropy trace during “burn-in”, Watch the policy’s entropy during the very first few updates
                ent = actions_distribution.entropy().numpy()   # shape = (batch_size, state_dim)
                print(f"[step {n:3d}] mean entropy per-dim = {ent.mean():.4f}")
                #######                
                ####debugging info for std and mean
                print("Std variables being trained:", [v.name for v in variables_to_train if 'std' in v.name])
                print("Total trainable variables:", len(variables_to_train))
                ######
                for v in variables_to_train:
                    print(f"  {v.name}: trainable={v.trainable}, shape={v.shape}")

            
                #log(pi(action|state)), shape = (batch_size, epsode_length)
                action_log_prob = common.log_probability(actions_distribution, 
                                                        actions,
                                                        self._REINFORCE_agent.action_spec)

                # J = tf.reduce_sum(tf.reduce_sum(action_log_prob,axis=1)*rewards_sum)/self._sub_episode_num
                logp_episode = tf.reduce_sum(action_log_prob, axis=1)      # [E]
                returns = rewards_sum                                      # [E]
                adv = returns - tf.reduce_mean(returns)                     # [E]
                J = tf.reduce_mean(logp_episode * tf.stop_gradient(adv))   # scalar

                ###########Compute regularization loss from actor_net params
                regu_term = tf.reduce_sum(variables_to_train[0]**2)
                num = len(variables_to_train) #number of vectors in variables_to_train
                for i in range(1,num):
                    regu_term += tf.reduce_sum(variables_to_train[i]**2)
                
                # total = -J + train_cfg["param_alpha"]*regu_term
                # total = -J + train_cfg["param_alpha"]*regu_term + 0.01*ent.mean()
                # Improved loss: stronger entropy bonus (0.1 instead of 0.05) for better exploration
                # Reduced regularization (0.00005 instead of 0.0001) to allow more flexibility
                total = -J + 0.00005 * regu_term - 0.1 * tf.reduce_mean(ent) 

            if collect_data:
                    collector.save(suffix='final')
                    print("Training complete! Data saved for analysis.")  
            
            #update parameters in the actor_network in the policy
            grads = tape.gradient(total, variables_to_train)
            
            # Gradient clipping for stability (prevents exploding gradients)
            grads, global_norm = tf.clip_by_global_norm(grads, clip_norm=1.0)
            
            grads_and_vars = list(zip(grads, variables_to_train))
            self._opt.apply_gradients(grads_and_vars=grads_and_vars)
            self._train_step_num += 1
            
            # Enhanced gradient analysis
            std_grads = [g for g, v in zip(grads, variables_to_train) if 'stddev' in v.name]
            mean_grads = [g for g, v in zip(grads, variables_to_train) if 'means' in v.name]
            
            if std_grads and mean_grads:
                std_norm = tf.sqrt(sum([tf.reduce_sum(g**2) for g in std_grads]))
                mean_norm = tf.sqrt(sum([tf.reduce_sum(g**2) for g in mean_grads]))
                ratio = std_norm / (mean_norm + 1e-8)
                
                print(f"Raw Std grad norm: {std_norm:.2f}")
                print(f"Raw Mean grad norm: {mean_norm:.2f}")
                print(f"Ratio: {ratio:.1f}x")
                print(f"Global grad norm (clipped): {global_norm:.2f}")

            ###collecting the key information for the training process
            batch_rewards = rewards.numpy()
            batch_obs = observations.numpy()
            
            batch_rewards[:,-1] = -np.power(10,8) #The initial reward is set as 0, we set it as this value to not affect the best_obs_index 
            best_step_reward = np.max(batch_rewards)
            best_step_index = [int(batch_rewards.argmax()/self._sub_episode_length),batch_rewards.argmax()%self._sub_episode_length+1]
            best_step = batch_obs[best_step_index[0],best_step_index[1],:] #best solution
            ###The observation structure is [sub_episode_num, sub_episode_length, state_dim]
            #best_step_reward = f(best_solution)
            avg_step_reward = np.mean(batch_rewards[:,0:-1])
            self._reward_list.append(best_step_reward)
            self._solution_list.append(denormalize_action(best_step))
            
            # NEW: Collect RL metrics for this update
            rewards_flat = batch_rewards[:,0:-1].flatten()
            actions_flat = actions.numpy().reshape(-1, self._state_dim)
            current_lr = self._opt.learning_rate(self._train_step_num)
            
            metrics = {
                'update': n,
                'avg_reward': float(np.mean(rewards_flat)),
                'std_reward': float(np.std(rewards_flat)),
                'min_reward': float(np.min(rewards_flat)),
                'max_reward': float(np.max(rewards_flat)),
                'median_reward': float(np.median(rewards_flat)),
                'best_reward_so_far': float(self._final_reward) if self._final_reward is not None else float(best_step_reward),
                'policy_entropy': float(ent.mean()),
                'learning_rate': float(current_lr.numpy() if hasattr(current_lr, 'numpy') else current_lr),
                'grad_norm': float(global_norm.numpy() if hasattr(global_norm, 'numpy') else global_norm),
            }
            
            # Add action statistics per dimension
            for dim in range(self._state_dim):
                metrics[f'action_mean_dim{dim}'] = float(np.mean(actions_flat[:, dim]))
                metrics[f'action_std_dim{dim}'] = float(np.std(actions_flat[:, dim]))
            
            self._metrics_history.append(metrics)

            for ep_idx in range(self._sub_episode_num):
                ep_reward   = batch_rewards[ep_idx,0:-1] #get the reward of the sub-episode, excluding the last time step
                # pick final-step observation as the “solution” for that episode:
                ep_solution = denormalize_action(batch_obs[ep_idx, 1:, :])       
                
                self._REINFORCE_logs.append({
                    'update':   n,           # which generation
                    'episode':  ep_idx,      # which sub-episode
                    'reward':   ep_reward,
                    'solution': ep_solution, # numpy array of shape (state_dim,)
                })


            ###collecting the best solution and reward
            if self._final_reward is None or best_step_reward>self._final_reward: 
                print("final reward before udpate:",self._final_reward)
                self._final_reward = best_step_reward
                self._final_solution = best_step
                print("final reward after udpate:",self._final_reward)
                print('updated final_solution=', denormalize_action(self._final_solution))

            #print(compute_reward(best_obs,alpha))
            if n%eval_intv==0:
                print("train_step no.=",self._train_step_num)
                print('best_solution of this generation=', denormalize_action(best_step))
                print('best step reward=',best_step_reward)

                print(f"Best solution changed: {denormalize_action(best_step) != denormalize_action(self._final_solution)}")
                print(f"Reward improvement: {best_step_reward - self._final_reward:.6f}")
                
                print('avg step reward=', avg_step_reward)
                print('final_solution=', denormalize_action(self._final_solution), 'final_reward=', self._final_reward)
                #print('episode of rewards', rewards.round(3))
                
                print('act_std:', actions_distribution.stddev()[0,0].numpy())
                print('act_mean:', actions_distribution.mean()[0,0].numpy()) #second action mean

                ####print out the observations
                flat_obs = tf.reshape(batch_obs, [-1, self._state_dim])
                obs_mean = tf.reduce_mean(flat_obs, axis=0)
                print('obs_mean:', obs_mean)
                obs_std  = tf.sqrt(tf.reduce_mean((flat_obs - obs_mean)**2, axis=0))
                print('obs_std:', obs_std)

                print('best_step_index:',best_step_index)
                print(' ')

        print('final_solution=',denormalize_action(self._final_solution),'final_reward=',self._final_reward)
#        self._REINFORCE_logs = [max(self._REINFORCE_logs[0:i]) for i in range(1,update_num+1)] #rolling max  
    def set_environments(self, reward_fn: Callable[[np.ndarray, bool], float]):
        make_env = partial(Env,reward_fn = reward_fn, sub_episode_length=self._sub_episode_length)
        print('train_env.batch_size = parallel environment number = ', self._env_num)        
        if self._env_num > 1:
            parallel_env = ParallelPyEnvironment(env_constructors=[make_env]*self._env_num, 
                                     start_serially=False,
                                     blocking=False,
                                     flatten=False
                                    )
#Use the wrapper to create a TFEnvironments obj. (so that parallel computation is enabled)
            self._train_env = tf_py_environment.TFPyEnvironment(parallel_env, check_dims=True) 
            #instance of parallel environments
        else:
            self._train_env = tf_py_environment.TFPyEnvironment(make_env(), check_dims=True)


        

    def set_agent(self):
        actor_net = actor_distribution_network.ActorDistributionNetwork(   
                                                self._train_env.observation_spec(),
                                                self._train_env.action_spec(),
                                                fc_layer_params=(256,256), #2 Hidden layers (was incorrectly (256,256,256,256))
                                                seed=0, #seed used for Keras kernal initializers for NormalProjectionNetwork.
                                                #discrete_projection_net=_categorical_projection_net
                                                activation_fn = tf.math.tanh,
                                                #continuous_projection_net=(NormalProjectionNetwork),
                                                continuous_projection_net=lambda spec: NormalProjectionNetwork(
                                                    spec,
                                                    state_dependent_std=True     # <— HERE!
                                                    )
                                                    # <-- makes std a function of the state, adding 
                                                # in time of 17/7, in this way to let the prediction of seperate in different dimensions
                                                )

        train_step_counter = tf.Variable(0)

        self._REINFORCE_agent = reinforce_agent.ReinforceAgent(
                    time_step_spec = self._train_env.time_step_spec(),
                    action_spec = self._train_env.action_spec(),
                    actor_network = actor_net,
                    value_network = None,
                    value_estimation_loss_coef = 0.2,
                    optimizer = self._opt,
                    advantage_fn = None,
                    use_advantage_loss = False,
                    gamma = 1.0, #discount factor for future returns
                    normalize_returns = True, #The instruction says it's better to normalize
                    # gradient_clipping = 0.5,
                    # entropy_regularization = 0.01,
                    train_step_counter = train_step_counter
                    )
        actor_net.trainable = True  # Ensure entire network is trainable
        for layer in actor_net.layers:
            layer.trainable = True
        
        self._REINFORCE_agent.initialize()

    #Functions needed for training
    def extract_episode(self,traj_batch,epi_length,attr_name = 'observation'):
        """
        This function extract episodes (each episode consists of consecutive time_steps) from a batch of trajectories.
        Inputs.
        -----------
        traj_batch:replay_buffer.gather_all(), a batch of trajectories
        epi_length:int, number of time_steps in each extracted episode
        attr_name:str, specify which data from traj_batch to extract
        
        Outputs.
        -----------
        tf.constant(new_attr,dtype=attr.dtype), shape = [new_batch_size, epi_length, state_dim]
                                            or shape = [new_batch_size, epi_length]
        """
        attr = getattr(traj_batch,attr_name)
        original_batch_dim = attr.shape[0]
        traj_length = attr.shape[1]
        epi_num = int(traj_length/epi_length) #number of episodes out of each trajectory
        batch_dim = int(original_batch_dim*epi_num) #new batch_dim
        
        if len(attr.shape)==3:
            stat_dim = attr.shape[2]
            new_attr = np.zeros([batch_dim, epi_length, stat_dim])
        else:
            new_attr = np.zeros([batch_dim, epi_length])
            
        for i in range(original_batch_dim):
            for j in range(epi_num):
                new_attr[i*epi_num+j] = attr[i,j*epi_length:(j+1)*epi_length].numpy()
            
        return tf.constant(new_attr,dtype=attr.dtype)

    def _save_metrics(self, data_fold_path, timestamp, filename_suffix=""):
        """
        Helper method to save RL metrics to CSV.
        
        Args:
            data_fold_path: Directory to save metrics
            timestamp: Timestamp string for filename
            filename_suffix: Optional suffix for filename
            
        Returns:
            DataFrame of metrics or None if no metrics collected
        """
        if not self._metrics_history:
            return None
            
        # Build filename
        if filename_suffix:
            metrics_file = f'RL_Metrics_{timestamp}_{filename_suffix}.csv'
        else:
            metrics_file = f'RL_Metrics_{timestamp}.csv'
        
        metrics_path = os.path.join(data_fold_path, metrics_file)
        RL_Metrics = pd.DataFrame(self._metrics_history)
        RL_Metrics.to_csv(metrics_path, index=False)
        
        print(f"✓ Saved RL metrics: {metrics_path}")
        print(f"  Metrics tracked: {len(RL_Metrics.columns)} columns, {len(RL_Metrics)} updates")
        
        return RL_Metrics
    
    def _save_project_scores(self, data_fold_path, timestamp, filename_suffix=""):
        """
        Helper method to save per-project scores to CSV.
        
        Args:
            data_fold_path: Directory to save scores
            timestamp: Timestamp string for filename
            filename_suffix: Optional suffix for filename
            
        Returns:
            DataFrame of project scores or None if no scores collected
        """
        if not self._project_scores_history:
            return None
            
        # Build filename
        if filename_suffix:
            scores_file = f'ProjectScores_{timestamp}_{filename_suffix}.csv'
        else:
            scores_file = f'ProjectScores_{timestamp}.csv'
        
        scores_path = os.path.join(data_fold_path, scores_file)
        ProjectScores = pd.DataFrame(self._project_scores_history)
        ProjectScores.to_csv(scores_path, index=False)
        
        print(f"✓ Saved project scores: {scores_path}")
        print(f"  Projects tracked: {len([c for c in ProjectScores.columns if c.startswith('reward_')])}")
        
        return ProjectScores
    
    def log_project_scores(self, update_num, episode_num, timestep, aggregated_reward, project_scores_dict):
        """
        Log per-project scores during training.
        Call this from your reward function if you want to track individual project scores.
        
        Args:
            update_num: Current update number
            episode_num: Current episode number  
            timestep: Current timestep
            aggregated_reward: The aggregated reward value
            project_scores_dict: Dictionary of {project_name: score}
            
        Example:
            # In your reward function:
            scores = {'Reggae': 0.234, 'Rock': 0.245, ...}
            if hasattr(trainer, 'log_project_scores'):
                trainer.log_project_scores(n, ep, t, final_score, scores)
        """
        record = {
            'update': update_num,
            'episode': episode_num,
            'timestep': timestep,
            'aggregated_reward': aggregated_reward,
        }
        # Add individual project scores
        for project_name, score in project_scores_dict.items():
            record[f'reward_{project_name}'] = score
        
        self._project_scores_history.append(record)

    def save_best_results(self, filefold, para_columns=[], filename_suffix=""):
            """
            Save the best results (one row per update).
            Also saves RL metrics and project scores if collected.
            
            Args:
                filefold: Base folder for saving results
                para_columns: Column names for parameters (default: dim_0, dim_1, ...)
                filename_suffix: Optional suffix for filenames (e.g., 'experiment1')
                
            Returns:
                Tuple of (best_results_df, metrics_df, project_scores_df)
            """
            if filefold is None:
                raise ValueError("filefold cannot be None")
            
            data_fold_path = os.path.join(filefold, 'Data')
            if not os.path.exists(data_fold_path): 
                os.makedirs(data_fold_path)

            dims = cfg["env"]["state_dim"]
            if not para_columns:
                para_columns = [f'dim_{i}' for i in range(dims)]
            
            timestamp = datetime.now().strftime("%Y%m%d%H%M")

            # ================================================================
            # Save Best Results (with update numbers)
            # ================================================================
            score_df = pd.DataFrame(self._reward_list, columns=['score'])
            manip_df = pd.DataFrame(self._solution_list, columns=para_columns)
            RL_Data = pd.concat([score_df, manip_df], axis=1)
            
            # Add update column at the beginning
            RL_Data.insert(0, 'update', range(len(RL_Data)))
            
            # Build filename
            if filename_suffix:
                best_file = f'RL_Data_BestResults_{timestamp}_{filename_suffix}.csv'
            else:
                best_file = f'RL_Data_BestResults_{timestamp}.csv'
            
            data_file_path = os.path.join(data_fold_path, best_file)
            RL_Data.to_csv(data_file_path, index=False)
            print(f"✓ Saved best results: {data_file_path}")

            # Save metrics and project scores
            RL_Metrics = self._save_metrics(data_fold_path, timestamp, filename_suffix)
            ProjectScores = self._save_project_scores(data_fold_path, timestamp, filename_suffix)
                        
            return RL_Data, RL_Metrics, ProjectScores
    
    def save_full_results(self, filefold, para_columns=[], filename_suffix=""):
            """
            Save the full trajectory results (all timesteps of all episodes).
            
            Args:
                filefold: Base folder for saving results
                para_columns: Column names for parameters (default: dim_0, dim_1, ...)
                filename_suffix: Optional suffix for filenames (e.g., 'experiment1')
                
            Returns:
                DataFrame of full results or None if no logs collected
            """
            if filefold is None:
                raise ValueError("filefold cannot be None")
            
            data_fold_path = os.path.join(filefold, 'Data')
            if not os.path.exists(data_fold_path): 
                os.makedirs(data_fold_path)

            dims = cfg["env"]["state_dim"]
            if not para_columns:
                para_columns = [f'dim_{i}' for i in range(dims)]
            
            timestamp = datetime.now().strftime("%Y%m%d%H%M")

            # ================================================================
            # Save Full Results
            # ================================================================
            if self._REINFORCE_logs is None or len(self._REINFORCE_logs) == 0:
                print("⚠ No full trajectory data to save")
                return None
            
            self._REINFORCE_logs = np.array(self._REINFORCE_logs)
            
            # Build filename
            if filename_suffix:
                full_file = f'RL_Data_FullResults_{timestamp}_{filename_suffix}.csv'
            else:
                full_file = f'RL_Data_FullResults_{timestamp}.csv'
            
            full_path = os.path.join(data_fold_path, full_file)

            rows = []
            for rec in self._REINFORCE_logs:
                for t, r in enumerate(rec['reward']):
                    sol = rec['solution'][t]
                    row = {
                        'update': rec['update'],
                        'episode': rec['episode'],
                        'timestep': t,
                        'reward': r,
                    }
                    for label, val in zip(para_columns, sol):
                        row[label] = val

                    rows.append(row)
                    
            RL_Data_Full = pd.DataFrame(rows)
            RL_Data_Full.to_csv(full_path, index=False)
            print(f"✓ Saved full results: {full_path}")
            
            return RL_Data_Full
    
    def save_results(self, filefold, para_columns=[], is_outputfulldata=False, filename_suffix="_WithRL"):
            """
            Convenience method: Save best results and optionally full results.
            Calls save_best_results() and save_full_results() internally.
            
            Args:
                filefold: Base folder for saving results
                para_columns: Column names for parameters
                is_outputfulldata: Whether to also save full trajectory data
                filename_suffix: Optional suffix for filenames
                
            Returns:
                Tuple of (best_results_df, full_results_df, metrics_df, project_scores_df)
            """
            # Save best results (always)
            RL_Data, RL_Metrics, ProjectScores = self.save_best_results(
                filefold, para_columns, filename_suffix
            )
            
            # Save full results (if requested)
            RL_Data_Full = None
            if is_outputfulldata:
                RL_Data_Full = self.save_full_results(filefold, para_columns, filename_suffix)
                        
            return RL_Data, RL_Data_Full, RL_Metrics, ProjectScores
            
 


class lr_schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, C):
        self.initial_learning_rate = initial_lr
        self.C = C
    def __call__(self, step):
        return self.initial_learning_rate*self.C/(self.C+step)
    
class DataCollector:
    """Simple data collector for RL parameter analysis"""
    def __init__(self, save_dir='rl_analysis_data'):
        self.save_dir = save_dir
        self.actions = []
        self.rewards = []
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"Data collector initialized. Saving to: {save_dir}")
        
    def add(self, actions, rewards):
        """Add batch of actions and rewards"""
        # Convert TF tensors to numpy
        if hasattr(actions, 'numpy'):
            actions = actions.numpy()
        if hasattr(rewards, 'numpy'):
            rewards = rewards.numpy()
        # Flatten if multi-dimensional
        if len(actions.shape) > 2:
            actions = actions.reshape(-1, actions.shape[-1])
        if len(rewards.shape) > 1:
            rewards = rewards.reshape(-1)
        self.actions.append(actions)
        self.rewards.append(rewards)
        
    def save(self, suffix=''):
        """Save collected data to NPY files"""
        if len(self.actions) == 0:
            print("No data to save!")
            return None, None
        
        from datetime import datetime
        actions = np.vstack(self.actions)
        rewards = np.concatenate(self.rewards)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if suffix:
            actions_file = os.path.join(self.save_dir, f'actions_{timestamp}_{suffix}.npy')
            rewards_file = os.path.join(self.save_dir, f'rewards_{timestamp}_{suffix}.npy')
        else:
            actions_file = os.path.join(self.save_dir, f'actions_{timestamp}.npy')
            rewards_file = os.path.join(self.save_dir, f'rewards_{timestamp}.npy')
        
        np.save(actions_file, actions)
        np.save(rewards_file, rewards)
        
        print(f"\n{'='*80}")
        print(f"SAVED DATA FOR ANALYSIS!")
        print(f"{'='*80}")
        print(f"Actions: {actions_file}")
        print(f"  Shape: {actions.shape}")
        print(f"Rewards: {rewards_file}")
        print(f"  Shape: {rewards.shape}")
        print(f"  Mean: {rewards.mean():.3f} ± {rewards.std():.3f}")
        print(f"{'='*80}\n")
        
        return actions_file, rewards_file