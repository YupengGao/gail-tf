import tensorflow as tf
import gailtf.baselines.common.tf_util as U
from gailtf.baselines import logger
from tqdm import tqdm
from gailtf.baselines.common.mpi_adam import MpiAdam
import tempfile, os
from gailtf.common.statistics import stats
import ipdb

def evaluate(env, policy_func, load_model_path, stochastic_policy=False, number_trajs=10):
  from gailtf.algo.trpo_mpi import traj_episode_generator
  ob_space = env.observation_space
  ac_space = env.action_space
  pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
  # placeholder
  ob = U.get_placeholder_cached(name="ob")
  ac = pi.pdtype.sample_placeholder([None])
  stochastic = U.get_placeholder_cached(name="stochastic")
  ep_gen = traj_episode_generator(pi, env, 1024, stochastic=stochastic_policy)
  U.load_state(load_model_path)
  len_list = []
  ret_list = []
  for _ in tqdm(range(number_trajs)):
    traj = ep_gen.__next__()
    ep_len, ep_ret = traj['ep_len'], traj['ep_ret']
    len_list.append(ep_len)
    ret_list.append(ep_ret)
  if stochastic_policy:
    print ('stochastic policy:')
  else:
    print ('deterministic policy:' )
  print ("Average length:", sum(len_list)/len(len_list))
  print ("Average return:", sum(ret_list)/len(ret_list))

def learn(env, policy_func, dataset, pretrained, optim_batch_size=128, max_iters=1e4,
           adam_epsilon=1e-5, optim_stepsize=3e-4, ckpt_dir=None, log_dir=None, task_name=None):
  val_per_iter = int(max_iters/10)
  ob_space = env.observation_space
  ac_space = env.action_space
  #pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
  pi_high = policy_func("pi_high", ob_space, ac_space.spaces[0])
  pi_low = policy_func("pi_low", ob_space, ac_space.spaces[1])

  # placeholder
  ob = U.get_placeholder_cached(name="ob")
  ac_high = pi_high.pdtype.sample_placeholder([None])
  stochastic_high = U.get_placeholder_cached(name="stochastic")
  loss_high = tf.reduce_mean(tf.square(ac_high-pi_high.ac))
  var_list_high = pi_high.get_trainable_variables()
  adam_high = MpiAdam(var_list_high, epsilon=adam_epsilon)
  lossandgrad_high = U.function([ob, ac_high, stochastic_high], [loss_high]+[U.flatgrad(loss_high, var_list_high)])

  ac_low = pi_low.pdtype.sample_placeholder([None])
  stochastic_low = U.get_placeholder_cached(name="stochastic")
  loss_low = tf.reduce_mean(tf.square(ac_low - pi_low.ac))
  var_list_low = pi_low.get_trainable_variables()
  adam_low = MpiAdam(var_list_low, epsilon=adam_epsilon)
  lossandgrad_low = U.function([ob, ac_low, stochastic_low], [loss_low] + [U.flatgrad(loss_low, var_list_low)])

  if not pretrained:
    writer = U.FileWriter(log_dir)
    ep_stats = stats(["Loss"])
  U.initialize()
  adam_high.sync()
  logger.log("Pretraining with Behavior Cloning...")
  for iter_so_far in tqdm(range(int(max_iters))):
    ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'train')
    loss, g = lossandgrad_high(ob_expert, ac_expert, True)
    adam_high.update(g, optim_stepsize)
    if not pretrained:
      ep_stats.add_all_summary(writer, [loss], iter_so_far)
    if iter_so_far % val_per_iter == 0:
      ob_expert, ac_expert = dataset.get_next_batch(-1, 'val')
      loss, g = lossandgrad_high(ob_expert, ac_expert, False)
      logger.log("Validation:")
      logger.log("Loss: %f"%loss)
      if not pretrained:
        U.save_state(os.path.join(ckpt_dir, task_name), counter=iter_so_far)
  if pretrained:
    savedir_fname = tempfile.TemporaryDirectory().name
    U.save_state(savedir_fname, var_list=pi.get_variables())
    return savedir_fname