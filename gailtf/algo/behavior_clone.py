import tensorflow as tf
import gailtf.baselines.common.tf_util as U
from gailtf.baselines import logger
from tqdm import tqdm
from gailtf.baselines.common.mpi_adam import MpiAdam
import tempfile, os
from gailtf.common.statistics import stats
import ipdb
import time

def evaluate(env, policy_func, load_model_path_high, load_model_path_low, stochastic_policy=False, number_trajs=1000):
  from gailtf.algo.trpo_mpi import traj_episode_generator_combine
  ob_space = env.observation_space
  ac_space = env.action_space



  graph_high = tf.Graph()
  with graph_high.as_default():
    pi_high = policy_func("pi_high", ob_space, ac_space.spaces[0])  # high -> action_label

  graph_low = tf.Graph()
  with graph_low.as_default():
    pi_low = policy_func("pi_low", ob_space, ac_space.spaces[1])

  sess_high = tf.Session(graph=graph_high)
  sess_low = tf.Session(graph=graph_low)

  with sess_high.as_default():
    with graph_high.as_default():
      tf.global_variables_initializer().run()
      saver_high = tf.train.import_meta_graph(load_model_path_high + '.meta', clear_devices=True)  # saver = tf.train.Saver()
      saver_high.restore(sess_high, load_model_path_high)

  with sess_low.as_default():
    with graph_low.as_default():
      tf.global_variables_initializer().run()
      saver_low = tf.train.import_meta_graph(load_model_path_low + '.meta', clear_devices=True)  # saver = tf.train.Saver()
      saver_low.restore(sess_low, load_model_path_low)
  # placeholder
  # ob = U.get_placeholder_cached(name="ob")

  # ac_high = pi_high.pdtype.sample_placeholder([None,1])
  # ac_low = pi_low.pdtype.sample_placeholder([None])

  # stochastic = U.get_placeholder_cached(name="stochastic")

  # U.load_state(load_model_path_high)
  # U.load_state(load_model_path_low)
  ep_gen = traj_episode_generator_combine(sess_high, sess_low, pi_high, pi_low, env, 1024, stochastic=stochastic_policy)

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

def learn(env, policy_func, dataset, pretrained, optim_batch_size=128, max_iters=1e3,
           adam_epsilon=1e-6, optim_stepsize=2e-4, ckpt_dir=None, log_dir=None, task_name=None, high_level=False):
  val_per_iter = int(max_iters/100)
  ob_space = env.observation_space
  ac_space = env.action_space
  start_time = time.time()
  if not high_level:

    pi_low = policy_func("pi_low", ob_space, ac_space.spaces[1])

    # placeholder
    # ob_low = U.get_placeholder_cached(name="ob")
    ob_low = pi_low.ob
    ac_low = pi_low.pdtype.sample_placeholder([None])
    # stochastic_low = U.get_placeholder_cached(name="stochastic")
    stochastic_low = pi_low.stochastic
    loss_low = tf.reduce_mean(tf.square(ac_low - pi_low.ac))
    var_list_low = pi_low.get_trainable_variables()
    adam_low = MpiAdam(var_list_low, epsilon=adam_epsilon)
    lossandgrad_low = U.function([ob_low, ac_low, stochastic_low], [loss_low] + [U.flatgrad(loss_low, var_list_low)])


    if not pretrained:
      writer = U.FileWriter(log_dir)
      ep_stats_low = stats(["Loss_low"])
    U.initialize()
    adam_low.sync()
    logger.log("Pretraining with Behavior Cloning Low...")
    for iter_so_far in tqdm(range(int(max_iters))):

      ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'train', high_level)
      loss, g = lossandgrad_low(ob_expert, ac_expert, True)
      adam_low.update(g, optim_stepsize)
      if not pretrained:
        ep_stats_low.add_all_summary(writer, [loss], iter_so_far)
      if iter_so_far % val_per_iter == 0:
        ob_expert, ac_expert = dataset.get_next_batch(-1, 'val', high_level)
        loss, g = lossandgrad_low(ob_expert, ac_expert, False)
        logger.log("Validation:")
        logger.log("Loss: %f"%loss)
        if not pretrained:
          U.save_state(os.path.join(ckpt_dir, task_name), counter=iter_so_far)


    if pretrained:
      savedir_fname = tempfile.TemporaryDirectory().name
      U.save_state(savedir_fname, var_list=pi_low.get_variables())
      return savedir_fname

  else:
    pi_high = policy_func("pi_high", ob_space, ac_space.spaces[0])  # high -> action_label
    # ob_high = U.get_placeholder_cached(name="ob")
    ob_high = pi_high.ob
    ac_high = pi_high.pdtype.sample_placeholder([None,1])
    onehot_labels = tf.one_hot(indices=tf.cast(ac_high,tf.int32), depth=3)
    # stochastic_high = U.get_placeholder_cached(name="stochastic")
    stochastic_high = pi_high.stochastic
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=pi_high.logits,
                                                            labels=onehot_labels)
    loss_high = tf.reduce_mean(cross_entropy)
    var_list_high = pi_high.get_trainable_variables()
    adam_high = MpiAdam(var_list_high, epsilon=adam_epsilon)
    lossandgrad_high = U.function([ob_high, ac_high, stochastic_high], [loss_high]+[U.flatgrad(loss_high, var_list_high)])

    # train high level policy
    if not pretrained:
      writer = U.FileWriter(log_dir)
      # ep_stats_low = stats(["Loss_low"])
      ep_stats_high = stats(["loss_high"])
    U.initialize()
    adam_high.sync()
    logger.log("Pretraining with Behavior Cloning High...")
    for iter_so_far in tqdm(range(int(max_iters))):

      ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'train', high_level)
      loss, g = lossandgrad_high(ob_expert, ac_expert, True)
      adam_high.update(g, optim_stepsize)
      if not pretrained:
        ep_stats_high.add_all_summary(writer, [loss], iter_so_far)
      if iter_so_far % val_per_iter == 0:
        ob_expert, ac_expert = dataset.get_next_batch(-1, 'val', high_level)
        loss, g = lossandgrad_high(ob_expert, ac_expert, False)
        logger.log("Validation:")
        logger.log("Loss: %f"%loss)
        if not pretrained:
          U.save_state(os.path.join(ckpt_dir, task_name), counter=iter_so_far)
    if pretrained:
      savedir_fname = tempfile.TemporaryDirectory().name
      U.save_state(savedir_fname, var_list=pi_high.get_variables())
      return savedir_fname

  print("--- %s seconds ---" % (time.time() - start_time))


