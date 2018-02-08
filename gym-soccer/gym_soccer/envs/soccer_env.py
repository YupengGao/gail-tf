import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import socket
from tornado import netutil
import numpy as np
try:
    import hfo_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you can install HFO dependencies with 'pip install gym[soccer].)'".format(e))

import logging
logger = logging.getLogger(__name__)

class SoccerEnvInit(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.viewer = None
        self.server_process = None
        self.server_port = None
        self.hfo_path = hfo_py.get_hfo_path()
        # need to check unused port
        sock, port2use = self.bind_unused_port()
        num_offense_agents = 0
        num_offense_agents_npcs = 1
        self._configure_environment(port2use, num_offense_agents, num_offense_agents_npcs)
        #process = subprocess.Popen(self.hfo_path+' --offense-agents=1 --defense-npcs=1', shell=True, stdout=subprocess.PIPE)
        self.env = hfo_py.HFOEnvironment()
        self.env.connectToServer(feature_set=hfo_py.HIGH_LEVEL_FEATURE_SET, config_dir=hfo_py.get_config_path(), server_port=port2use)
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(self.env.getStateSize()))
        # Action space omits the Tackle/Catch actions, which are useful on defense
        self.action_space = spaces.Tuple((spaces.Discrete(3),
                                          spaces.Box(low=0, high=100, shape=1),
                                          spaces.Box(low=-180, high=180, shape=1),
                                          spaces.Box(low=-180, high=180, shape=1),
                                          spaces.Box(low=0, high=100, shape=1),
                                          spaces.Box(low=-180, high=180, shape=1)))

        # self.action_space = spaces.Tuple(spaces.Discrete(3),
        #                                    spaces.Box(low=np.array([0,-180,-180,0,-180]), high=np.array([100,180,180,100,180])))
        self.status = hfo_py.IN_GAME

    def __del__(self):
        self.env.act(hfo_py.QUIT)
        self.env.step()
        os.kill(self.server_process.pid, signal.SIGINT)
        if self.viewer is not None:
            os.kill(self.viewer.pid, signal.SIGKILL)

    def _configure_environment(self, port2use, num_offense_agents, num_offense_agents_npcs):
        """
        Provides a chance for subclasses to override this method and supply
        a different server configuration. By default, we initialize one
        offense agent against no defenders.
        """
        self._start_hfo_server(port=port2use, offense_agents = num_offense_agents, offense_npcs = num_offense_agents_npcs)
        #self._start_hfo_server()
    def _start_hfo_server(self, trials = 0, frames = 0, frames_per_trial=0,
                          untouched_time=100, offense_agents=1,
                          defense_agents=0, offense_npcs=1,
                          defense_npcs=0, sync_mode=True, port=6000,
                          offense_on_ball=0, fullstate=True, seed=-1,
                          ball_x_min=0.0, ball_x_max=0.2,
                          verbose=False, log_game=False,
                          log_dir="log"):
        """
        Starts the Half-Field-Offense server.
        frames_per_trial: Episodes end after this many steps.
        untouched_time: Episodes end if the ball is untouched for this many steps.
        offense_agents: Number of user-controlled offensive players.
        defense_agents: Number of user-controlled defenders.
        offense_npcs: Number of offensive bots.
        defense_npcs: Number of defense bots.
        sync_mode: Disabling sync mode runs server in real time (SLOW!).
        port: Port to start the server on.
        offense_on_ball: Player to give the ball to at beginning of episode.
        fullstate: Enable noise-free perception.
        seed: Seed the starting positions of the players and ball.
        ball_x_[min/max]: Initialize the ball this far downfield: [0,1]
        verbose: Verbose server messages.
        log_game: Enable game logging. Logs can be used for replay + visualization.
        log_dir: Directory to place game logs (*.rcg).
        """

        self.server_port = port
        cmd = self.hfo_path + \
              " --trials %i --frames %i --frames-per-trial %i --untouched-time %i --offense-agents %i"\
              " --defense-agents %i --offense-npcs %i --defense-npcs %i"\
              " --port %i --offense-on-ball %i --seed %i --ball-x-min %f"\
              " --ball-x-max %f --log-dir %s"\
              % (trials, frames, frames_per_trial, untouched_time, offense_agents,
                 defense_agents, offense_npcs, defense_npcs, port,
                 offense_on_ball, seed, ball_x_min, ball_x_max,
                 log_dir)
        if not sync_mode: cmd += " --no-sync"
        if fullstate:     cmd += " --fullstate"
        if verbose:       cmd += " --verbose"
        if not log_game:  cmd += " --no-logging"
        print('Starting server with command: %s' % cmd)
        self.server_process = subprocess.Popen(cmd.split(' '), shell=False)
        time.sleep(10) # Wait for server to startup before connecting a player

    def _start_viewer(self):
        """
        Starts the SoccerWindow visualizer. Note the viewer may also be
        used with a *.rcg logfile to replay a game. See details at
        https://github.com/LARG/HFO/blob/master/doc/manual.pdf.
        """
        cmd = hfo_py.get_viewer_path() +\
              " --connect --port %d" % (self.server_port)
        self.viewer = subprocess.Popen(cmd.split(' '), shell=False)

    def _step(self, action):
        self._take_action(action)
        self.status = self.env.step()
        reward = self._get_reward()
        ob = self.env.getState()
        episode_over = self.status != hfo_py.IN_GAME
        return ob, reward, episode_over, {}

    def _take_action(self, action):
        """ Converts the action space into an HFO action. """
        action_type = ACTION_LOOKUP[action[0]]
        if action_type == hfo_py.DASH:
            self.env.act(action_type, action[1], action[2])
        elif action_type == hfo_py.TURN:
            self.env.act(action_type, action[3])
        elif action_type == hfo_py.KICK:
            self.env.act(action_type, action[4], action[5])
        else:
            print('Unrecognized action %d' % action_type)
            self.env.act(hfo_py.NOOP)

    def _get_reward(self):
        """ Reward is given for scoring a goal. """
        if self.status == hfo_py.GOAL:
            return 1
        else:
            return 0

    def _reset(self):
        """ Repeats NO-OP action until a new episode begins. """
        while self.status == hfo_py.IN_GAME:
            self.env.act(hfo_py.NOOP)
            self.status = self.env.step()
        while self.status != hfo_py.IN_GAME:
            self.env.act(hfo_py.NOOP)
            self.status = self.env.step()
        return self.env.getState()

    def _render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        if close:
            if self.viewer is not None:
                os.kill(self.viewer.pid, signal.SIGKILL)
        else:
            if self.viewer is None:
                self._start_viewer()
    # function to sample trajectories
    def sampleTrajs(self, numofTrajs):
        for episode in range(numofTrajs): # replace with xrange(5) for Python 2.X
            status = hfo_py.IN_GAME
            total_steps = 0
            while status == hfo_py.IN_GAME:
                features = self.env.getState()
                self.env.act(hfo_py.DASH, 20.0, 0.0)
                status = self.env.step()
                total_steps += 1
            print('Miao Episode', episode, 'lasted ', total_steps, 'steps ended')
    def bind_unused_port(self, reuse_port=False):
        """Binds a server socket to an available port on localhost.
        Returns a tuple (socket, port).
        .. versionchanged:: 4.4
        Always binds to ``127.0.0.1`` without resolving the name
        ``localhost``.
        """
        sock = netutil.bind_sockets(None, 'localhost', family=socket.AF_INET)[0]
        port = sock.getsockname()[1]
        return sock, port

class SoccerEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.viewer = None
        self.server_process = None
        self.server_port = None
        self.hfo_path = hfo_py.get_hfo_path()
        # need to check unused port
        sock, port2use = self.bind_unused_port()
        num_offense_agents = 1
        num_offense_agents_npcs = 0
        self._configure_environment(port2use, num_offense_agents, num_offense_agents_npcs)
        #process = subprocess.Popen(self.hfo_path+' --offense-agents=1 --defense-npcs=1', shell=True, stdout=subprocess.PIPE)
        self.env = hfo_py.HFOEnvironment()
        self.env.connectToServer(feature_set=hfo_py.HIGH_LEVEL_FEATURE_SET, config_dir=hfo_py.get_config_path(), server_port=port2use)
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(self.env.getStateSize()))
        # Action space omits the Tackle/Catch actions, which are useful on defense
        # self.action_space = spaces.Tuple((spaces.Discrete(3),
        #                                   spaces.Box(low=0, high=100, shape=1),
        #                                   spaces.Box(low=-180, high=180, shape=1),
        #                                   spaces.Box(low=-180, high=180, shape=1),
        #                                   spaces.Box(low=0, high=100, shape=1),
        #                                   spaces.Box(low=-180, high=180, shape=1)))

        self.action_space = spaces.Tuple((spaces.Discrete(3),spaces.Box(low=np.array([0, -180, -180, 0, -180]),high=np.array([100, 180, 180, 100, 180]))))
        self.status = hfo_py.IN_GAME

    def __del__(self):
        self.env.act(hfo_py.QUIT)
        self.env.step()
        os.kill(self.server_process.pid, signal.SIGINT)
        if self.viewer is not None:
            os.kill(self.viewer.pid, signal.SIGKILL)

    def _configure_environment(self, port2use, num_offense_agents, num_offense_agents_npcs):
        """
        Provides a chance for subclasses to override this method and supply
        a different server configuration. By default, we initialize one
        offense agent against no defenders.
        """
        self._start_hfo_server(port=port2use, offense_agents = num_offense_agents, offense_npcs = num_offense_agents_npcs)
        #self._start_hfo_server()
    def _start_hfo_server(self, frames_per_trial=500,
                          untouched_time=100, offense_agents=1,
                          defense_agents=0, offense_npcs=0,
                          defense_npcs=0, sync_mode=True, port=6000,
                          offense_on_ball=0, fullstate=True, seed=-1,
                          ball_x_min=0.0, ball_x_max=0.2,
                          verbose=False, log_game=False,
                          log_dir="log"):
        """
        Starts the Half-Field-Offense server.
        frames_per_trial: Episodes end after this many steps.
        untouched_time: Episodes end if the ball is untouched for this many steps.
        offense_agents: Number of user-controlled offensive players.
        defense_agents: Number of user-controlled defenders.
        offense_npcs: Number of offensive bots.
        defense_npcs: Number of defense bots.
        sync_mode: Disabling sync mode runs server in real time (SLOW!).
        port: Port to start the server on.
        offense_on_ball: Player to give the ball to at beginning of episode.
        fullstate: Enable noise-free perception.
        seed: Seed the starting positions of the players and ball.
        ball_x_[min/max]: Initialize the ball this far downfield: [0,1]
        verbose: Verbose server messages.
        log_game: Enable game logging. Logs can be used for replay + visualization.
        log_dir: Directory to place game logs (*.rcg).
        """

        self.server_port = port
        cmd = self.hfo_path + \
              " --headless --frames-per-trial %i --untouched-time %i --offense-agents %i"\
              " --defense-agents %i --offense-npcs %i --defense-npcs %i"\
              " --port %i --offense-on-ball %i --seed %i --ball-x-min %f"\
              " --ball-x-max %f --log-dir %s"\
              % (frames_per_trial, untouched_time, offense_agents,
                 defense_agents, offense_npcs, defense_npcs, port,
                 offense_on_ball, seed, ball_x_min, ball_x_max,
                 log_dir)
        if not sync_mode: cmd += " --no-sync"
        if fullstate:     cmd += " --fullstate"
        if verbose:       cmd += " --verbose"
        if not log_game:  cmd += " --no-logging"
        print('Starting server with command: %s' % cmd)
        self.server_process = subprocess.Popen(cmd.split(' '), shell=False)
        time.sleep(10) # Wait for server to startup before connecting a player

    def _start_viewer(self):
        """
        Starts the SoccerWindow visualizer. Note the viewer may also be
        used with a *.rcg logfile to replay a game. See details at
        https://github.com/LARG/HFO/blob/master/doc/manual.pdf.
        """
        cmd = hfo_py.get_viewer_path() +\
              " --connect --port %d" % (self.server_port)
        self.viewer = subprocess.Popen(cmd.split(' '), shell=False)

    def _step(self, action):
        self._take_action(action)
        self.status = self.env.step()
        reward = self._get_reward()
        ob = self.env.getState()
        episode_over = self.status != hfo_py.IN_GAME
        return ob, reward, episode_over, {}

    def _take_action(self, action):
        """ Converts the action space into an HFO action. """
        action_type = ACTION_LOOKUP[action[0]]
        if action_type == hfo_py.DASH:
            self.env.act(action_type, action[1], action[2])
        elif action_type == hfo_py.TURN:
            self.env.act(action_type, action[3])
        elif action_type == hfo_py.KICK:
            self.env.act(action_type, action[4], action[5])
        else:
            print('Unrecognized action %d' % action_type)
            self.env.act(hfo_py.NOOP)

    def _get_reward(self):
        """ Reward is given for scoring a goal. """
        if self.status == hfo_py.GOAL:
            return 1
        else:
            return 0

    def _reset(self):
        """ Repeats NO-OP action until a new episode begins. """
        while self.status == hfo_py.IN_GAME:
            self.env.act(hfo_py.NOOP)
            self.status = self.env.step()
        while self.status != hfo_py.IN_GAME:
            self.env.act(hfo_py.NOOP)
            self.status = self.env.step()
        return self.env.getState()

    def _render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        if close:
            if self.viewer is not None:
                os.kill(self.viewer.pid, signal.SIGKILL)
        else:
            if self.viewer is None:
                self._start_viewer()
    # function to sample trajectories
    def sampleTrajs(self, numofTrajs):
        for episode in range(numofTrajs): # replace with xrange(5) for Python 2.X
            status = hfo_py.IN_GAME
            total_steps = 0
            while status == hfo_py.IN_GAME:
                features = self.env.getState()
                self.env.act(hfo_py.DASH, 20.0, 0.0)
                status = self.env.step()
                total_steps += 1
            print('Miao Episode', episode, 'lasted ', total_steps, 'steps ended')
    def bind_unused_port(self, reuse_port=False):
        """Binds a server socket to an available port on localhost.
        Returns a tuple (socket, port).
        .. versionchanged:: 4.4
        Always binds to ``127.0.0.1`` without resolving the name
        ``localhost``.
        """
        sock = netutil.bind_sockets(None, 'localhost', family=socket.AF_INET)[0]
        port = sock.getsockname()[1]
        return sock, port

#ACTION_LOOKUP = {
#    0 : hfo_py.DASH,
#    1 : hfo_py.TURN,
#    2 : hfo_py.KICK,
#    3 : hfo_py.TACKLE, # Used on defense to slide tackle the ball
#    4 : hfo_py.CATCH,  # Used only by goalie to catch the ball
#}
ACTION_LOOKUP = {
    0 : hfo_py.MOVE,
    1 : hfo_py.SHOOT,
    2 : hfo_py.PASS,
    3 : hfo_py.DRIBBLE,
    4 : hfo_py.CATCH,
    4 : hfo_py.NOOP, # Used on defense to slide tackle the ball
    5 : hfo_py.QUIT,  # Used only by goalie to catch the ball
}


