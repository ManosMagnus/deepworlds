import numpy as np
from controller import Supervisor
from deepbots.supervisor.controllers.supervisor_env import SupervisorEnv

import PPO_runner
from utilities import normalizeToRange


class TestSupervisor(SupervisorEnv):
    def __init__(self):
        super().__init__()

        self.timestep = int(self.getBasicTimeStep())
        self.communication = self.initialize_comms(2)

        self.observationSpace = 4
        self.actionSpace = 2
        self.robot = [self.getFromDef("ROBOT" + str(i)) for i in range(2)]
        self.initPositions = [
            self.robot[i].getField("translation").getSFVec3f()
            for i in range(2)
        ]
        self.poleEndpoint = [
            self.getFromDef("POLE_ENDPOINT_" + str(i)) for i in range(2)
        ]

        self.messageReceived = None  # Variable to save the messages received from the robots

        self.stepsPerEpisode = 200  # Number of steps per episode
        self.episodeScore = 0  # Score accumulated during an episode
        self.episodeScoreList = [
        ]  # A list to save all the episode scores, used to check if task is solved
        self.test = False  # Whether the agent is in test mode

    def initialize_comms(self, robots_num):
        communication = []
        for i in range(robots_num):
            emitter = self.getDevice('emitter{}'.format(i))
            receiver = self.getDevice('receiver{}'.format(i))

            emitter.setChannel(i)
            receiver.setChannel(i)

            receiver.enable(self.timestep)

            communication.append({
                'emitter': emitter,
                'receiver': receiver,
            })
        return communication

    def step(self, action):
        if super(Supervisor, self).step(self.timestep) == -1:
            exit()

        self.handle_emitter(action)
        return (
            self.get_observations(),
            self.get_reward(action),
            self.is_done(),
            self.get_info(),
        )

    def get_info(self):
        pass

    def handle_emitter(self, actions):
        for i, action in enumerate(actions):
            message = str(action).encode("utf-8")
            self.communication[i]['emitter'].send(message)

    def handle_receiver(self):
        messages = []
        for com in self.communication:
            receiver = com['receiver']
            if receiver.getQueueLength() > 0:
                messages.append(receiver.getData().decode("utf-8"))
                receiver.nextPacket()
            else:
                messages.append(None)

        return messages

    def get_observations(self):
        robots_num = len(self.robot)
        # Position on z axis
        cartPosition = [
            normalizeToRange(self.robot[i].getPosition()[2], -0.4, 0.4, -1.0,
                             1.0) for i in range(robots_num)
        ]

        # Linear velocity on z axis
        cartVelocity = [
            normalizeToRange(self.robot[i].getVelocity()[2],
                             -0.2,
                             0.2,
                             -1.0,
                             1.0,
                             clip=True) for i in range(robots_num)
        ]

        self.messageReceived = self.handle_receiver()

        poleAngle = [None for _ in range(robots_num)]
        for i, message in enumerate(self.messageReceived):
            if message is not None:
                poleAngle[i] = normalizeToRange(message,
                                                -0.23,
                                                0.23,
                                                -1.0,
                                                1.0,
                                                clip=True)
            else:
                poleAngle = [0.0 for _ in range(robots_num)]

        # Angular velocity x of endpoint
        endpointVelocity = [
            normalizeToRange(self.poleEndpoint[i].getVelocity()[3],
                             -1.5,
                             1.5,
                             -1.0,
                             1.0,
                             clip=True) for i in range(robots_num)
        ]

        messages = [None for _ in range(robots_num)]
        for i in range(robots_num):
            messages[i] = [
                cartPosition[i], cartVelocity[i], poleAngle[i],
                endpointVelocity[i]
            ]

        return messages

    def get_reward(self, action=None):
        """
        Reward is +1 for each step taken, including the termination step.

        :param action: Not used, defaults to None
        :type action: None, optional
        :return: Always 1
        :rtype: int
        """
        return 1

    def is_done(self):
        """
        An episode is done if the score is over 195.0, or if the pole is off balance, or the cart position is a certain distance 
        awat from the initial position for either of the carts

        :return: True if termination conditions are met, False otherwise
        :rtype: bool
        """

        robots_num = len(self.robot)
        if self.episodeScore > 195.0:
            return True

        poleAngle = [None for i in range(robots_num)]
        for i, message in enumerate(self.messageReceived):
            if message is not None:
                poleAngle[i] = round(float(message), 2)
            else:
                poleAngle = [0.0 for _ in range(robots_num)]

        if not all(abs(x) < 0.261799388
                   for x in poleAngle):  # 15 degrees off vertical
            print("From angle")
            return True

        cartPosition = [
            round(self.robot[i].getPosition()[2] - self.initPositions[i][2], 2)
            for i in range(robots_num)
        ]
        if not all(abs(x) < 0.89 for x in cartPosition):
            return True

        return False

    def get_default_observation(self):
        """
        Returns the default observation of zeros.

        :return: Default observation zero vector
        :rtype: list
        """
        observation = []

        for _ in range(len(self.robot)):
            robot_obs = [0.0 for _ in range(self.observationSpace)]
            observation.append(robot_obs)

        # print("Default Supervisor Receiver0: ",
        #       self.communication[0]['receiver'].getQueueLength())
        # print("Default Supervisor Receiver1: ",
        #       self.communication[1]['receiver'].getQueueLength())

        return observation

    def solved(self):
        """
        This method checks whether the CartPole task is solved, so training terminates.
        Solved condition requires that the average episode score of last 100 episodes is over 195.0.

        :return: True if task is solved, False otherwise
        :rtype: bool
        """
        if len(self.episodeScoreList) > 100:  # Over 100 trials thus far
            if np.mean(self.episodeScoreList[-100:]
                       ) > 195.0:  # Last 100 episode scores average value
                return True
        return False

    def reset(self):
        """
        Used to reset the world to an initial state.

        Default, problem-agnostic, implementation of reset method,
        using Webots-provided methods.

        *Note that this works properly only with Webots versions >R2020b
        and must be overridden with a custom reset method when using
        earlier versions. It is backwards compatible due to the fact
        that the new reset method gets overridden by whatever the user
        has previously implemented, so an old supervisor can be migrated
        easily to use this class.

        :return: default observation provided by get_default_observation()
        """
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        super(Supervisor, self).step(int(self.getBasicTimeStep()))

        # print("Before Supervisor Receiver0: ",
        #       self.communication[0]['receiver'].getQueueLength())
        # print("Before Supervisor Receiver1: ",
        #       self.communication[1]['receiver'].getQueueLength())

        self.communication[0]['receiver'].disable()
        self.communication[1]['receiver'].disable()

        self.communication[0]['receiver'].enable(self.timestep)
        self.communication[1]['receiver'].enable(self.timestep)

        for i in range(2):
            receiver = self.communication[i]['receiver']
            while receiver.getQueueLength() > 0:
                receiver.nextPacket()

        # print("After Supervisor Receiver0: ",
        #       self.communication[0]['receiver'].getQueueLength())
        # print("After Supervisor Receiver1: ",
        #       self.communication[1]['receiver'].getQueueLength())

        return self.get_default_observation()


PPO_runner.run(TestSupervisor())
