from controller import Robot


class TestRobot(Robot):
    def __init__(self):
        super().__init__()

        self.robot_num = int(self.getName()[-1])
        self.timestep = int(self.getBasicTimeStep())

        self.emitter, self.receiver = self.initialize_comms()

        self.emitter.setChannel(self.robot_num)
        self.receiver.setChannel(self.robot_num)

        self.positionSensor = self.getDevice("polePosSensor")
        self.positionSensor.enable(self.timestep)

        self.wheels = [None for _ in range(4)]
        self.setup_motors()

    def initialize_comms(self):
        emitter = self.getDevice('emitter')
        receiver = self.getDevice('receiver')
        receiver.enable(self.timestep)
        return emitter, receiver

    def setup_motors(self):
        """
        This method initializes the four wheels, storing the references inside a list and setting the starting
        positions and velocities.
        """
        self.wheels[0] = self.getDevice('wheel1')
        self.wheels[1] = self.getDevice('wheel2')
        self.wheels[2] = self.getDevice('wheel3')
        self.wheels[3] = self.getDevice('wheel4')
        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(0.0)

    def handle_receiver(self):
        """
        Modified handle_receiver from the basic implementation of deepbots.
        This one consumes all available messages in the queue during the step it is called.
        """
        while self.receiver.getQueueLength() > 0:
            # Receive and decode message from supervisor
            message = self.receiver.getData().decode("utf-8")
            # Convert string message into a list
            # message = message.split(",")

            self.use_message_data(message)

            self.receiver.nextPacket()

    def use_message_data(self, message):
        action = int(message)

        assert action == 0 or action == 1, "CartPoleRobot controller got incorrect action value: " + str(
            action)

        if action == 0:
            motorSpeed = 5.0
        else:
            motorSpeed = -5.0

        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(motorSpeed)

    def handle_emitter(self):
        data = self.create_message()

        # assert isinstance(data,
        #                   Iterable), "The action object should be Iterable"

        string_message = ""
        # message can either be a list that needs to be converted in a string
        # or a straight-up string
        if type(data) is list:
            string_message = ",".join(map(str, data))
        elif type(data) is str:
            string_message = data
        else:
            raise TypeError(
                "message must be either a comma-separated string or a 1D list")

        string_message = string_message.encode("utf-8")
        self.emitter.send(string_message)

    def create_message(self):
        # print("Position sensor", self.positionSensor.getValue())
        return [self.positionSensor.getValue()]

    def run(self):
        """
        This method is required by Webots to update the robot in the
        simulation. It steps the robot and in each step it runs the two
        handler methods to use the emitter and receiver components.

        This method should be called by a robot manager to run the robot.
        """
        while self.step(self.timestep) != -1:
            self.handle_receiver()
            self.handle_emitter()


robot_controller = TestRobot()
robot_controller.run()
