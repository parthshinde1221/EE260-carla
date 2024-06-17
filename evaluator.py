import carla

class Evaluator:
    def __init__(self, vehicle,agent, destination, max_time, world):
        """
        Initialize the Evaluator class.
        
        :param agent: The autonomous agent to be evaluated
        :param destination: The destination location (carla.Location)
        :param max_time: The maximum allowed time to reach the destination (in seconds)
        :param world: The CARLA world instance
        """
        self.vehicle = vehicle
        self.agent = agent
        self.destination = destination
        self.max_time = max_time
        self.world = world
        self.start_time = None
        self.collisions = 0
        self.lane_crossings = 0
        self.finished = False
        self.destination_reached = False
        self.elapsed_time = 0.0
        self.total_distance_traveled = 0.0
        self.last_location = None
        self.setup_collision_sensor()
        self.setup_lane_invasion_sensor()
        self.callback_id = self.world.on_tick(self.on_tick)

    def setup_collision_sensor(self):
        """
        Setup the collision sensor to monitor collisions.
        """
        bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self.on_collision(event))

    def setup_lane_invasion_sensor(self):
        """
        Setup the lane invasion sensor to monitor lane crossings.
        """
        bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.lane_invasion_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        self.lane_invasion_sensor.listen(lambda event: self.on_lane_invasion(event))

    def on_collision(self, event):
        """
        Collision event handler.
        
        :param event: The collision event
        """
        self.collisions += 1

    def on_lane_invasion(self, event):
        """
        Lane invasion event handler.
        
        :param event: The lane invasion event
        """
        self.lane_crossings += 1

    def on_tick(self, timestamp):
        """
        Tick event handler. Called at every simulation tick.
        
        :param timestamp: The simulation timestamp
        """
        if self.start_time is None and self.vehicle is not None:
            self.start_time = timestamp.elapsed_seconds
            self.last_location = self.vehicle.get_location()

        self.elapsed_time = timestamp.elapsed_seconds - self.start_time

        current_location = self.vehicle.get_location()
        distance_traveled = self.last_location.distance(current_location)
        self.total_distance_traveled += distance_traveled
        self.last_location = current_location
        
        self.evaluate()

        if self.finished:
            self.world.remove_on_tick(self.callback_id)

    def check_destination(self):
        """
        Check if the agent has reached the destination.
        """
        agent_location = self.vehicle.get_location()
        distance = agent_location.distance(self.destination)
        if distance < 2.0:  # Threshold distance to consider destination reached
            self.destination_reached = True

    def check_time(self):
        """
        Check if the agent has run out of time.
        """
        if self.elapsed_time > self.max_time:
            self.finished = True

    def evaluate(self):
        """
        Perform evaluation checks and update status.
        """
        self.check_destination()
        self.check_time()

        if self.destination_reached:
            self.finished = True

    def get_results(self):
        """
        Get the evaluation results.
        
        :return: A dictionary with evaluation results
        """
        infractions_per_km = 0.0
        if self.total_distance_traveled > 0:
            infractions_per_km = (self.collisions + self.lane_crossings) / (self.total_distance_traveled)
        
        results = {
            'destination_reached': self.destination_reached,
            'time_taken': self.elapsed_time,
            'collisions': self.collisions,
            'lane_crossings': self.lane_crossings,
            'total_distance_traveled': self.total_distance_traveled,
            'infractions_per_km': infractions_per_km,
            'finished': self.finished
        }
        return results
