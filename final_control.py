# 
import glob
import os
import sys
import time
import numpy as np
import carla
from IPython.display import display, clear_output
import logging
import random
from datetime import datetime
import pygame
from agents.navigation.basic_agent import BasicAgent
from agent_new import ModelAgent
import traceback
from evaluator import Evaluator

WIDTH = 229
HEIGHT = 229

class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0,255,(HEIGHT,WIDTH,3),dtype='uint8')
        self.processed_image = None
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0,1))

# def create_dataset():
vehicle = None
cam = None

# Enable logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

# Creating a client
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(10.0)
# client.reload_world()
# for mapName in client.get_available_maps():
#     print(mapName)

world = client.get_world()
world_map = world.get_map()

# Clean up existing actors
def cleanup_actors(world):
    actors = world.get_actors()
    vehicle_actors = actors.filter('vehicle.*')
    walker_actors = actors.filter('walker.*')
    sensor_actors = actors.filter('sensor.*')
    
    static_actor_ids = [actor.id for actor in vehicle_actors] + [actor.id for actor in walker_actors] + [actor.id for actor in sensor_actors]

    for actor_id in static_actor_ids:
        actor = world.get_actor(actor_id)
        if actor is not None:
            actor.destroy()

    print(f"Destroyed {len(static_actor_ids)} static actors")


cleanup_actors(world)
actor_list = []
seed_value = 42

random.seed(seed_value)



    
# Spawn vehicle
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

spawn_points = world_map.get_spawn_points()
start_point = random.choice(spawn_points)

vehicle = world.spawn_actor(vehicle_bp, start_point)
actor_list.append(vehicle)

# Create the BasicAgent
# agent = BasicAgent(vehicle)
agent = ModelAgent(vehicle,'testing_Xception_ViT_1.pt')
# agent = 

# Set a random destination
destination = random.choice(spawn_points).location
agent.set_destination(destination)

    
# Adding a RGB camera sensor
cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
cam_bp.set_attribute("image_size_x", str(WIDTH))
cam_bp.set_attribute("image_size_y", str(HEIGHT))
cam_bp.set_attribute("fov", str(105))
cam_location = carla.Location(1, 0, 2.4)
cam_rotation = carla.Rotation(0, 0, 0)
cam_transform = carla.Transform(cam_location, cam_rotation)
cam = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)

# Function to convert image to a numpy array
def process_image(image):
    raw_image = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    raw_image = np.reshape(raw_image, (image.height, image.width, 4))
    processed_image = raw_image[:, :, :3]
    processed_image = processed_image[:, :, ::-1]
    return processed_image

# Save required data
def save_image(carla_image, obj):
    image = process_image(carla_image)
    pygame_image = pygame.surfarray.make_surface(image.swapaxes(0, 1))
    obj.processed_image = image
    obj.surface = pygame_image
    # pygame.image.save(pygame_image, f"{directory}/frame_{frame_id}.png")



renderObj = RenderObject(WIDTH, HEIGHT)
cam.listen(lambda image: save_image(image, renderObj))

# Pygame inits
pygame.init()
gameDisplay = pygame.display.set_mode((WIDTH,HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
# Draw black to the display
gameDisplay.fill((0,0,0))
gameDisplay.blit(renderObj.surface, (0,0))

# Initialize Evaluator class
evaluator_agent = Evaluator(vehicle,agent, destination, max_time=50, world=world)

pygame.display.flip()

pygame_clock = pygame.time.Clock()
try:
    i = 0
    while not evaluator_agent.finished:
        world.tick()
        pygame_clock.tick_busy_loop(20)
        # print(f"{str(i)} frames saved")
        # i += 1

        # Advance the simulation time
        # world.tick()
        # Update the display
        # pygame_clock.tick_busy_loop(20)


        control = agent.run_step(renderObj.processed_image,measurements=vehicle.get_velocity().length())
        # control = agent.run_step()
        vehicle.apply_control(control)

        gameDisplay.blit(renderObj.surface, (0,0))
        pygame.display.flip()

        print(f"Steering: {control.steer}, Throttle: {control.throttle}, Brake: {control.brake}")
        

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
        
        if agent.done():
            print("Destination reached")
            evaluator_agent.finished = True
            evaluator_agent.destination_reached = True
            break

except KeyboardInterrupt:
    print('\nSimulation interrupted.')
except Exception as e:
    print(f'\nSimulation error: {e}')
    traceback.print_exc()

results = evaluator_agent.get_results()
print(results)


if vehicle is not None:
    if cam is not None:
        cam.stop()
        cam.destroy()
    vehicle.destroy()

del evaluator_agent
del world
print('Evaluation done!!')
print('Evaluator destroyed.')

pygame.quit()

# AGENTS = 2

# for i in range(AGENTS):
#     create_dataset()
#     print(f'{i}-agent finished collecting dataset')
