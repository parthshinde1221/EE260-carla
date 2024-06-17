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

WIDTH = 229
HEIGHT = 229

class RenderObject(object):
    def __init__(self, width, height):
        init_image = np.random.randint(0,255,(HEIGHT,WIDTH,3),dtype='uint8')
        self.surface = pygame.surfarray.make_surface(init_image.swapaxes(0,1))

def create_dataset():
    vehicle = None
    cam = None

    # Enable logging
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    # Creating a client
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    # client.reload_world()
    for mapName in client.get_available_maps():
        print(mapName)
    world = client.get_world()

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

    # Create folder to store data
    today = datetime.now()
    h = f"{today.hour:02}"
    m = f"{today.minute:02}"
    directory = f"./training_data_test_final_proj_3/{today.strftime('%Y%m%d_')}{h}{m}_npy"
    print(directory)

    try:
        os.makedirs(directory)
    except:
        print("Directory already exists")
    
    try:
        inputs_file = open(directory + "/inputs.npy", "ba+") 
        outputs_file = open(directory + "/outputs.npy", "ba+")     
    except:
        print("Files could not be opened")
        
    # Spawn vehicle
    bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    bp.set_attribute('role_name', 'brax')
    color = random.choice(bp.get_attribute('color').recommended_values)
    bp.set_attribute('color', color)

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)

    if number_of_spawn_points > 0:
        random.shuffle(spawn_points)
        transform = spawn_points[0]
        vehicle = world.spawn_actor(bp, transform)
        print('\nVehicle spawned')
    else:
        logging.warning('Could not find any spawn points')
        
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
        control = vehicle.get_control()
        data = [control.steer, control.throttle, control.brake]
        np.save(inputs_file, image)
        np.save(outputs_file, data)

        # Save image as PNG
        # frame_id = carla_image.frame_number
        pygame_image = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        obj.surface = pygame_image
        # pygame.image.save(pygame_image, f"{directory}/frame_{frame_id}.png")
    
    # Traffic Manager
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    # traffic_manager.auto_lane_change(vehicle, True)  # Enable auto lane change
    # traffic_manager.set_global_distance_to_leading_vehicle(2.5)  # Adjust distance to leading vehicle
    traffic_manager.vehicle_percentage_speed_difference(vehicle, -50)  # Reduce speed by 30%

    vehicle.set_autopilot(True, traffic_manager.get_port())
        
    # vehicle.set_autopilot(True)

    renderObj = RenderObject(WIDTH, HEIGHT)
    cam.listen(lambda image: save_image(image, renderObj))

    # Pygame inits
    pygame.init()
    gameDisplay = pygame.display.set_mode((WIDTH,HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    # Draw black to the display
    gameDisplay.fill((0,0,0))
    gameDisplay.blit(renderObj.surface, (0,0))

    pygame.display.flip()

    pygame_clock = pygame.time.Clock()
    try:
        i = 0
        while i < 1500:
            world.tick()
            pygame_clock.tick_busy_loop(20)
            print(f"{str(i)} frames saved")
            i += 1

            # Advance the simulation time
            # world.tick()
            # Update the display
            # pygame_clock.tick_busy_loop(20)
            gameDisplay.blit(renderObj.surface, (0,0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        print('\nSimulation interrupted.')
    except Exception as e:
        print(f'\nSimulation error: {e}')

    if vehicle is not None:
        if cam is not None:
            cam.stop()
            cam.destroy()
        vehicle.destroy()

    inputs_file.close()
    outputs_file.close()
    print("Data retrieval finished")
    print(directory)

    pygame.quit()

AGENTS = 3

for i in range(AGENTS):
    create_dataset()
    print(f'{i}-agent finished collecting dataset')
