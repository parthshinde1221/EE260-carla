# Autonomous car CARLA project
This is a project for developing a new baseline model for end-to-end autonomous driving using Imitation learning

# Steps to run
1. Download the Xception_ViT.pt trained baseline model file from this drive link into the folder:- https://drive.google.com/drive/folders/1FergNAMp9R-S3I6Jx8wEI4-iRj1NVKmt?usp=drive_link
2. Setup the environment accroding to the env yaml file given using conda
3. To run the file use python final_control.py or vglrun python final_control.py
4. Alternativaly, if you want to generate the dataset and train the file follow the steps below
    1. Run the dataset_creator.py file
    2. Run the model_vit.py for training vit_sception models and if you want to run only xception models you could so by running model_newer.py
    3. Then you can run the final_control directly,or if you want to check xception results uncomment 'self.model' according to your model_baseline in the agent_new.py file
    4. Then run the final_control.py file

# The file structure/description of the project repo is given below
    - agents folder
       - Contains all the predefinded carla agents
       - We inherit these agents
       - We have specfically used the basic agent
    - dataset_creator.py
       - Used create datasets using batching,pin-memory,and multiple workers with Dataloaders
       - It runs CARLA and generate datasets for us from random spawn point for n-number of frames and stores it .npy compressed format for easy access
    - model_newer.py/model_vit.py
       - Models are defined and trained here for n-epochs in a batched format
       - It helps to efficiently load data from cpu to device(i.e gpu,cuda()) and train our models on it
    - agent_new.py
       - It defines our own custom agent
       - It is built upon/inherits the BasicAgent module which gives use various functionalties
       - We have created according to the CARLA documentation
       - It takes in images and according to the model structure and weights file used it takes in images and gives out the controls for it
    - evaluator.py
       - Here the agents/vehicles are evaluated by gathering as simulation.ticks(time taken simulation),lane crossing infractions and collisions,etc
       - It takes in the agent and its current location,and the world and checks the time,location infraction errors accordingly
       - The game loop ends when the evalutor is finished evaluating or the pygame window closes
    - All '.pt' files
       - Models are saved in this standard pytorch format for saving tensors(ie weights of the model trained)
    - final_control.py
       - This is the main file everything is run
       - It intializes the world,sensors,pygame,agents(and in turn models)
       - It runs the simulation in the pygame loop and then destoys any actors that are present

**In this project we tested without pedestrians to learn the very basic behavorial policies
  Can be used with pedestrians too
  In the proposed architecture there was topological planner which is similar to the GlobalRoutePlanner in the basicAgent
  The end goal of our project is to build a robust model that can beat the localPlanner ,while the the Global route planner acts like oracle dhoeing the route to the agent.

      
      
