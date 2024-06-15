# import numpy as np
# import os

# base_dir = "/home/user/Project_test/training_data_new"
# input_files = []
# output_files = []

# # Traverse directories to collect file paths
# for root, dirs, files in os.walk(base_dir):
#     # print(files)
#     for file in files:
#         # print(file)
#         if file.endswith("inputs.npy"):
#             input_files.append(os.path.join(root, file))
#         elif file.endswith("outputs.npy"):
#             output_files.append(os.path.join(root, file))
    
#     if files != []:
#         break


# print(input_files,output_files)

# # Load and concatenate arrays from files
# def load_arrays(file_paths):
#     array_list = []
#     for file_path in file_paths:
#         with open(file_path, "br") as f:
#             while True:
#                 try:
#                     array = np.load(f)
#                     array_list.append(array)
#                 except:
#                     break
#     return np.concatenate(array_list, axis=0)

# input_np = load_arrays(input_files)
# output_np = load_arrays(output_files)

# # Remove the first 400 frames
# input_np = input_np[400:]
# output_np = output_np[400:]

# # Print metrics
# print("Input Shape:", input_np.shape)
# print("Output Shape:", output_np.shape)
# print("Input min axis 0:", input_np.min(axis=0))
# print("Input max axis 0:", input_np.max(axis=0))
# print("First line of input:", input_np[0])
# print("First line of output:", output_np[0])

import numpy as np

#We open the training data
INPUTS_FILE = open("/home/user/Project_test/training_data_new/20240604_2237_npy" + "/inputs.npy","br") 
OUTPUTS_FILE = open("/home/user/Project_test/training_data_new/20240604_2237_npy" + "/outputs.npy","br")  

#We get the data
inputs = []
outputs = []

#We put the data into arrays
while True:
    try:
        input = np.load(INPUTS_FILE)
        inputs.append(input)
    except: 
        break
while True:
    try:
        output = np.load(OUTPUTS_FILE)
        outputs.append(output)
    except: 
        break

# with STRATEGY.scope():
input_np = np.array(inputs)
output_np = np.array(outputs)

print(input_np.shape)    
    
#we close everything
inputs = None
outputs = None

INPUTS_FILE.close()
OUTPUTS_FILE.close()

#We take out the first 400 frames to avoid having the car idle
input_np = input_np[400:,:,:]
output_np = output_np[400:,:]

#Let's print some metrics
print("Input Shape")
print(input_np.shape)
print("-------------------------------------------------------------------------------------")
    
print("Output Shape")
print(output_np.shape)
print("-------------------------------------------------------------------------------------")

print("Input min axis 0")
print(input_np.min(axis=0))
print("-------------------------------------------------------------------------------------")

print("Input max axis 0")
print(input_np.max(axis=0))
print("-------------------------------------------------------------------------------------")

print("First line of input")
print(input_np[0])
print("-------------------------------------------------------------------------------------")

print("Output Shape")
print(output_np.shape)
print("-------------------------------------------------------------------------------------")

print("First line of output")
print(output_np[0])
print("-------------------------------------------------------------------------------------")