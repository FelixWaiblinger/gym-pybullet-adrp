import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import yaml
import matplotlib.cm as cm

# Load the gate configurations from the YAML file
with open('/home/amin/Documents/repos/lsy_drone_racing/config/twogates.yaml', 'r') as file:
    gate_config = yaml.safe_load(file)

# Extract the gate positions from the configuration
gate_positions = gate_config['gates']

# Read trajectory data from CSV for two different trajectories
data1 = "/home/amin/Documents/repos/lsy_drone_racing/plots/rl_agent.csv"
data2 = "/home/amin/Documents/repos/lsy_drone_racing/plots/hardcoded.csv"
trajectory_data1 = pd.read_csv(data1)
trajectory_data2 = pd.read_csv(data2)

# Assign columns names
trajectory_data1.columns = ['Time', 'x', 'y', 'z', 'vx', 'vy', 'vz']
trajectory_data2.columns = ['Time', 'x', 'y', 'z', 'vx', 'vy', 'vz']

# Calculate the speed
trajectory_data1['speed'] = np.sqrt(trajectory_data1['vx']**2 + trajectory_data1['vy']**2 + trajectory_data1['vz']**2)
trajectory_data2['speed'] = np.sqrt(trajectory_data2['vx']**2 + trajectory_data2['vy']**2 + trajectory_data2['vz']**2)

# Create a figure
fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the first trajectory with speed as color
sc1 = ax.scatter(trajectory_data1['x'], trajectory_data1['y'], trajectory_data1['z'], c=trajectory_data1['speed'], cmap='RdPu', s=10)

# Plot the second trajectory with speed as color
sc2 = ax.scatter(trajectory_data2['x'], trajectory_data2['y'], trajectory_data2['z'], c=trajectory_data2['speed'], cmap='YlGn', s=10)

# Set labels for the 3D plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Change the angle of view for the 3D plot
ax.view_init(elev=20, azim=10)

# Plot the gates as hollow rectangles in 3D with gradient color map and thicker lines
color_map = cm.get_cmap('plasma')  # You can change 'plasma' to any other color map
line_thickness = 3  # Set the line thickness

# Normalize gate index to [0, 1] for color mapping
gate_indices = np.arange(len(gate_positions))
norm = plt.Normalize(gate_indices.min(), gate_indices.max())

for j, gate in enumerate(gate_positions):
    x = gate[0]
    y = gate[1]
    yaw = gate[5]
    half_length = 0.1875
    if gate[6] == 0:
        height = 1
    else:
        height = 0.525

    delta_x = half_length * np.cos(yaw)
    delta_y = half_length * np.sin(yaw)

    # Define the four corners of the rectangle
    corners = np.array([
        [x - delta_x, y - delta_y, height - half_length],
        [x + delta_x, y + delta_y, height - half_length],
        [x + delta_x, y + delta_y, height + half_length],
        [x - delta_x, y - delta_y, height + half_length]
    ])

    # Get color for current gate based on index
    color = color_map(norm(0))

    # Plot the edges of the rectangle
    for i in range(4):
        ax.plot([corners[i][0], corners[(i+1) % 4][0]], [corners[i][1], corners[(i+1) % 4][1]], 
                [corners[i][2], corners[(i+1) % 4][2]], color=color, linewidth=line_thickness)

# Add one color bar for each trajectory
cbar1 = fig.colorbar(sc1, ax=ax, shrink=0.5, aspect=10)
cbar1.set_label('Speed (m/s) - RL Controller')

cbar2 = fig.colorbar(sc2, ax=ax, shrink=0.5, aspect=10)
cbar2.set_label('Speed (m/s) - Hardcoded Controller')

# Draw the drones at the starting points of the trajectories
start_x1, start_y1, start_z1 = trajectory_data1[['x', 'y', 'z']].iloc[0]
start_x2, start_y2, start_z2 = trajectory_data2[['x', 'y', 'z']].iloc[0]

# Define the propellers and body frame with smaller size
propeller_radius = 0.05  # Smaller propellers
propeller_distance = 0.05  # Smaller distance between propellers

def draw_drone(ax, start_x, start_y, start_z, color):
    # Define the positions of the propellers
    propellers = np.array([
        [propeller_distance, propeller_distance, 0],
        [-propeller_distance, propeller_distance, 0],
        [-propeller_distance, -propeller_distance, 0],
        [propeller_distance, -propeller_distance, 0]
    ])

    # Draw the propellers as circles
    for prop in propellers:
        phi = np.linspace(0, 2 * np.pi, 100)
        px = start_x + prop[0] + propeller_radius * np.cos(phi)
        py = start_y + prop[1] + propeller_radius * np.sin(phi)
        pz = start_z + prop[2] + np.zeros_like(phi)
        ax.plot(px, py, pz, color=color)

    # Draw the body frame in black with shorter axes
    ax_length = 0.1  # Make axis shorter
    ax.quiver(start_x, start_y, start_z, ax_length, 0, 0, color='red', linewidth=2)
    ax.quiver(start_x, start_y, start_z, 0, ax_length, 0, color='green', linewidth=2)
    ax.quiver(start_x, start_y, start_z, 0, 0, ax_length, color='blue', linewidth=2)

# Draw the drones with distinct colors
draw_drone(ax, start_x1, start_y1, start_z1, 'black')  # RL Controller drone in red
draw_drone(ax, start_x2, start_y2, start_z2, 'blue')   # Hardcoded Controller drone in blue

# Show the plot
plt.legend()
plt.show()
