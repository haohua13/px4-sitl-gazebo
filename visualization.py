import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import visualization

def draw_spherical_flow(prev_pi, spherical_OF, grid, N_phi, N_theta, phi_0, phi_1, theta_0, theta_1):
    # whole semi hemisphere
    phi_mesh, theta_mesh = np.meshgrid(np.linspace(0, 2*np.pi, N_phi), np.linspace(0, np.pi/2, N_theta))
    x_whole = np.sin(theta_mesh) * np.cos(phi_mesh)
    y_whole = np.sin(theta_mesh) * np.sin(phi_mesh)
    z_whole = -np.cos(theta_mesh)

    # Generate phi and theta meshgrid
    phi_mesh, theta_mesh = np.meshgrid(np.linspace(phi_0, phi_1, N_phi), np.linspace(theta_0, theta_1, N_theta))
    x = np.sin(theta_mesh) * np.cos(phi_mesh)
    y = np.sin(theta_mesh) * np.sin(phi_mesh)
    z = -np.cos(theta_mesh)

    # Create a 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D surface
    h = ax.plot_surface(x, y, z, alpha=0.7, color='b')
    z = ax.plot_surface(x_whole, y_whole, z_whole, alpha=0.1, color='b')
    # Compute the norms of each column
    norms = np.linalg.norm(prev_pi, axis=1)
    # Normalize each column by its norm
    spherical_prev_pi = prev_pi / norms[:, np.newaxis]
    chosen_flows = np.zeros((len(spherical_prev_pi), 3))
    for i in range(len(grid[0])):
        for j in range(len(grid[1])):
            if grid[i][j] != 0:
                chosen_flows[int(grid[i][j]), :] = spherical_OF[int(grid[i][j]), :]

    # only points that are not zero
    chosen_flows = chosen_flows[chosen_flows[:,0] != 0]
    # mesh grid centers
    theta_center = (theta_mesh[:, :] + theta_mesh[:, 0:])/2
    phi_center = (phi_mesh[:, :] + phi_mesh[0:, :])/2
    x = np.sin(theta_center) * np.cos(phi_center)
    y = np.sin(theta_center) * np.sin(phi_center)
    z = -np.cos(theta_center)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    # plot optical flow vectors until size of chosen_flows
    plot = ax.quiver(x[:len(chosen_flows)], y[:len(chosen_flows)], z[:len(chosen_flows)], chosen_flows[:,0], chosen_flows[:,1], chosen_flows[:,2], length=0.2, color='red')
    # match the points
    # Plot the optical flow vectors
    plt.legend(handles=[plot], labels=['Image Kinematics $\mathbf{\dot{p}}$'])
    # Show the plot in latex
    plt.title('Translational Optical Flow Computation $\Theta_0 = 30^\circ \Theta_1 = 60^\circ$', fontsize=12, fontweight='bold')
    # set labels
    ax.grid(which = 'both')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    fig.savefig('matlab_of_final3.png', dpi=300)

def draw_area(img, theta_0, theta_1, phi_0, phi_1, image_width, image_height):
    # Define spherical coordinates for the circle
    phi_range = np.linspace(0, 2 * np.pi, 50)  # 100 points for the circle
    # Convert spherical coordinates to Cartesian coordinates
    cartesian_coordinates = np.zeros((len(phi_range), 2), dtype=np.float32)
    for j, phi in enumerate(phi_range):
        x = int((np.sin(theta_0) * np.cos(phi) + 1) * (image_width - 1) / 2)
        y = int((np.sin(theta_0) * np.sin(phi) + 1) * (image_height - 1) / 2)
        cartesian_coordinates[j] = (x, y)
    # Draw the circle using dots
    for (x, y) in cartesian_coordinates:
        cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)
    # Define spherical coordinates for the circle
    phi_range = np.linspace(0, 2 * np.pi, 100)  # 100 points for the circle
    
    # Convert spherical coordinates to Cartesian coordinates
    cartesian_coordinates = np.zeros((len(phi_range), 2), dtype=np.float32)
    for j, phi in enumerate(phi_range):
        x = int((np.sin(theta_1) * np.cos(phi) + 1) * (image_width - 1) / 2)
        y = int((np.sin(theta_1) * np.sin(phi) + 1) * (image_height - 1) / 2)
        cartesian_coordinates[j] = (x, y)
    # Draw the circle using dots
    for (x, y) in cartesian_coordinates:
        cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)