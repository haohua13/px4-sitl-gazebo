import numpy as np
import math
import matplotlib.pyplot as plt
image_width = 720
image_height = 480
mid_x = image_width // 2
mid_y = image_height // 2
a = 200
b = 400
N = 200
# Generate random x and y coordinates between a and b
# Generate random integers between a and b (inclusive)
points_x = np.random.randint(a, b + 1, size=N)  # x-coordinates between a and b
points_y = np.random.randint(a, b + 1, size=N)  # y-coordinates between a and b
# Stack the points to form a 2xN matrix
points = np.vstack((points_x, points_y))

# Stack the points to form a 2xN matrix and then transpose it
points = np.vstack((points_x, points_y)).T

# Convert the points to the desired format
p0 = points.tolist()

print("Points in the desired format:")
print(p0)

N = 200
radius = 200
center_x = image_width // 2
center_y = image_height // 2

rho = np.sqrt(np.random.uniform(0, radius, N))
phi = np.random.uniform(0, 2*np.pi, N)

x = rho * np.cos(phi) + center_x
y = rho * np.sin(phi) + center_y
points = np.vstack((x, y)).T
print(points)


def generate_inside_points(N, phi_0, phi_1, theta_0, theta_1):
    # Generate random points inside the defined area of integration
    phi_points = np.random.uniform(phi_0, phi_1, N)
    theta_points = np.random.uniform(theta_0, theta_1, N)
    x = ((np.sin(theta_points) * np.cos(phi_points) + 1) * (image_width - 1) / 2)
    y = ((np.sin(theta_points) * np.sin(phi_points) + 1 )* (image_height - 1) / 2)
    points = np.unique(np.round(np.vstack((x, y)).T), axis = 0)
    return points

# Example usage
N = 100  # Number of points to generate
phi_0 = 0  # Minimum phi value
phi_1 = 2*np.pi  # Maximum phi value
theta_0 = 0.7*np.pi/12  # Minimum theta value
theta_1 = 2*np.pi/12  # Maximum theta value

# Plot the points as a scatter plot
plt.scatter(points[:, 0], points[:, 1], marker='o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot of points')
plt.show()

