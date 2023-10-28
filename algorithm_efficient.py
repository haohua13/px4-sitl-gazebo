import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import visualization

# Parameters
image_width = 720
image_height = 480

h_fov = 1.5708
aspect_ratio = image_width/image_height
f = (image_width)/(2*math.tan(h_fov/2))
fy = (image_height)/(2*math.tan(f/2))
fx = (image_width)/(2*math.tan(f/2))
print(fx, fy)
v_fov = 2*math.atan(math.tan(h_fov/2)/aspect_ratio)

ox = image_width/2
oy = image_height/2
sx = f/fx
sy = f/fy
A = np.array([
    [sx , 0, ox],
    [0, sy, oy],
    [0, 0, 1]
])

K1 = np.array([
    [fx , 0, ox],
    [0, f, oy],
    [0, 0, 1]
])

# Camera intrinsic matrix K
K = np.array([
    [fx , 0, ox],
    [0, fy, oy],
    [0, 0, 1]
])

# generate initial image points for LK-Optical Flow method
theta_low = (1*np.pi)/12
theta_top = (5*np.pi)/12
phi_low = 0
phi_top = 2*np.pi
N_phi_generate = 30
N_theta_generate  = 10

N = (N_phi_generate+N_theta_generate)*N_theta_generate*2 # number of random points inside the area of integration

def generate_points():
    # Generate N points in a structured manner to cover the specified area
    # Generate evenly spaced phi and theta values
    phi_values = np.linspace(phi_low, phi_top, N_phi_generate)
    theta_values = np.linspace(theta_low, theta_top, N_theta_generate)

    # Create a meshgrid of phi and theta values
    phi_mesh, theta_mesh = np.meshgrid(phi_values, theta_values)

    # Calculate corresponding Cartesian coordinates (x, y)
    x = ((np.sin(theta_mesh) * np.cos(phi_mesh)+1) * (image_width-1) / 2).flatten()
    y = ((np.sin(theta_mesh) * np.sin(phi_mesh)+1) * (image_height-1) / 2).flatten()
    # Stack x and y coordinates to get the points
    points = np.column_stack((x, y))
    return points

def generate_points_inverse():
    # Generate N points in a structured manner to cover the specified area
    # Generate evenly spaced phi and theta values
    phi_values = np.linspace(phi_low, phi_top, N_phi_generate)
    theta_values = np.linspace(theta_low, theta_top, N_theta_generate)

    # Create a meshgrid of phi and theta values
    phi_mesh, theta_mesh = np.meshgrid(phi_values, theta_values)
    
    x = (np.sin(theta_mesh) * np.cos(phi_mesh)+1).flatten()
    y = (np.sin(theta_mesh) * np.sin(phi_mesh)+1).flatten()
    z = np.zeros(x.shape)
    # Stack the 3D coordinates (x, y, z) and add a column of ones for homogeneous coordinates
    points_3d = np.column_stack((x, y, z))
    pixels = np.dot(K, points_3d.T).T
    pixels = np.column_stack((pixels[:, 0], pixels[:, 1]))
    # visualization.draw_catersian_points(pixels[:, 0], pixels[:, 1])
    return pixels


def generate_inside_points():
    # Generate random points inside the defined area of integration
    phi_points = np.random.uniform(phi_low, phi_top, N)
    theta_points = np.random.uniform(theta_low, theta_top, N)

    x = ((np.sin(theta_points) * np.cos(phi_points) + 1) * (image_width - 1) / 2)
    y = ((np.sin(theta_points) * np.sin(phi_points) + 1 )* (image_height - 1) / 2)
    points = np.unique(np.vstack((x, y)).T, axis = 0)
    return points

def convert_to_perspective(p):
    return np.dot(p, np.linalg.inv(K).T)  # Transpose K for proper multiplication

def convert_to_perspective1(p):
    return np.dot(p, np.linalg.inv(K1).T)  # Transpose K for proper multiplication


def detect_corners(frame):
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ## Gen lower mask (0-5) and upper mask (175-180) of RED
    mask1 = cv2.inRange(img_hsv, (0,50,20), (5,255,255))
    mask2 = cv2.inRange(img_hsv, (170,50,20), (180,255,255))
    ## Merge the mask and crop the red regions
    mask = cv2.bitwise_or(mask1, mask2 )
    cropped = cv2.bitwise_and(frame, frame, mask=mask)
    cropped = cv2.bitwise_and(cropped, cropped, mask=mask)
    ## Display
    # cv2.imshow("mask", mask)
    # Convert the cropped image to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    # Detect corners
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=7, L2gradient = True)
    # Find contours of the edges
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the index of the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    not_found = False
    if areas == []:
        P = np.array([ox+1, oy+1, 1])
        q = convert_to_perspective(P)/np.linalg.norm(convert_to_perspective(P))
        print("centroid vector q NOT FOUND: ", q)
        print("pixel image point center P NOT FOUND: ", P)
        not_found = True
        return q, P, not_found
    
    mask = cv2.inRange(img_hsv, (0, 0, 0), (180, 255, 255))
    cropped = cv2.bitwise_and(frame, frame, mask=mask)
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
        
    
    # center = (x+w//2, y+h//2)
    # radius = 2
    # cv2.rectangle(cropped,(x,y),(x+w,y+h),(0,255,0), 2)
    # center of the rectangle is considered the center of the target
    # cv2.circle(cropped, center, radius, (255, 255, 0), 2)
    # cv2.putText(cropped, "Center Of Target", (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    # cv2.putText(cropped, "Detected Square Landmark", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    # cv2.imshow("Target Image", cropped)
    # cv2.imwrite('target.png', cropped)
    
    P = np.array([x+w//2, y+h//2, 1]) # pixel image point center in homogenous coordinates

    # convert to perspective
    p = convert_to_perspective1(P) # centroid vector in perspective projection
    # convert to spherical centroid vector
    norm = np.linalg.norm(p)
    q = p
        
    print("centroid vector q: ", q)
    print("pixel image point center P: ", P)
    return q, P, not_found


def calculate_spherical_OF(optical_flow_vector, pi_planar):
    # optical_flow_vector: optical flow under perspective projectiion 2D 3xN
    # pi_planar: planar image points under perspective projection 2D 3xN
    size = optical_flow_vector.shape[0]
    OF_spherical = np.zeros((size, 3))
    # Compute the norms of planar image point
    norms = np.linalg.norm(pi_planar, axis=1)
    # Normalize each column by its norm to obtain spherical image point
    spherical_prev_pi = pi_planar / norms[:, np.newaxis]
    for i in range(size):
        value = 1/np.linalg.norm(pi_planar[i,:])
        projector = np.eye(3)-np.outer(pi_planar[i, :], pi_planar[i, :].T)
        OF_spherical[i, :] = value*(np.dot(projector, optical_flow_vector[i,:]))
    return OF_spherical, spherical_prev_pi


def skew(vector):
    """
    this function returns a numpy array with the skew symmetric cross product matrix for vector.
    the skew symmetric cross product matrix is defined such that
    np.cross(a, b) = np.dot(skew(a), b)

    :param vector: An array like vector to create the skew symmetric cross product matrix for
    :return: A numpy array of the skew symmetric cross product vector
    """

    return np.array([[0, -vector[2], vector[1]], 
                     [vector[2], 0, -vector[0]], 
                     [-vector[1], vector[0], 0]])


def translational_optical_flow(spherical_prev_pi, OF, R, omega, normal, img):
    # prev_pi: previous spherical image point (p0) 3xN
    # OF: optical flow under spherical projection 3xN
    # R: rotation matrix 3x3 (body to world)
    # omega: angular velocity 3x1
    # normal: 3x1

    # visualization.draw_perspective_points(spherical_prev_pi)
    theta_angle = np.arcsin(np.sqrt(spherical_prev_pi[:, 0]**2 + spherical_prev_pi[:, 1]**2)) # theta
    phi_angle = np.arctan2(spherical_prev_pi[:,1], spherical_prev_pi[:,0]) # phi
    phi_angle = np.mod(phi_angle, 2*np.pi) # normalize to 0 to 2pi

    # visualization.draw_angles(theta_angle, phi_angle)
    print(max(theta_angle))
    print(min(theta_angle))
    print(np.average(theta_angle))


    # theta_0 = theta_low
    # theta_1 = theta_top
    # phi_0 = phi_low
    # phi_1 = phi_top

    # N_theta = N_theta_generate
    # N_phi = N_phi_generate

    theta_0 = min(theta_angle)
    theta_1 = max(theta_angle)
    phi_0 = min(phi_angle)
    phi_1 = max(phi_angle)

    ratio = N_phi_generate/N_theta_generate # ratio of phi to theta

    # to prevent the error of the size of the spherical projection (not all spherical points are projected to the surface since not all OF points are obtained)
    current_size = spherical_prev_pi.shape[0]
    N_theta = int(np.sqrt(current_size/ratio))+1
    N_phi = int(ratio*N_theta)

    # draw area of integration on the image for visualization
    # visualization.draw_area(img, theta_low, theta_top, phi_low, phi_top, image_width, image_height)

    area = (phi_1-phi_0)*(np.cos(theta_0)-np.cos(theta_1)) # area of integration

    # Calculate step sizes for integration 
    dtheta = (theta_1 - theta_0) / (N_theta - 1)
    dphi = (phi_1 - phi_0) / (N_phi - 1)

    integral = 0.0 # total observed optical flow
    # Calculate the integral of the optical flow over the observed area
    count = 0
    for i in range(spherical_prev_pi.shape[0]):
            count = count + 1
            integral += OF[i, :] * np.sin(theta_angle[i])
    
    # average observed optical flow over the surface of the sphere
    print(count)
    phi_w = (integral * dphi * dtheta) / area
    print("phi_w: ", phi_w)

    # D matrix and sigma matrix
    D = np.array([12 * np.cos(2 * theta_0) - 12 * np.cos(2 * theta_1) + np.cos(4 * theta_0) - np.cos(4 * theta_1),
                12 * np.cos(2 * theta_0) - 12 * np.cos(2 * theta_1) + np.cos(4 * theta_0) - np.cos(4 * theta_1),
                16 * np.sin(theta_1)**4 - 16 * np.sin(theta_0)**4])
    sigma = (np.pi / 32) * np.diag(D)

    # Derotation - translational optical flow for visual measurement
    value2  = phi_w+(np.pi/2)*(np.cos(theta_0)-np.cos(theta_1))*np.dot(skew(omega), normal)
    W =  -R.T @ np.linalg.inv(sigma) @ R @ value2
    print("W: ", W)

    return W, phi_w


if __name__ == '__main__':
    generate_points_inverse()