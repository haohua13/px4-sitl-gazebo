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
f = (image_width)/(2*math.tan(h_fov/2))
fy = (image_height)/(2*math.tan(f/2))
fx = (image_width)/(2*math.tan(f/2))
print(fx, fy)
ox = image_width/2
oy = image_height/2

# Camera intrinsic matrix K
K = np.array([
    [-fx , 0, ox],
    [0, -fy, oy],
    [0, 0, 1]
])

# generate initial image points
theta_low = (0.5*np.pi)/12
theta_top = (5*np.pi)/12
phi_low = 0
phi_top = 2*np.pi
N_phi_generate = 40
N_theta_generate = 20


# for the section of the area of integration
N_phi =  10 # grid size on the surface of the sphere in phi direction
N_theta = 5 # grid size on the surface of the sphere in theta direction
phi_0 = 0 
phi_1 = 2*np.pi
theta_0 = 0.8*np.pi/12 # increasing would mean ignoring OF closer to the center which is mostly 2-D
theta_1 = 3.4*np.pi/12 # related to the FOV of the camera, can't be too high

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

    # Stack x and y coordinates to get the points and round to nearest integer
    points = np.unique(np.round(np.column_stack((x, y))), axis = 0)
    return points

def generate_inside_points():
    # Generate random points inside the defined area of integration
    phi_points = np.random.uniform(phi_low, phi_top, N)
    theta_points = np.random.uniform(theta_low, theta_top, N)

    x = ((np.sin(theta_points) * np.cos(phi_points) + 1) * (image_width - 1) / 2)
    y = ((np.sin(theta_points) * np.sin(phi_points) + 1 )* (image_height - 1) / 2)
    points = np.unique(np.round(np.vstack((x, y)).T), axis = 0)
    return points

def convert_to_perspective(p):
    return np.dot(p, np.linalg.inv(K).T)  # Transpose K for proper multiplication


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
    if areas == []:
        P = np.array([0, 0, 1])
        return np.array([0.01, 0.01, 1]), P
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

    P = np.array([x+w//2, y+h//2, 1]) # perspective image point
    q = convert_to_perspective(P) # centroid vector
    if q == []:
        q = np.array([0.01, 0.01, 1])
        
    print("centroid vector q: ", q)
    print("perspective image point center P: ", P)
    return q, P


def calculate_spherical_OF(optical_flow_vector, pi_planar):
    size = optical_flow_vector.shape[0]
    OF_spherical = np.zeros((3, size))
    for i in range(size):
        value = 1/np.linalg.norm(pi_planar[i,:])
        projection = np.eye(3)-pi_planar[i, :]*np.transpose(pi_planar[i, :])
        OF_spherical[:,i] = value*np.dot(projection, optical_flow_vector[i,:])
    return OF_spherical


def translational_optical_flow(prev_pi, OF, wRb, omega, normal, img):
    # prev_pi: previous perspective image point (p0) 3xN
    # OF: optical flow under spherical projection 3xN
    # wRb: rotation matrix 3x3
    # omega: angular velocity 3x1
    # normal: 3x1

    # 1. spherical projection
    # Compute the norms of each column
    norms = np.linalg.norm(prev_pi, axis=1)
    # Normalize each column by its norm
    spherical_prev_pi = prev_pi / norms[:, np.newaxis]
    theta_angle = np.arcsin(np.sqrt(spherical_prev_pi[:, 0]**2 + spherical_prev_pi[:, 1]**2)) # theta
    phi_angle = np.arctan(spherical_prev_pi[:,1]/spherical_prev_pi[:,0]) # phi
    phi_angle = (phi_angle +np.pi*2) % (2*np.pi)
    print(max(theta_angle))
    print(min(theta_angle))
    print(np.average(theta_angle))

    # draw area of integration on the image for visualization
    visualization.draw_area(img, theta_0, theta_1, phi_0, phi_1, image_width, image_height)

    area = (phi_1-phi_0)*(np.cos(theta_0)-np.cos(theta_1)) # area of integration

    phi_lin = np.linspace(phi_0, phi_1, N_phi) # phi
    theta_lin = np.linspace(theta_0, theta_1, N_theta) # theta

    # Create meshgrid
    theta, phi = np.meshgrid(theta_lin, phi_lin)

    # Calculate step sizes
    dtheta = (theta_1 - theta_0) / (N_theta - 1)
    dphi = (phi_1 - phi_0) / (N_phi - 1)

    # Initialize arrays
    # original_grid = np.zeros(theta.shape) # grid of detected image points
    integral = 0.0 # total observed optical flow

    # Calculate the integral of the optical flow over the observed area
    # brute-force solution to find closest image point to each grid point on the surface of the sphere
    count = 0
    time_test = time.time()
    OF = np.transpose(OF)
    for row in range(N_phi):
        for column in range(N_theta):
            min_distance = 0.1 # minimum distance tolerance (5 approx degrees)
            min_distance_2 = 1.5 # minimum distance tolerance
            closest_point = -1
            # check for each image point if it is the closest to the grid point
            for i in range(prev_pi.shape[0]):
                current_theta = theta_angle[i]  # theta
                current_phi = phi_angle[i]  # phi
                # obtain distance between the current image point and the grid point
                distance = np.sqrt((current_theta - theta[row, column]) ** 2)
                distance2 = np.sqrt((current_phi - phi[row, column]) ** 2)
                if distance < min_distance and distance2 < min_distance_2:
                    closest_point = i
                    min_distance = distance
                    min_distance_2 = distance2
            # add the optical flow of the closest image point to the integral
            if closest_point != -1:
                # print(theta_angle[closest_point], phi_angle[closest_point], theta[row, column], phi[row, column])
                integral += OF[closest_point, :] * np.sin(theta[row, column])
                count = count + 1

    print(count)
    print("time for brute force: ", time.time() - time_test)

    # average observed optical flow over the surface of the sphere
    phi_w = (integral * dphi * dtheta) / area
    print("phi_w: ", phi_w)

    # D matrix and sigma matrix
    D = np.array([12 * np.cos(2 * theta_0) - 12 * np.cos(2 * theta_1) + np.cos(4 * theta_0) - np.cos(4 * theta_1),
                12 * np.cos(2 * theta_0) - 12 * np.cos(2 * theta_1) + np.cos(4 * theta_0) - np.cos(4 * theta_1),
                16 * np.sin(theta_1)**4 - 16 * np.sin(theta_0)**4])
    sigma = (np.pi / 32) * np.diag(D)

    # Derotation - translational optical flow for visual measurement
    W = np.dot(np.dot(wRb, np.linalg.inv(sigma)), np.dot(wRb.T, (phi_w + (np.pi / 2) * (np.cos(2 * theta_0) - np.cos(2 * theta_1)) * np.dot(omega, normal))))
    print("W: ", W)

    # in case we want to draw the spherical flow
    # visualization.draw_spherical_flow(prev_pi, OF, original_grid, N_phi, N_theta, phi_0, phi_1, theta_0, theta_1)

    return W, phi_w
