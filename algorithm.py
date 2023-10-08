import numpy as np
import math
import cv2


# Parameters
image_width = 720
image_height = 480

h_fov = 1.5708
f = (image_width)/(2*math.tan(h_fov/2))
ox = image_width/2
oy = image_height/2

# Camera intrinsic matrix K
K = np.array([
    [-f , 0, ox],
    [0, -f, oy],
    [0, 0, 1]
])

# define section of integration of sphere
theta_0 = (1*np.pi)/12; # increasing would mean ignoring OF closer to the center
theta_1 = (3*np.pi)/12; # related to the FOV of the cameras
phi_0 = 0
phi_1 = 2*np.pi

def generate_inside_points(N):
    # Generate random points inside the defined area of integration
    phi_points = np.random.uniform(phi_0, phi_1, N)
    theta_points = np.random.uniform(theta_0, theta_1, N)
    x = ((np.sin(theta_points) * np.cos(phi_points) + 1) * (image_width - 1) / 2)
    y = ((np.sin(theta_points) * np.sin(phi_points) + 1 )* (image_height - 1) / 2)
    points = np.unique(np.round(np.vstack((x, y)).T), axis = 0)
    return points

def convert_to_perspective(p):
    return np.dot(p, np.linalg.inv(K).T)  # Transpose K for proper multiplication

# def generate_circular_points(N):
#     radius = 350
#     radius_y = 230
#     center_x = image_width / 2
#     center_y = image_height / 2

#     rho = np.random.randint(15, radius, N)
#     rho_y = np.random.randint(15, radius_y, N)
#     phi = np.random.random(N) * 2 * np.pi

#     x = np.round(rho * np.cos(phi) + center_x)
#     y = np.round(rho_y * np.sin(phi) + center_y)  # Use rho_y for y coordinates

#     points = np.vstack((x, y)).T
#     # ensure only unique points are returned
#     unique_points = np.unique(points, axis=0)
#     return points

# def generate_initial_points(N):
#     # generate random points for the initial frame to track
#     # return: 3xN matrix of points
#     # Calculate the middle of the image
#     mid_x = image_width // 2
#     mid_y = image_height // 2
#     # Generate random points around the middle
#     points_x = np.random.randint(mid_x - mid_x/2, mid_x + mid_x/2, size=N)
#     points_y = np.random.randint(mid_y - mid_y/2, mid_y + mid_y/2, size=N)
#     # Stack the points to form a 2xN matrix
#     points = np.vstack((points_x, points_y)).T
#     return points

def draw_area(img, theta_0, theta_1):
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
        return 0, P
    mask = cv2.inRange(img_hsv, (0, 0, 0), (180, 255, 255))
    cropped = cv2.bitwise_and(frame, frame, mask=mask)
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(cropped,(x,y),(x+w,y+h),(0,255,0), 2)

    # center of the rectangle is considered the center of the target
    center = (x+w//2, y+h//2)
    radius = 2
    cv2.circle(cropped, center, radius, (255, 255, 0), 2)
    cv2.putText(cropped, "Center Of Target", (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(cropped, "Detected Square Landmark", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("Target Image", cropped)
    cv2.imwrite('target.png', cropped)

    P = np.array([x+w//2, y+h//2, 1]) # perspective image point
    q = convert_to_perspective(P) # centroid vector
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

    elevation = np.mod(np.arccos(spherical_prev_pi[:,2]), np.pi) # theta
    azimuth = np.mod(np.arctan(spherical_prev_pi[:,1]/spherical_prev_pi[:,0]), 2*np.pi) # phi

    # draw area of integration on the image
    draw_area(img, theta_0, theta_1)

    area = (phi_1-phi_0)*(np.cos(theta_0)-np.cos(theta_1)) # area of integration

    N_phi =  30 # grid size on the surface of the sphere in phi direction
    N_theta = 30 # grid size on the surface of the sphere in theta direction

    phi_lin = np.linspace(phi_0, phi_1, N_phi) # phi
    theta_lin = np.linspace(theta_0, theta_1, N_theta) # theta

    # Create meshgrid
    theta, phi = np.meshgrid(theta_lin, phi_lin)

    # Calculate step sizes
    dtheta = (theta_1 - theta_0) / (N_theta - 1)
    dphi = (phi_1 - phi_0) / (N_phi - 1)

    # Initialize arrays
    original_grid = np.zeros(theta.shape) # grid of detected image points
    integral = 0.0 # total observed optical flow

    angle1_indices = np.random.permutation(len(elevation))
    angle1 = elevation[angle1_indices]

    angle2_indices = np.random.permutation(len(azimuth))
    angle2 = azimuth[angle2_indices]
    print(max(angle1))
    print(min(angle1))
    print(np.average(angle1))

    # Calculate pairwise distances between grid points and image points
    theta_expand = theta[..., np.newaxis]
    phi_expand = phi[..., np.newaxis]
    distance_matrix = np.sqrt((theta_expand - angle1)**2 + (phi_expand - angle2)**2)

    # Calculate the integral of the optical flow over the observed area
    # brute-force solution to find closest image point to each grid point
    count = 0
    OF = np.transpose(OF)

    for row in range(N_phi):
        for column in range(N_theta):
            min_distance = float('inf')
            closest_point = 0
            for i in range(prev_pi.shape[1]):
                current_theta = angle1[i]  # theta
                current_phi = angle2[i]  # phi
                distance = np.sqrt((current_theta - theta[row, column]) ** 2 + (current_phi - phi[row, column]) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = i
            original_grid[row, column] = closest_point
            angle1[closest_point] = np.nan
            angle2[closest_point] = np.nan
            integral += OF[closest_point, :] * np.sin(theta[row, column])
            count += 1

    # average observed optical flow
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

    return W, phi_w

