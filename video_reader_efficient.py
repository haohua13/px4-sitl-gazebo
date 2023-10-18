#!/usr/bin/env python

import cv2
import numpy as np
import algorithm
import matplotlib.pyplot as plt
import time
from scipy.spatial.transform import Rotation as R
import scipy

# check if the points are in the image
def checkedTrace(img0, img1, p0, back_threshold = 1.0):
    p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    status = d < back_threshold
    return p1, status
    
if __name__ == '__main__':
    cap = cv2.VideoCapture('automatic_landing.mp4')
    # Previous image frameqq
    ret, prev_frame = cap.read()
    lk_params = dict( winSize  = (10, 10),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Generate initial points for LK  Metqhod (points on a grid)
    prev_pi = algorithm.generate_points()
    # Generate initial points for LK method (random points)
    # qprev_pi = algorithm.generate_inside_points()
    count = 0
    # sample time between frames
    interval = 3
    sample_time = interval/30
    q_save = []
    iteration = 0
    W_save = []
    prev_W = np.zeros(3)
    prev_q = np.zeros(3)
    while True:
    # Wait for the next frame
        ret, frame = cap.read()
        if ret:
            count += interval # i.e. at 30 fps, this advances count/30 seconds per iteration
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        if prev_frame is not None and frame is not None:
            start = time.time()
            # Convert frames to grayscale for optical flow
            gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Create a mask image for drawing purposes
            mask = np.zeros_like(prev_frame)

            prev_pi_reshaped = prev_pi.reshape(-1, 1, 2).astype(np.float32)

            pi, st1, err1 = cv2.calcOpticalFlowPyrLK(gray_prev,
                                                gray_frame,
                                                prev_pi_reshaped, None,
                                                **lk_params)
            # Check traces of image pointsq
            p2, trace_status = checkedTrace(gray_prev, gray_frame, prev_pi_reshaped)
            filtered_pi = p2[trace_status].copy()
            filtered_prev_pi = prev_pi[trace_status].copy()
            filtered_pi = filtered_pi.reshape(-1, 2).astype(np.float32)
            filtered_pi = np.round(filtered_pi)
            filtered_prev_pi = filtered_prev_pi.reshape(-1, 2).astype(np.float32)

            print('Chosen Image Points: ', filtered_pi.shape)
            
            # convert to homogeneous coordinates
            size =filtered_prev_pi.shape[0]
            z_axis = np.ones((size, 1))
            filtered_prev= np.concatenate((filtered_prev_pi, z_axis), axis=1)
            filtered_current = np.concatenate((filtered_pi, z_axis), axis=1)
            # convert to 3-D perspective image points
            filtered_perspective_prev = algorithm.convert_to_perspective(filtered_prev) # q_prev
            filtered_perspective_current = algorithm.convert_to_perspective(filtered_current) # q_curent
            filtered_OF = (filtered_perspective_current - filtered_perspective_prev)/sample_time # qdot
            filtered_spherical_OF = algorithm.calculate_spherical_OF(filtered_OF, filtered_perspective_prev) #pdot

            # calculates the translational optical flow for visual velocity measurement information
            time_w = time.time()
            W, phi_w = algorithm.translational_optical_flow(filtered_perspective_prev, filtered_spherical_OF, np.eye(3), np.array([0, 0, 0]), np.eye(3)*np.array([0, 0, 1]), frame)
            print('Time for W: ', time.time() - time_w)
            
            # calculates the centroid vector for visual position measurement information
            q, P = algorithm.detect_corners(frame)
            q[0] = -q[0]
            q[1] = -q[1]
            # # ignore peaks of the translational optical flow and use previous value
            # if (np.linalg.norm(W[2] - prev_W[2])>0.2):
            #     W = prev_W
            #     q = prev_q
            # else:
            #     prev_W = W
            #     prev_q = q

            # save the translational optical flow and centroid vector
            W_save = np.append(W_save, W)
            q_save = np.append(q_save, q)

            # draw the tracks
            for i, (new, old) in enumerate(zip(filtered_pi,
                                            filtered_prev_pi)):
                a, b = new.ravel()
                c, d = old.ravel()

                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (255, 0, 0), 2) # draws optical flow lines

                frame = cv2.circle(frame, (int(a), int(b)), 2, (255, 0, 255), -1) # draws image points

            img = cv2.add(frame, mask)

            cv2.putText(img, "Image points", (550, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(img, "Optical flow", (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.circle(img, (int(P[0]), int(P[1])), 2, (255, 255, 0), 2)
            cv2.putText(img, "Center Of Target", (550, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img, "Area of Integration", (550, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('Optical Flow', img)
            cv2.imwrite('optical_flow.png', img)
            end = time.time()
            print('Total Computation Time: ', end-start)

        else:
            # cv2.imshow('Image', frame)
            pass

        # Update the previous frame for the next iteration
        prev_frame = frame.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Ensure the length of W_save and q_save is a multiple of 3
        remainder_W = len(W_save) % 3
        remainder_q = len(q_save) % 3
        if remainder_W != 0:
            W_save = np.append(W_save, [0] * (3 - remainder_W))  # Pad with zeros to make length a multiple of 3

        if remainder_q != 0:
            q_save = np.append(q_save, [0] * (3 - remainder_q))  # Pad with zeros to make length a multiple of 3

        # Save the reshaped arrays to files
        W_save_reshaped = W_save.reshape(-1, 3)
        q_save_reshaped = q_save.reshape(-1, 3)
        np.save('W_save_reshaped.npy', W_save_reshaped)
        np.save('q_save_reshaped.npy', q_save_reshaped)
        scipy.io.savemat('q_estimation.mat', mdict={'q': q_save_reshaped})
        scipy.io.savemat('W_estimation.mat', mdict={'W': W_save_reshaped})
    print('Finished Landing!')
