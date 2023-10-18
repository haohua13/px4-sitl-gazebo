#!/usr/bin/env python

import cv2
import gi
import numpy as np
import algorithm
import matplotlib.pyplot as plt
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import server
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

class Video():
    """BlueRov video capture class constructor

    Attributes:
        port (int): Video UDP port
        video_codec (string): Source h264 parser
        video_decode (string): Transform YUV (12bits) to BGR (24bits)
        video_pipe (object): GStreamer top-level pipeline
        video_sink (object): Gstreamer sink element
        video_sink_conf (string): Sink configuration
        video_source (string): Udp source ip and port
    """

    def __init__(self, port=5600):
        """Summary

        Args:
            port (int, optional): UDP port
        """

        Gst.init(None)

        self.port = port
        self._frame = None
        self.prev_frame = None

        self.prev_time = 0
        self.time = 0

        # [Software component diagram](https://www.ardusub.com/software/components.html)
        # UDP video stream (:5600)
        self.video_source = 'udpsrc port={}'.format(self.port)
        # [Rasp raw image](http://picamera.readthedocs.io/en/release-0.7/recipes2.html#raw-image-capture-yuv-format)
        # Cam -> CSI-2 -> H264 Raw (YUV 4-4-4 (12bits) I420)
        self.video_codec = '! application/x-rtp, payload=96 ! rtph264depay ! h264parse ! avdec_h264'
        # Python don't have nibble, convert YUV nibbles (4-4-4) to OpenCV standard BGR bytes (8-8-8)
        self.video_decode = \
            '! decodebin ! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert'
        # Create a sink to get data
        self.video_sink_conf = \
            '! appsink emit-signals=true sync=false max-buffers=2 drop=true'

        self.video_pipe = None
        self.video_sink = None

        self.run()

    def start_gst(self, config=None, useCuda=True):
        """ Start gstreamer pipeline and sink
        Pipeline description list e.g:
            [
                'videotestsrc ! decodebin', \
                '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                '! appsink'
            ]

        Args:
            config (list, optional): Gstreamer pileline description list
        """

        if not config:
            config = \
                [
                    'videotestsrc ! decodebin',
                    '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                    '! appsink'
                ]
            
        command = ' '.join(config)
        self.video_pipe = Gst.parse_launch(command)
        self.video_pipe.set_state(Gst.State.PLAYING)
        self.video_sink = self.video_pipe.get_by_name('appsink0')

    @staticmethod
    def gst_to_opencv(sample):
        """Transform byte array into np array

        Args:
            sample (TYPE): Description

        Returns:
            TYPE: Description
        """
        buf = sample.get_buffer()
        caps = sample.get_caps()
        array = np.ndarray(
            (
                caps.get_structure(0).get_value('height'),
                caps.get_structure(0).get_value('width'),
                3
            ),
            buffer=buf.extract_dup(0, buf.get_size()), dtype=np.uint8)
        return array

    def frame(self):
        """ Get Frame

        Returns:
            iterable: bool and image frame, cap.read() output
        """
        return self._frame
    
    def get_time(self):
        return self.prev_time 

    def frame_available(self):
        """Check if frame is available

        Returns:
            bool: true if frame is available
        """
        return type(self._frame) != type(None)

    def run(self):
        """ Get frame to update _frame
        """

        self.start_gst(
            [
                self.video_source,
                self.video_codec,
                self.video_decode,
                self.video_sink_conf
            ])

        self.video_sink.connect('new-sample', self.callback)

    def callback(self, sink):
        sample = sink.emit('pull-sample')
        new_frame = self.gst_to_opencv(sample)
        self._frame = new_frame
        current_time = time.time()
        if self.prev_time != 0:
            self.time = current_time-self.prev_time
        self.prev_time = current_time
        return Gst.FlowReturn.OK

if __name__ == '__main__':
    # Create the video object
    # Add port= if is necessary to use a different one
    video = Video()

     # Previous image frame
    prev_frame = None
    # current image frame
    frame = None
    
    lk_params = dict( winSize  = (10, 10),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Generate initial points for LK  Method (points on a grid)
    prev_pi = algorithm.generate_points()
    # Generate initial points for LK method (random points)
    # prev_pi = algorithm.generate_inside_points(1000)
    # sample time between frames
    interval = 2
    sample_time = interval/30
    iteration = 0
    prev_W = np.zeros(3)
    prev_q = np.zeros(3)
    # initialize inertial data
    angular_velocity = np.array([0, 0, 0])
    rotation_matrix = np.eye(3)

    while True:
        # receive UDP messages to obtain inertial data (orientation and angular velocity)
        output =server.receive_message()
        if(output != None):
            euler = np.array(output[0:3])
            angular_velocity = np.array(output[3:6])
            # convert to rotation matrix
            R = R.from_euler('XYZ', euler, degrees = True)
            rotation_matrix = np.array(R.as_matrix())

            # print('Rotation: ', rotation_matrix)
            # print('Angular Velocity:', angular_velocity)

        # Wait for the next frame. Checks if frame is available, if not continue
        if not video.frame_available():
            continue
        frame = video.frame()
        # cv2.imshow(f'Image {iteration}', frame)
        if prev_frame is not None and iteration == interval:       
            start = time.time()
            iteration = 0
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
            # Check traces
            p2, trace_status = checkedTrace(gray_prev, gray_frame, prev_pi_reshaped)
            filtered_pi = p2[trace_status].copy()
            filtered_prev_pi = prev_pi[trace_status].copy()
            filtered_pi = filtered_pi.reshape(-1, 2).astype(np.float32)
            filtered_prev_pi = filtered_prev_pi.reshape(-1, 2).astype(np.float32)

            print('Chosen Image Points: ', filtered_pi.shape)

            # convert to homogeneous coordinates
            size =filtered_prev_pi.shape[0]
            z_axis = np.ones((size, 1))
            filtered_prev= np.concatenate((filtered_prev_pi, z_axis), axis=1)
            filtered_current = np.concatenate((filtered_pi, z_axis), axis=1)
            # convert to 3-D perspective image points
            filtered_perspective_prev = algorithm.convert_to_perspective(filtered_prev)
            filtered_perspective_current = algorithm.convert_to_perspective(filtered_current)
            filtered_OF = (filtered_perspective_current - filtered_perspective_prev)/(sample_time)
            filtered_spherical_OF = algorithm.calculate_spherical_OF(filtered_OF, filtered_perspective_prev)

            W, phi_w = algorithm.translational_optical_flow(filtered_perspective_prev, filtered_spherical_OF, rotation_matrix, angular_velocity, rotation_matrix*np.array([0, 0, 1]), frame)
            if W.all() == 0 :
                W = prev_W

            # calculates the centroid vector for visual position measurement information
            q, P = algorithm.detect_corners(frame)

            # W_save = np.append(W_save, W)
            # q_save = np.append(q_save, q)
            q[1] = -q[1]
            q[2] = -q[2]

            # # ignore peaks of the translational optical flow and use previous value
            # if (np.linalg.norm(W[2] - prev_W[2])>0.2):
            #     W = prev_W
            #     q = prev_q
            # else:
            #     prev_W = W
            #     prev_q = q

            

            # send udp message to matlab (works!)
            server.send_message(W, q)

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
            # cv2.imwrite('optical_flow.png', img)
            # print(prev_pi.shape)

            end = time.time()
            
            print('Time: ', end-start)
        else:
            pass
            # cv2.imshow('Image', frame)

            # Update the previous frame for the next iteration

        # previous frame is at iteration 0 and every interval frames
        if iteration == 0:
            prev_frame = frame.copy()
        iteration += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
