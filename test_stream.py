import cv2


if __name__ == '__main__':
    video = cv2.VideoCapture('udpsrc port=5600 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264" ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink fps-update-interval=1000 sync=false', cv2.CAP_GSTREAMER)
    while True:
        ret, frame = video.read()
        cv2.imshow("Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
