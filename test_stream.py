import cv2
pipeline = "udpsrc port=5600 caps='application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264' ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink fps-update-interval=1000 sync=false"

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)


if not cap.isOpened():
    print('Error: Unable to open pipeline')
    exit()

while True:
    # Read a frame from the pipeline
    ret, frame = cap.read()

    # Check if a frame was successfully read
    if not ret:
        print('Error: Unable to read frame')
        break

    # Display the frame
    cv2.imshow('Frame', frame)

    # Check if the user pressed 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
    cv2.destroyAllWindows()