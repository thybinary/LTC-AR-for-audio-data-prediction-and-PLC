import numpy as np
import cv2 as cv
from pythonosc import udp_client

# Setup OSC client
osc_ip = "127.0.0.1"  # IP address (use 'localhost' if running locally)
osc_port = 8000       # Port number for OSC communication
client = udp_client.SimpleUDPClient(osc_ip, osc_port)

# Capture video from camera
cap = cv.VideoCapture(0)
ret, frame1 = cap.read()

# Reduce frame size for smoother performance
scale_factor = 0.5  # Resize factor (smaller values make it smoother)
frame1_resized = cv.resize(frame1, None, fx=scale_factor, fy=scale_factor)

prvs = cv.cvtColor(frame1_resized, cv.COLOR_BGR2GRAY)

# Set a counter to throttle OSC message sending
osc_counter = 0
osc_send_frequency = 1  # Send OSC data every frames

while True:
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    # Resize the frame for faster processing
    frame2_resized = cv.resize(frame2, None, fx=scale_factor, fy=scale_factor)
    next_frame = cv.cvtColor(frame2_resized, cv.COLOR_BGR2GRAY)

    # Calculate optical flow
    flow = cv.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.1, 0)
    
    # Convert flow to polar coordinates (magnitude and angle)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

    # Initialize vector field image (empty image with arrows)
    vector_field_img = frame2_resized.copy()

    # Send OSC data less frequently (every osc_send_frequency frames)
    if osc_counter % osc_send_frequency == 0:
        for y in range(0, flow.shape[0], 15):  # Skip some rows for less dense data
            for x in range(0, flow.shape[1], 15):  # Skip some columns
                flow_mag = mag[y, x]
                flow_ang = ang[y, x]
                # Calculate flow vectors for drawing
                fx = int(np.cos(flow_ang) * flow_mag * 10)  # Scale magnitude for visual purposes
                fy = int(np.sin(flow_ang) * flow_mag * 10)

                # Start point (x, y) and end point (x + fx, y + fy)
                start_point = (x, y)
                end_point = (x + fx  , y + fy )

                # Draw arrow on vector field image
                cv.arrowedLine(vector_field_img, start_point, end_point, (0, 255, 0), 1, tipLength=0.3)

                # Send optical flow data via OSC
                client.send_message("/flow", [x, y, float(flow_mag), float(flow_ang)])

    osc_counter += 1
    
    # Display vector field visualization
    cv.imshow('Optical Flow Vector Field', vector_field_img)

    # Keyboard interaction
    k = cv.waitKey(30) & 0xff
    if k == 27:  # ESC key to exit
        break
    elif k == ord('s'):  # Save frame and optical flow vector field visualization
        cv.imwrite('optical_frame.png', frame2_resized)
        cv.imwrite('opticalflow_vector_field.png', vector_field_img)
    
    prvs = next_frame
 
cap.release()
cv.destroyAllWindows()
