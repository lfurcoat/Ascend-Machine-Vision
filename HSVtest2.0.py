import numpy as np
import cv2

def nothing(x):



    
    pass

# Global variable to store current HSV image
current_hsv = None

def mouse_callback(event, x, y, flags, param):
    """Mouse callback to display HSV values at cursor position"""
    global current_hsv
    if current_hsv is not None and x < current_hsv.shape[1] and y < current_hsv.shape[0]:
        h, s, v = current_hsv[y, x]
        print(f"HSV at ({x}, {y}): H={h}, S={s}, V={v}")
        # Update window title with HSV values
        cv2.setWindowTitle("HSV", f"HSV - H:{h} S:{s} V:{v} at ({x},{y})")

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L-H","Trackbars",0,255,nothing)
cv2.createTrackbar("L-S","Trackbars",0,255,nothing)
cv2.createTrackbar("L-V","Trackbars",0,255,nothing)
cv2.createTrackbar("U-H","Trackbars",0,255,nothing)
cv2.createTrackbar("U-S","Trackbars",0,255,nothing)
cv2.createTrackbar("U-V","Trackbars",0,255,nothing)

def main():
    global current_hsv
    
    cap = cv2.VideoCapture("/home/masslab/Desktop/MachineVision/Paraglider_Video/ParagliderVideo/IMGtest1.MOV")
    
    if (cap.isOpened() == False):
        print("Error with video opening")
        return
    
    # Get total number of frames for looping
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")
    
    # Set up mouse callback for HSV window
    cv2.namedWindow("HSV")
    cv2.setMouseCallback("HSV", mouse_callback)
    
    running = True
    paused = False
    print("Controls: Press SPACEBAR to pause/resume, 'q' to quit")
    while running:
        if not paused:
            success, frame = cap.read() # read video frame
            
            # Check if we've reached the end of the video
            if not success:
                print("End of video reached, restarting...")
                # Reset to beginning of video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue  # Skip this iteration and read the first frame
            
            frame = cv2.resize(frame, (640,480)) # resize resolution
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert to HSV mapping
            
            # Update global HSV for mouse callback
            current_hsv = hsv
            
            # GET TRACKBARS POSITION
            l_h = cv2.getTrackbarPos("L-H","Trackbars")
            l_s = cv2.getTrackbarPos("L-S","Trackbars")
            l_v = cv2.getTrackbarPos("L-V","Trackbars")
            u_h = cv2.getTrackbarPos("U-H","Trackbars")
            u_s = cv2.getTrackbarPos("U-S","Trackbars")
            u_v = cv2.getTrackbarPos("U-V","Trackbars")
            
            # TRACKBAR ARRAY FOR MASKING
            lower = np.array([l_h,l_s,l_v])
            upper = np.array([u_h,u_s,u_v])
            lower_magenta = np.array([141,206,0])
            upper_magenta = np.array([255,255,255])
            lower_yellow = np.array([14,105,70])
            upper_yellow = np.array([31,255,255])
            lower_orange = np.array([0,20,100])
            upper_orange = np.array([20,207,255])
            
            mask1 = cv2.inRange(hsv, lower, upper) # threshold
            #mask1 = cv2.inRange(hsv, lower_magenta, upper_magenta) # threshold Magenta
            mask2 = cv2.inRange(hsv, lower_yellow, upper_yellow) # threshold Yellow
            mask3 = cv2.inRange(hsv, lower_orange, upper_orange) # threshold Orange
            mask_combined = mask1 + mask2 + mask3
            
            contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours3, _ = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            result = cv2.bitwise_and(frame,frame, mask = mask_combined)
            contour_thickness = 1
            
            for contour in contours1:
                cv2.drawContours(frame, [contour], -1, (255,255,255),contour_thickness)
            for contour in contours2:
                cv2.drawContours(frame, [contour], -1, (255,255,255),contour_thickness)
            for contour in contours3:
                cv2.drawContours(frame, [contour], -1, (255,255,255),contour_thickness)
        
        # Display windows (always show, even when paused)
        if 'frame' in locals():
            cv2.imshow("Frame", frame) # Display frame w/o masking
            cv2.imshow("HSV",hsv) # Display frame to HSV mapping
            cv2.imshow("Result", result) # Display threshold frame
        
        # Handle key presses
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break # Break frame loop 'Q'
        elif key == ord(' '):  # Spacebar
            paused = not paused
            if paused:
                print("Video PAUSED - Press SPACEBAR to resume")
            else:
                print("Video RESUMED")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()