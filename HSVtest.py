import numpy as np
import cv2

def nothing(x):
    pass

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L-H","Trackbars",0,255,nothing)
cv2.createTrackbar("L-S","Trackbars",0,255,nothing)
cv2.createTrackbar("L-V","Trackbars",0,255,nothing)
cv2.createTrackbar("U-H","Trackbars",0,255,nothing)
cv2.createTrackbar("U-S","Trackbars",0,255,nothing)
cv2.createTrackbar("U-V","Trackbars",0,255,nothing)

def main():
    
    cap = cv2.VideoCapture("/home/masslab/Desktop/MachineVision/Paraglider_Video/ParagliderVideo/IMG.MOV")
    
    if (cap.isOpened() == False):
        print("Error with video opening")

    running = cap.isOpened()
    while running:
        #cap = cv2.VideoCapture("/home/masslab/Desktop/MachineVision/Paraglider_Video/ParagliderVideo/IMG3.MOV")
 
        success, frame = cap.read() # read video frame
        frame = cv2.resize(frame, (640,480)) # resize resolution
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert to HSV mapping

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
        lower_yellow = np.array([17,181,0])
        upper_yellow = np.array([26,255,255])
        lower_orange = np.array([0,88,0])
        upper_orange = np.array([12,255,255])

        mask1 = cv2.inRange(hsv, lower, upper) # threshold
        #mask1 = cv2.inRange(hsv, lower_magenta, upper_magenta) # threshold Magenta
        mask2 = cv2.inRange(hsv, lower_yellow, upper_yellow) # threshold Yellow
        mask3 = cv2.inRange(hsv, lower_orange, upper_orange) # threshold Orange
        mask_combined = mask1 + mask2 + mask3

        contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours3, _ = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = cv2.bitwise_and(frame,frame, mask = mask1)
        contour_thickness = 1
        for contour in contours1:
            cv2.drawContours(frame, [contour], -1, (255,255,255),contour_thickness)
        for contour in contours2:
            cv2.drawContours(frame, [contour], -1, (255,255,255),contour_thickness)
        for contour in contours3:
            cv2.drawContours(frame, [contour], -1, (255,255,255),contour_thickness)

        if success:
            
            
            cv2.imshow("Frame", frame) # Display frame w/o masking
            cv2.imshow("HSV",hsv) # Display frame to HSV mapping
            cv2.imshow("Result", result) # Display threshold frame
            
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break # Break frame loop 'Q'

        else:
            break # Break upon frame read failure
    cap.release()    
    cv2.destroyAllWindows()   
if __name__ == "__main__":
    main()
    


