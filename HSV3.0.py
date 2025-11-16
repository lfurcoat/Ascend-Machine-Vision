import cv2
import numpy as np


def nothing(x):
    pass

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L-H","Trackbars",0,255,nothing)
cv2.createTrackbar("L-S","Trackbars",0,255,nothing)
cv2.createTrackbar("L-V","Trackbars",0,255,nothing)
cv2.createTrackbar("U-H","Trackbars",0,255,nothing)
cv2.createTrackbar("U-S","Trackbars",0,255,nothing)
cv2.createTrackbar("U-V","Trackbars",0,255,nothing)

# Adaptive options that actually help with HSV thresholding
cv2.createTrackbar("Brightness Adapt","Trackbars",0,1,nothing)  # Auto-adjust V channel
cv2.createTrackbar("Hist Equalize","Trackbars",0,1,nothing)    # Histogram equalization
cv2.createTrackbar("CLAHE","Trackbars",0,1,nothing)           # Contrast enhancement

def adaptive_hsv_adjustment(hsv, enable_brightness_adapt, enable_hist_eq, enable_clahe):
    """Improve HSV thresholding with adaptive techniques"""
    
    if not (enable_brightness_adapt or enable_hist_eq or enable_clahe):
        return hsv  # No changes
    
    # Work with a copy
    hsv_adapted = hsv.copy()
    
    # Extract V channel (brightness)
    v_channel = hsv_adapted[:, :, 2]
    
    if enable_hist_eq:
        # Histogram equalization on V channel
        v_channel = cv2.equalizeHist(v_channel)
    
    if enable_clahe:
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        v_channel = clahe.apply(v_channel)
    
    if enable_brightness_adapt:
        # Auto-adjust brightness based on mean
        mean_brightness = np.mean(v_channel)
        if mean_brightness < 100:  # Dark image
            v_channel = cv2.add(v_channel, 30)  # Brighten
        elif mean_brightness > 180:  # Bright image
            v_channel = cv2.subtract(v_channel, 20)  # Darken slightly
    
    # Put the modified V channel back
    hsv_adapted[:, :, 2] = v_channel
    
    return hsv_adapted

def main():
    
    cap = cv2.VideoCapture("/home/masslab/Desktop/MachineVision/Paraglider_Video/ParagliderVideo/IMG.MOV")
    
    if (cap.isOpened() == False):
        print("Error with video opening")

    running = cap.isOpened()
    while running:
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
        
        # Get adaptive enhancement options
        brightness_adapt = cv2.getTrackbarPos("Brightness Adapt","Trackbars")
        hist_eq = cv2.getTrackbarPos("Hist Equalize","Trackbars")
        clahe = cv2.getTrackbarPos("CLAHE","Trackbars")

        # Apply adaptive enhancements to HSV
        hsv_enhanced = adaptive_hsv_adjustment(hsv, brightness_adapt, hist_eq, clahe)

        # TRACKBAR ARRAY FOR MASKING
        lower = np.array([l_h,l_s,l_v])
        upper = np.array([u_h,u_s,u_v])
        lower_magenta = np.array([154,201,44])
        upper_magenta = np.array([194,255,255])
        lower_yellow = np.array([18,124,130])
        upper_yellow = np.array([30,255,255])
        lower_orange = np.array([0,88,0])
        upper_orange = np.array([12,255,255])

        # Use enhanced HSV for thresholding
        mask1 = cv2.inRange(hsv_enhanced, lower_magenta, upper_magenta) # threshold Magenta
        mask2 = cv2.inRange(hsv_enhanced, lower_yellow, upper_yellow) # threshold Yellow
        mask3 = cv2.inRange(hsv_enhanced, lower_orange, upper_orange) # threshold Orange
        mask_combined = mask1 + mask2 + mask3

        mask1_inv = cv2.bitwise_not(mask1)
        mask2_inv = cv2.bitwise_not(mask2)
        mask3_inv = cv2.bitwise_not(mask3)

        contours1, _ = cv2.findContours(mask1_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours1:
            largest_contour1 = sorted(contours1, key=cv2.contourArea,reverse=True)[0]
        
        contours2, _ = cv2.findContours(mask2_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours2:
            largest_contour2 = sorted(contours2, key=cv2.contourArea,reverse=True)[0]
        
        contours3, _ = cv2.findContours(mask3_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours3:
            largest_contour3 = sorted(contours3, key=cv2.contourArea,reverse=True)[0]

        result = np.ones(frame.shape[:2])
        if contours1:
            result = cv2.drawContours(result, [largest_contour1],-1, color=(0, 255, 0), thickness=cv2.FILLED)

        contour_thickness = 2
        for contour in contours1:
            cv2.drawContours(frame, [contour], -1, (255,255,255),contour_thickness)
        for contour in contours2:
            cv2.drawContours(frame, [contour], -1, (255,255,255),contour_thickness)

        if success:
            cv2.imshow("Frame", frame) # Display frame w/o masking
            cv2.imshow("HSV",hsv) # Display frame to HSV mapping
            cv2.imshow("HSV Enhanced",hsv_enhanced) # Show enhanced HSV
            cv2.imshow("Result", mask1_inv) # Display threshold frame
            
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break # Break frame loop 'Q'

        else:
            break # Break upon frame read failure
    cap.release()    
    cv2.destroyAllWindows()   

if __name__ == "__main__":
    main()