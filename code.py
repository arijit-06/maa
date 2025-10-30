import cv2
import mediapipe as mp
import requests
import time


CONTROL_URL = "http://192.168.1.1/" 


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils


TIP_IDS = [4, 8, 12, 16, 20]


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

last_command = None
command_send_interval = 0.1  
last_send_time = 0

print("Starting Hand Gesture Controller...")

try:
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        
        img = cv2.flip(img, 1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        command = "s"  
        finger_count = 0

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
           
            finger_list = []

            try:
                handedness = results.multi_handedness[0].classification[0].label
            except IndexError:
                handedness = "Right" 
            
            
            if handedness == "Right":
                if hand_landmarks.landmark[TIP_IDS[0]].x < hand_landmarks.landmark[TIP_IDS[0] - 1].x:
                    finger_list.append(1)
                else:
                    finger_list.append(0)
            else: 
                if hand_landmarks.landmark[TIP_IDS[0]].x > hand_landmarks.landmark[TIP_IDS[0] - 1].x:
                    finger_list.append(1)
                else:
                    finger_list.append(0)

            
            for id in range(1, 5):
                # Check if finger tip is above the joint below it
                if hand_landmarks.landmark[TIP_IDS[id]].y < hand_landmarks.landmark[TIP_IDS[id] - 2].y:
                    finger_list.append(1)
                else:
                    finger_list.append(0)

            finger_count = finger_list.count(1)
            
            
           
            if finger_count == 0:
                command = "s"  # Stop (Closed palm)
            elif finger_count == 2:
                command = "b"  # Backward
            elif finger_count == 3:
                command = "r"  # Right
            elif finger_count == 4:
                command = "l"  # Left
            elif finger_count == 5:
                command = "f"  # Forward (Open palm)
            else:
                command = "s" # Default for 1 finger, etc.

            # Draw landmarks on the image for visualization
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
       
        current_time = time.time()
        # Send command only if it's new or enough time has passed
        if command != last_command or (current_time - last_send_time > command_send_interval):
            try:
                # MODIFIED: Send "movement" parameter, not "command"
                params = {'movement': command}
                response = requests.get(CONTROL_URL, params=params, timeout=1)
                
                if response.status_code == 200:
                    print(f"Sent: {command}")
                    last_command = command
                    last_send_time = current_time
                else:
                    print(f"Failed to send command. Status: {response.status_code}")
            
            except requests.exceptions.RequestException as e:
                print(f"Error connecting to CAP 10: {e}")
                
        cv2.putText(img, f"Fingers: {finger_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"Command: {command.upper()}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the image
        cv2.imshow("Hand Gesture Controller", img)

        # Break loop on 'q' key press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

finally:
    
    print("Shutting down...")
    
    try:
       
        requests.get(CONTROL_URL, params={'movement': 's'}, timeout=1)
    except requests.exceptions.RequestException:
        pass 
        
    cap.release()
    cv2.destroyAllWindows()
    hands.close()