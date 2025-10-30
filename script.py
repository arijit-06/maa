import cv2
import mediapipe as mp
import time
from datetime import datetime

# Configuration
WINDOW_NAME = "Hand Gesture Controller"

# Command mappings for better readability
COMMAND_MAPPINGS = {
    0: ("STOP", "s", (0, 0, 255)),      # Red
    2: ("BACKWARD", "b", (255, 165, 0)), # Orange
    3: ("RIGHT", "r", (0, 255, 0)),      # Green
    4: ("LEFT", "l", (255, 255, 0)),     # Yellow
    5: ("FORWARD", "f", (0, 255, 0))     # Green
}

class HandGestureController:
    def __init__(self):
        # MediaPipe initialization
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles

        # Video capture initialization
        self.cap = cv2.VideoCapture(0)
        self.setup_camera()

        # Controller state
        self.last_command = None
        self.command_send_interval = 0.3
        self.last_send_time = 0
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.last_finger_count = None

    def setup_camera(self):
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open webcam.")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def update_fps(self):
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()

    def draw_interface(self, img, finger_count, command, connection_status):
        # Create a semi-transparent overlay for the interface
        overlay = img.copy()
        
        # Draw background rectangle for telemetry
        cv2.rectangle(overlay, (20, 20), (400, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

        # Display telemetry
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"Fingers Detected: {finger_count}", (30, 50), font, 0.8, (255, 255, 255), 2)
        
        # Show current command with its mapped color
        command_info = COMMAND_MAPPINGS.get(finger_count, ("UNDEFINED", "s", (128, 128, 128)))
        cv2.putText(img, f"Command: {command_info[0]}", (30, 90), font, 0.8, command_info[2], 2)
        
        # Show FPS and connection status
        cv2.putText(img, f"FPS: {int(self.fps)}", (30, 130), font, 0.8, (255, 255, 255), 2)
        
        # Connection status indicator
        status_color = (0, 255, 0) if connection_status else (0, 0, 255)
        cv2.circle(img, (380, 40), 10, status_color, -1)

        # Draw command reference
        self.draw_command_reference(img)

    def draw_command_reference(self, img):
        # Draw command reference in bottom-right corner
        start_y = img.shape[0] - 180
        cv2.rectangle(img, (img.shape[1] - 200, start_y), 
                     (img.shape[1] - 20, img.shape[0] - 20), (0, 0, 0), -1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, (count, (name, _, color)) in enumerate(COMMAND_MAPPINGS.items()):
            y_pos = start_y + 30 * (i + 1)
            cv2.putText(img, f"{count} fingers: {name}", 
                       (img.shape[1] - 190, y_pos), font, 0.6, color, 2)

    def process_hand_landmarks(self, hand_landmarks, handedness):
        """
        Robust finger counting:
        - For index to pinky: tip.y < pip.y indicates finger is up (with a small tolerance)
        - For thumb: compare tip.x to ip.x depending on handedness (with tolerance)
        """
        finger_count = 0
        try:
            # Tip and pip indices
            tips = [4, 8, 12, 16, 20]
            # Tolerance to reduce flicker/noise
            tol = 0.02

            # Thumb: compare tip (4) with ip (3)
            thumb_tip = hand_landmarks.landmark[4].x
            thumb_ip = hand_landmarks.landmark[3].x
            if handedness == "Right":
                if thumb_tip < thumb_ip - tol:
                    finger_count += 1
            else:
                if thumb_tip > thumb_ip + tol:
                    finger_count += 1

            # Other fingers: tip y < pip y => finger up
            for tip_id in tips[1:]:
                tip_y = hand_landmarks.landmark[tip_id].y
                pip_y = hand_landmarks.landmark[tip_id - 2].y
                if tip_y < pip_y - tol:
                    finger_count += 1

        except Exception:
            # In case landmarks are missing for any reason, return 0
            return 0

        return finger_count

    def send_command(self, command):
        # Network/control removed — keep as local no-op (always succeeds)
        return True

    def run(self):
        print("\n=== Hand Gesture Controller Started ===")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting (offline mode)")
        print("Press 'Q' to quit\n")

        try:
            while self.cap.isOpened():
                success, img = self.cap.read()
                if not success:
                    print("[ERROR] Failed to read camera frame.")
                    continue

                # Update FPS
                self.update_fps()

                # Process the raw (not flipped) image for consistent handedness
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hands.process(img_rgb)

                # Default command and finger count
                command = "s"
                finger_count = 0
                connection_status = True

                # Process hand landmarks if detected (use landmarks from unflipped image)
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    try:
                        handedness = results.multi_handedness[0].classification[0].label
                    except IndexError:
                        handedness = "Right"

                    # Get finger count and corresponding command
                    finger_count = self.process_hand_landmarks(hand_landmarks, handedness)
                    command = COMMAND_MAPPINGS.get(finger_count, ("UNDEFINED", "s", (128, 128, 128)))[1]

                    # Draw hand landmarks with improved style
                    self.mp_draw.draw_landmarks(
                        img,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw_styles.get_default_hand_landmarks_style(),
                        self.mp_draw_styles.get_default_hand_connections_style()
                    )

                # Print concise console output only when the finger count changes
                if finger_count != self.last_finger_count:
                    if finger_count > 0:
                        cmd_name = COMMAND_MAPPINGS.get(finger_count, ("UNDEFINED", "s", (128, 128, 128)))[0]
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Fingers: {finger_count} -> {cmd_name}")
                    self.last_finger_count = finger_count

                # Mirror the image for display (so the view feels natural to the user)
                img_display = cv2.flip(img, 1)

                # Send command if needed
                current_time = time.time()
                if (command != self.last_command or 
                    current_time - self.last_send_time > self.command_send_interval):
                    connection_status = self.send_command(command)
                    if connection_status:
                        self.last_command = command
                        self.last_send_time = current_time

                # Draw interface on the mirrored display image
                self.draw_interface(img_display, finger_count, command, connection_status)

                # Show the image
                cv2.imshow(WINDOW_NAME, img_display)

                # Break loop on 'q' key press
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nReceived interrupt signal")

        finally:
            self.cleanup()

    def cleanup(self):
        print("\n=== Shutting down ===")
        # No network cleanup required in offline mode
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("Cleanup completed successfully")

if __name__ == "__main__":
    try:
        controller = HandGestureController()
        controller.run()
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {str(e)}")
        print("The application will now exit.")