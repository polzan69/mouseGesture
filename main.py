import cv2
import mediapipe as mp
import sys
import pyautogui
import numpy as np
import time

# Disable pyautogui's fail-safe
pyautogui.FAILSAFE = False

class HandTracker:
    def __init__(self, static_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        """Initialize the HandTracker with MediaPipe hands"""
        self.static_mode = static_mode
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=2,  # Now detect both hands
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Smoothing factor for mouse movement
        self.smoothing = 0.5
        self.prev_x, self.prev_y = 0, 0
        
        # Click handling
        self.last_click_time = 0
        self.last_click_state = False
        self.double_click_threshold = 0.3  # seconds

    def find_hands(self, frame, draw=True):
        """Detect hands in the frame and optionally draw landmarks"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        self.results = self.hands.process(rgb_frame)
        
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    # Draw landmarks and connections
                    self.mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
        
        return frame

    def get_hand_landmarks(self, hand_type="Right"):
        """Get landmarks for a specific hand"""
        if not self.results.multi_hand_landmarks or not self.results.multi_handedness:
            return None
            
        for idx, handedness in enumerate(self.results.multi_handedness):
            if handedness.classification[0].label == hand_type:
                return self.results.multi_hand_landmarks[idx]
        return None

    def get_raised_fingers(self, hand_landmarks):
        """Return the state of each finger (1 for raised, 0 for folded)"""
        if not hand_landmarks:
            return []
            
        fingers = []
        
        # Thumb - check if thumb tip is to the left of the IP joint (joint closest to tip)
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_mcp = hand_landmarks.landmark[2]
        
        # For thumb, check if it's extended away from palm
        if thumb_tip.x < thumb_ip.x and thumb_ip.x < thumb_mcp.x:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # For other fingers, compare tip position with PIP joint (second joint)
        for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):  # Index, Middle, Ring, Pinky
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers

    def is_hand_closed(self, hand_landmarks):
        """More reliable detection for closed hand"""
        if not hand_landmarks:
            return False
            
        # Get finger states
        fingers = self.get_raised_fingers(hand_landmarks)
        
        # For a closed fist, we expect all fingers to be down (all zeros)
        # Allow some tolerance (maybe one finger not perfectly detected)
        return sum(fingers) <= 1  # Hand is considered closed if at most one finger is detected as raised

    def check_scroll_gesture(self, hand_landmarks):
        """Check if index finger is pointing up or down for scrolling"""
        if not hand_landmarks:
            return None
            
        # Get y-coordinates of index finger tip and joints
        index_tip = hand_landmarks.landmark[8].y
        index_pip = hand_landmarks.landmark[6].y  # PIP joint
        index_mcp = hand_landmarks.landmark[5].y  # MCP joint
        
        # Check if only index finger is extended
        fingers = self.get_raised_fingers(hand_landmarks)
        if fingers != [0, 1, 0, 0, 0]:  # Only index finger should be up
            return None
            
        # Determine direction based on finger angle
        if index_tip < index_pip < index_mcp:
            return "up"
        elif index_tip > index_pip > index_mcp:
            return "down"
        return None

    def control_mouse(self, frame):
        """Control mouse based on hand position and gestures"""
        right_hand = self.get_hand_landmarks("Right")
        left_hand = self.get_hand_landmarks("Left")
        
        if right_hand:
            # Get index finger tip position for cursor movement
            index_tip = right_hand.landmark[8]
            frame_height, frame_width, _ = frame.shape
            
            # Convert coordinates to screen position
            screen_x = int(np.interp(index_tip.x, [0, 1], [0, self.screen_width]))
            screen_y = int(np.interp(index_tip.y, [0, 1], [0, self.screen_height]))
            
            # Apply smoothing
            smooth_x = int(self.prev_x + (screen_x - self.prev_x) * self.smoothing)
            smooth_y = int(self.prev_y + (screen_y - self.prev_y) * self.smoothing)
            
            # Update previous positions
            self.prev_x, self.prev_y = smooth_x, smooth_y
            
            # Move cursor with right index finger
            right_fingers = self.get_raised_fingers(right_hand)
            if right_fingers[1] == 1:  # If index finger is up
                pyautogui.moveTo(smooth_x, smooth_y)
            
            # Handle right click with right pinky
            if right_fingers == [0, 0, 0, 0, 1]:
                pyautogui.rightClick()
        
        # Handle left click and scrolling with left hand
        if left_hand:
            left_fingers = self.get_raised_fingers(left_hand)
            is_closed = self.is_hand_closed(left_hand)
            
            # Handle left click and hold with closed left hand
            if is_closed:
                current_time = time.time()
                if not self.last_click_state:
                    # Initial click
                    if current_time - self.last_click_time < self.double_click_threshold:
                        pyautogui.doubleClick()
                    else:
                        pyautogui.mouseDown()  # Press and hold
                    self.last_click_time = current_time
                self.last_click_state = True
            else:
                if self.last_click_state:
                    pyautogui.mouseUp()  # Release the hold
                self.last_click_state = False
            
            # Handle scrolling with left index finger
            scroll_direction = self.check_scroll_gesture(left_hand)
            if scroll_direction == "up":
                pyautogui.scroll(5)  # Scroll up
            elif scroll_direction == "down":
                pyautogui.scroll(-5)  # Scroll down

def initialize_camera():
    """Initialize the webcam and return the video capture object"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit(1)
    return cap

def main():
    # Initialize the webcam and hand tracker
    cap = initialize_camera()
    tracker = HandTracker()
    
    try:
        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Find and draw hands
            frame = tracker.find_hands(frame)
            
            # Control mouse based on hand position and gestures
            tracker.control_mouse(frame)
            
            # Display hand states
            right_hand = tracker.get_hand_landmarks("Right")
            left_hand = tracker.get_hand_landmarks("Left")
            
            if right_hand:
                right_fingers = tracker.get_raised_fingers(right_hand)
                cv2.putText(frame, f'Right: {right_fingers}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if left_hand:
                left_fingers = tracker.get_raised_fingers(left_hand)
                cv2.putText(frame, f'Left: {left_fingers}', (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Hand Mouse Controller', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 