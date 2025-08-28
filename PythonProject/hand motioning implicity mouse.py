import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Screen size for mapping cursor
screen_width, screen_height = pyautogui.size()

# Open webcam
cap = cv2.VideoCapture(0)

# Define helper functions
def left_click():
    pyautogui.click()
    print("Left Click Performed")

def right_click():
    pyautogui.click(button='right')
    print("Right Click Performed")

def scroll_down():
    pyautogui.scroll(-300)  # Negative = scroll down
    print("Scroll Down Performed")

def scroll_up():
    pyautogui.scroll(300)   # Positive = scroll up
    print("Scroll Up Performed")

# Hand tracking
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip image for mirror effect
        image = cv2.flip(image, 1)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image and detect hands
        results = hands.process(image_rgb)

        # If hands detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get landmark positions
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                # Convert normalized coordinates to screen size
                index_x = int(index_tip.x * screen_width)
                index_y = int(index_tip.y * screen_height)

                # Move mouse cursor to index finger position
                pyautogui.moveTo(index_x, index_y, duration=0.05)

                # Gesture Recognition
                if thumb_tip.y < index_tip.y and middle_tip.y < index_tip.y:
                    cv2.putText(image, "Gesture: Left Click", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    left_click()
                    time.sleep(1)

                elif index_tip.y < middle_tip.y:
                    cv2.putText(image, "Gesture: Right Click", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    right_click()
                    time.sleep(1)

                elif thumb_tip.y > index_tip.y and middle_tip.y > index_tip.y:
                    cv2.putText(image, "Gesture: Scroll Down", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    scroll_down()
                    time.sleep(1)

        # Show output
        cv2.imshow('Hand Gesture Recognition', image)

        # Exit on ESC
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
