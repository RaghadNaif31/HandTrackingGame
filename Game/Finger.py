import cv2
import time
import pyautogui
import mediapipe as mp
from mediapipe.tasks.python import vision

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

MODEL_PATH = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

cap = cv2.VideoCapture(0)

space_is_down = False

# حساسية رفع الإصبع
# لو ما يلتقط من بعيد: خليها 0.035
# لو يلقط غلط كثير: خليها 0.055
FINGER_THRESHOLD = 0.045

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        action_text = "NO HAND"

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            index_tip = hand[8]
            index_pip = hand[6]

            middle_tip = hand[12]
            middle_pip = hand[10]

            ring_tip = hand[16]
            ring_pip = hand[14]

            pinky_tip = hand[20]
            pinky_pip = hand[18]

            # بدل ما نعتمد على مكان اليد في الشاشة
            # نعتمد على الفرق بين طرف الإصبع والمفصل
            index_length = abs(index_pip.y - index_tip.y)
            middle_length = abs(middle_pip.y - middle_tip.y)
            ring_length = abs(ring_pip.y - ring_tip.y)
            pinky_length = abs(pinky_pip.y - pinky_tip.y)

            index_up = index_tip.y < index_pip.y and index_length > FINGER_THRESHOLD

            middle_down = not (middle_tip.y < middle_pip.y and middle_length > FINGER_THRESHOLD)
            ring_down = not (ring_tip.y < ring_pip.y and ring_length > FINGER_THRESHOLD)
            pinky_down = not (pinky_tip.y < pinky_pip.y and pinky_length > FINGER_THRESHOLD)

            final_gesture = index_up and middle_down and ring_down and pinky_down

            h, w, _ = frame.shape

            for lm in hand:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            if final_gesture:
                action_text = "JUMP"

                if not space_is_down:
                    pyautogui.keyDown("space")
                    space_is_down = True

            else:
                action_text = "ACTION"

                if space_is_down:
                    pyautogui.keyUp("space")
                    space_is_down = False

        else:
            action_text = "NO HAND"

            if space_is_down:
                pyautogui.keyUp("space")
                space_is_down = False

        cv2.putText(frame, action_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

if space_is_down:
    pyautogui.keyUp("space")

cap.release()
cv2.destroyAllWindows()