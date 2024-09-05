import cv2
import numpy as np
import mediapipe as mp
import autopy
import time

# Global variables
points = []
roi_defined = False
roi_corners = None
screen_width, screen_height = autopy.screen.size()
previous_position = (0, 0)  # Store previous mouse position

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils


def mouse_callback(event, x, y, flags, param):
    global points, roi_defined, roi_corners

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            if len(points) == 4:
                roi_defined = True
                roi_corners = np.array(points, dtype=np.float32)
                roi_corners = sort_roi_points(roi_corners)


def sort_roi_points(points):
    points = np.array(points, dtype=np.float32)
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


def is_point_inside_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def get_normalized_coordinates(point, roi_corners):
    x, y = point
    x_coords, y_coords = roi_corners[:, 0], roi_corners[:, 1]
    min_x, min_y = min(x_coords), min(y_coords)
    max_x, max_y = max(x_coords), max(y_coords)

    # Correctly map the x-coordinate without inversion
    normalized_x = (x - min_x) / (max_x - min_x)
    normalized_y = (y - min_y) / (max_y - min_y)

    return (normalized_x, normalized_y)


def draw_roi(frame, roi_corners):
    x, y, w, h = cv2.boundingRect(roi_corners)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(frame, "ROI", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)


def process_frame(frame):
    global results
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)


def should_move_mouse(previous_position, current_position, threshold=10):
    distance = np.linalg.norm(np.array(previous_position) - np.array(current_position))
    return distance > threshold


def main():
    global frame, roi_defined, roi_corners, previous_position

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Lower resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Lower resolution
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", mouse_callback)

    prev_time = time.time()
    target_fps = 60  # Lower target FPS
    frame_delay = 1.0 / target_fps

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        elapsed_time = current_time - prev_time

        if elapsed_time < frame_delay:
            time.sleep(frame_delay - elapsed_time)

        # Process frame
        process_frame(frame)

        if roi_defined:
            draw_roi(frame, roi_corners)

        if results and results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, _ = frame.shape
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

                if roi_defined and is_point_inside_polygon((index_x, index_y), roi_corners):
                    normalized_coords = get_normalized_coordinates((index_x, index_y), roi_corners)
                    screen_x = int(normalized_coords[0] * screen_width)
                    screen_y = int(normalized_coords[1] * screen_height)

                    if should_move_mouse(previous_position, (screen_x, screen_y)):
                        autopy.mouse.move(screen_x, screen_y)
                        previous_position = (screen_x, screen_y)

        cv2.putText(frame, f"FPS: {int(1.0 / elapsed_time)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()