import cv2
import mediapipe as mp
import numpy as np

# Initialiser MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Initialiser le dessinateur
mp_drawing = mp.solutions.drawing_utils

# Ouvrir la webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)


button_colors = {'Start': (0, 0, 255), 'Stop': (0, 0, 255)}

gesture_recognition_active = False

while True:
    # Lire l'image de la webcam
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    buttons = [{'name': 'Start', 'rect': (frame.shape[1] - 220, 10, 100, 50)}, {'name': 'Stop', 'rect': (frame.shape[1] - 110, 10, 100, 50)}]    # Convertir l'image en RGB
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Traiter l'image avec MediaPipe Hands
    result = hands.process(rgb)

    # Dessiner les résultats
    if result.multi_hand_landmarks is not None:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger_tip_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
            index_finger_tip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

            for button in buttons:
                x, y, w, h = button['rect']
                cv2.rectangle(frame, (x, y), (x + w, y + h), button_colors[button['name']], 2)
                cv2.putText(frame, button['name'], (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, button_colors[button['name']], 2, cv2.LINE_AA)
                if x < index_finger_tip_x < x + w and y < index_finger_tip_y < y + h:
                    if button['name'] == 'Start':
                        gesture_recognition_active = True
                        button_colors['Start'] = (0, 255, 0)    
                        button_colors['Stop'] = (0, 0, 255)                
                    elif button['name'] == 'Stop':
                        gesture_recognition_active = False
                        button_colors['Stop'] = (0, 255, 0)
                        button_colors['Start'] = (0, 0, 255) 


            if gesture_recognition_active:
                index_finger_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
                index_finger_mcp = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y])            
                middle_finger_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y])
                middle_finger_mcp = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y])
                middle_finger_pip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y])
                ring_finger_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y])
                ring_finger_mcp = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y])
                pinky_finger_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y])
                pinky_finger_mcp = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y])
                thumb_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y])
                thumb_mcp = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y])

                index_finger_direction = index_finger_tip - np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y])
                middle_finger_direction = middle_finger_tip - np.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y])
                thumb_to_index_distance = np.linalg.norm(thumb_tip - index_finger_tip)
                thumb_to_middle_distance = np.linalg.norm(thumb_tip - middle_finger_tip)
                thumb_to_ring_distance = np.linalg.norm(thumb_tip - ring_finger_tip)
                thumb_to_pinky_distance = np.linalg.norm(thumb_tip - pinky_finger_tip)

                distance_coeur = np.linalg.norm(index_finger_tip - thumb_tip)
                
                if (index_finger_tip < index_finger_mcp and middle_finger_tip < middle_finger_mcp and ring_finger_tip < ring_finger_mcp and pinky_finger_tip < pinky_finger_mcp and thumb_tip < thumb_mcp):
                    cv2.putText(frame, 'Main ouverte', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif (index_finger_tip < index_finger_mcp and middle_finger_tip < middle_finger_mcp and ring_finger_tip < ring_finger_mcp and pinky_finger_tip < pinky_finger_mcp and thumb_tip > thumb_mcp and thumb_to_index_distance < 0.05 and thumb_to_middle_distance < 0.05 and thumb_to_ring_distance < 0.05 and thumb_to_pinky_distance < 0.05):
                    cv2.putText(frame, 'Main fermée', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif index_finger_tip < index_finger_mcp and middle_finger_tip < middle_finger_mcp:
                    distance_peace = np.linalg.norm(index_finger_tip - middle_finger_tip)
                    angle = np.arctan2(index_finger_direction[1], index_finger_direction[0]) - np.arctan2(middle_finger_direction[1], middle_finger_direction[0])
                    angle = np.abs(angle) * 180.0 / np.pi
                    if distance_peace < 0.10 and angle < 30:
                        cv2.putText(frame, 'Paix', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif (middle_finger_pip < index_finger_tip and middle_finger_pip < ring_finger_tip and middle_finger_pip < pinky_finger_tip and middle_finger_pip < thumb_tip):
                    cv2.putText(frame, 'Fuck Nico', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif distance_coeur < 0.10:
                    cv2.putText(frame, 'Coeur', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    

    # Afficher l'image
    cv2.imshow('Frame', frame)

    # Quitter si 'q' est pressé
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()