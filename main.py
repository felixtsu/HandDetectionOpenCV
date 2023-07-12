from cvzone.HandTrackingModule import HandDetector
import cv2
from utils import cv2AddChineseText

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)


def detect_rock_paper_scissors(img, fingers, bbox):
    if fingers == [0, 0, 0, 0, 0]:
        return cv2AddChineseText(img, "石头", (bbox[0] + 60, bbox[1] - 60), (255, 0, 255))
        # cv2.putText(img, "石头", (bbox1[0] + 60, bbox1[1] - 30), cv2.FONT_HERSHEY_PLAIN,
        #             2, (255, 0, 255), 2)
    elif fingers == [1, 1, 1, 1, 1]:
        return cv2AddChineseText(img, "布", (bbox[0] + 60, bbox[1] - 60), (255, 0, 255))
        # cv2.putText(img, "布", (bbox1[0] + 60, bbox1[1] - 30), cv2.FONT_HERSHEY_PLAIN,
        #             2, (255, 0, 255), 2)
    elif fingers == [0, 1, 1, 0, 0]:
        return cv2AddChineseText(img, "剪刀", (bbox[0] + 60, bbox[1] - 60), (255, 0, 255))
        # cv2.putText(img, "剪刀", (bbox1[0] + 60, bbox1[1] - 30), cv2.FONT_HERSHEY_PLAIN,
        #             2, (255, 0, 255), 2)
    else:
        return img


while True:
    # Get image frame
    success, img = cap.read()
    # Find the hand and its landmarks
    hands, img = detector.findHands(img)  # with draw
    # hands = detector.findHands(img, draw=False)  # without draw

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points

        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right

        fingers1 = detector.fingersUp(hand1)
        img = detect_rock_paper_scissors(img, fingers1, bbox1)

        if len(hands) == 2:
            # Hand 2
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Landmark points
            bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
            centerPoint2 = hand2['center']  # center of the hand cx,cy
            handType2 = hand2["type"]  # Hand Type "Left" or "Right"

            fingers2 = detector.fingersUp(hand2)
            img = detect_rock_paper_scissors(img, fingers2, bbox2)

            # Find Distance between two Landmarks. Could be same hand or different hands
            # print(lmList1)
            # print(lmList2)
            length, info, img = detector.findDistance(lmList1[8][:2], lmList2[8][:2], img)  # with draw
            # length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw
    # Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
