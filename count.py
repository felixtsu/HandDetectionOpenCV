from cvzone.HandTrackingModule import HandDetector
import cv2
from utils import cv2AddChineseText

cap = cv2.VideoCapture(0)

# 生成一个手部侦测器
# 敏感度是0.8 可以改变模型的灵敏度 更灵敏有可能会出错
# maxHands 最多识别多少双手
detector = HandDetector(detectionCon=0.8, maxHands=2)


while True:
    # Get image frame
    success, img = cap.read()

    # 找到手放到第一个变量里面，并且把点标注在这一个图片帧上
    hands, img = detector.findHands(img)  # with draw
    ## 如果用下面的这一行，那么手部点就不会标注在这一个图片帧上
    # hands = detector.findHands(img, draw=False)  # without draw


    ## 我们要数数
    count = 0
    count2 = 0
    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points

        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right

        fingers1 = detector.fingersUp(hand1)

        # 这一行就是不管三七二十一都认为count就是1
        count = 1
        cv2.putText(img, str(count), (bbox1[0] + 60, bbox1[1] - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        if len(hands) == 2:
            # Hand 2
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # List of 21 Landmark points
            bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
            centerPoint2 = hand2['center']  # center of the hand cx,cy
            handType2 = hand2["type"]  # Hand Type "Left" or "Right"

            fingers2 = detector.fingersUp(hand2)


            # 这一行就是不管三七二十一都认为第二只手就只有3根手指竖起来
            count2 = 3
            cv2.putText(img, str(count2), (bbox2[0] + 60, bbox2[1] - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

            # 有两只手的时候，总数在两只手中间打印出来
            cv2.putText(img, "Total:" + str(count2 + count), ((bbox1[0] + bbox2[0])//2, (bbox1[1] + bbox2[1])//2), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 255), 2)
        else:
            # 只有一只手的时候，总数在手上方打印出来
            cv2.putText(img, "Total:" + str(count2 + count), (bbox1[0] + 60, bbox1[1] - 60), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 255), 2)
    


    # Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
