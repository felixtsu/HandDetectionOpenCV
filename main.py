from cvzone.HandTrackingModule import HandDetector
import cv2

cap = cv2.VideoCapture(0)

# 生成一个手部侦测器
# 敏感度是0.8 可以改变模型的灵敏度 更灵敏有可能会出错
# maxHands 最多识别多少双手
detector = HandDetector(detectionCon=0.8, maxHands=2)


## 如何用这个来判定石头剪刀布呢？
def detect_rock_paper_scissors(img, fingers, bbox):
    ## 以下三行分别在手部框框上方输出 rock paper和scissor
    ## 我这里为了展示，不管你怎么出我都判定出了石头
    cv2.putText(img, "rock", (bbox1[0] + 60, bbox1[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 0, 255), 2)
    # cv2.putText(img, "paper", (bbox1[0] + 60, bbox1[1] - 30), cv2.FONT_HERSHEY_PLAIN,
    #                 2, (255, 0, 255), 2)
    # cv2.putText(img, "scissor", (bbox1[0] + 60, bbox1[1] - 30), cv2.FONT_HERSHEY_PLAIN,
    #                 2, (255, 0, 255), 2)
    return "rock"
    


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

        ## 你要做的是实现这个函数，让程序可以识别现在出来的是石头剪刀还是布
        result = detect_rock_paper_scissors(img, fingers1, bbox1)

        ## 无敌的作弊开始，出啥都秒杀
        if result == "rock":
            cv2.putText(img, "AI plays paper", (500, 270), cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 0, 255), 2)
        elif result == "paper":
            cv2.putText(img, "AI plays scissor", (500, 270), cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 0, 255), 2)
        elif result == "scissor":
            cv2.putText(img, "AI plays rock", (500, 270), cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 0, 255), 2)



    # Display
    cv2.imshow("God of rock paper scissor", img)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
