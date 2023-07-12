import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# font=cv2.FONT_ITALIC
def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    # if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "./PingFang.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)