import cv2  #opencv函數庫
import numpy as np  #計算矩形
import dlib  #人臉偵測

# 初始化 Dlib 的人臉檢測器和形狀預測器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 讀取輸入的圖片
image_path = 's13.jpg'  # 請替換成您的圖片路徑
img = cv2.imread(image_path)
cv2.imshow('image', img)

# 將影像高度固定為 500，並等比例縮放寬度
#height, width = img.shape[:2]
#new_height = 500
#new_width = int(width * (new_height / height))
#img_resized = cv2.resize(img, (new_width, new_height))
#cv2.imshow('image', img_resized)

#gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
gray = cv2. cvtColor(img, cv2.COLOR_BGR2GRAY)

# 進行人臉檢測
faces = detector(gray)

# 遍歷檢測到的人臉
for face in faces:
    # 提取人臉的形狀
    shape = predictor(gray, face)
    
    # 提取左眼和右眼的關鍵點
    left_eye_points = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
    right_eye_points = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]

    # 獲取左眼的旋轉矩形
    rotated_left_eye_rect = cv2.minAreaRect(np.array(left_eye_points))

    # 獲取右眼的旋轉矩形
    rotated_right_eye_rect = cv2.minAreaRect(np.array(right_eye_points))

    # 獲取旋轉後的左眼方框的短邊長度
    short_side_left_eye = min(rotated_left_eye_rect[1])

    # 獲取旋轉後的右眼方框的短邊長度
    short_side_right_eye = min(rotated_right_eye_rect[1])

    # 計算左右眼方框的短邊長度差
    short_side_diff = abs(short_side_left_eye - short_side_right_eye)

    # 獲取左眼方框的四個角點
    left_eye_rect_points = cv2.boxPoints(rotated_left_eye_rect)
    left_eye_rect_points = np.int0(left_eye_rect_points)

    # 獲取右眼方框的四個角點
    right_eye_rect_points = cv2.boxPoints(rotated_right_eye_rect)
    right_eye_rect_points = np.int0(right_eye_rect_points)

    # 繪製左眼方框
    #cv2.drawContours(img_resized, [left_eye_rect_points], 0, (0, 255, 0), 2)
    cv2.drawContours(img, [left_eye_rect_points], 0, (0, 255, 0), 2)

    # 繪製右眼方框
    #cv2.drawContours(img_resized, [right_eye_rect_points], 0, (0, 255, 0), 2)
    cv2.drawContours(img, [right_eye_rect_points], 0, (0, 255, 0), 2)

    # 判斷是否在 wink
    is_winking = short_side_diff > 1.0

    # 在圖像上標記 wink 的狀態
    wink_status = "Winking" if is_winking else "Not Winking"
    #cv2.putText(img_resized, f"Status: {wink_status}", (face.left(), face.top() - 10),
                #cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img, f"status: {wink_status}", (face.left(), face.top() - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# 顯示結果
#cv2.imshow('Wink Detection (Resized and Rotated)', img_resized)
cv2.imshow('Wink Dectection', img)
#cv2.imshow(img)
cv2.waitKey(0)
cv2.destroyAllWindows()

