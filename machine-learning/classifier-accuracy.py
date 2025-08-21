import cv2
import dlib
import numpy as np
import json
import os

# 初始化 Dlib 的人臉檢測器和形狀預測器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# .json檔案
json_dir = "D:\\Advanced Image Processing\\project"

# 初始化變數以統計正確率
total_images = 0
correct_predictions = 0

json_file_path = os.path.join(json_dir, "s01.json")  # 將 "your_json_file.json" 替換為實際的 JSON 檔案名稱
with open(json_file_path, 'r') as file:
    labels_data = json.load(file)

# 圖像標籤
for entry in labels_data:
    total_images += 1

    image_path = os.path.join(json_dir, entry["image_path"])
    true_label = entry["label"]

    img = cv2.imread(image_path)

    if img is not None:
        # 將影像高度固定為 500，並等比例縮放寬度
        height, width = img.shape[:2]
        new_height = 500
        new_width = int(width * (new_height / height))
        img_resized = cv2.resize(img, (new_width, new_height))

        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

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

            # 判斷是否在 wink
            is_winking = short_side_diff > 1.0

            # 將結果標記在圖片上
            detected_label = "Winking" if is_winking else "Not Winking"
            cv2.putText(img_resized, f"Detected: {detected_label}", (face.left(), face.top() - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 比對標籤計算正確率
            if detected_label == true_label:
                correct_predictions += 1

        # 顯示結果
        cv2.imshow('Wink Detection (Resized and Rotated)', img_resized)
        cv2.waitKey(1)

    else:
        print(f"圖片 {image_path} 無法讀取。")

# 關閉所有視窗
cv2.destroyAllWindows()

# 計算正確率
accuracy = (correct_predictions / total_images) * 100
print(f"System Accuracy: {accuracy:.2f}%")
