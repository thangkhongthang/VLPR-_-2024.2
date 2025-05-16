import os
import numpy as np
import cv2

# Tạo thư mục output nếu chưa tồn tại
os.makedirs("./data", exist_ok=True)

# Xử lý digits
path = 'c:/Users/Nguyen Van Thang/Documents/GitHub/License-Plate-Recognition/output_characters/'
data = []


img_fi_path = os.listdir(path)
for img_path in img_fi_path:
    img = cv2.imread(path + "/" + img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to read image: {path + '/' + img_path}")
        continue
    img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
    img = img.reshape((28, 28, 1))
    data.append((img))

output_path = "./data/noise.npy"
np.save(output_path, np.array(data, dtype=object))
print(f"Saved {len(data)} noises to {output_path}")
