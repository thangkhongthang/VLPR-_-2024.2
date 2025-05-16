import cv2
import matplotlib.pyplot as plt

def hsv_to_adaptive_thresh(hsv_img):
    # Chuyển HSV sang Grayscale bằng cách chỉ lấy kênh V (Value)
    v_channel = hsv_img[:, :, 2]
    # Áp dụng Adaptive Threshold lên kênh V
    adaptive = cv2.adaptiveThreshold(
        v_channel, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    return adaptive

# Load image in RGB format
image_bgr = cv2.imread('./test/i (18).jpg')  # Đổi đường dẫn nếu cần
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Convert RGB to HSV
image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

# Convert RGB to Grayscale
image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

# Convert HSV to Adaptive Threshold
image_adaptive = hsv_to_adaptive_thresh(image_hsv)

# Plotting the images
plt.figure(figsize=(14, 7))


plt.title('Adaptive Threshold (from HSV)')
plt.imshow(image_adaptive, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
