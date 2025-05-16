import cv2
from pathlib import Path
import argparse
import time


from src.knn_lp_recognition import knn_E2E



def get_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image_path', help='link to image', default='./test/i (18).jpg')

    return arg.parse_args()


args = get_arguments()
img_path = Path(args.image_path)

# read image, Shape: (height, width, 3) (3 kênh: Blue, Green, Red).Giá trị pixel: [0, 255]
img = cv2.imread(str(img_path))

# start
start = time.time()

# load model
model = knn_E2E()

# recognize license plate
image = model.predict(img)
lis = model.get_license_plate_list()

def filter(lis):
    replace = {'D': '0', 'T': '1', 'S': '5','P':'8'}
    new_lis = []
    for i, char in enumerate(lis):
        if i >= 0 and char in replace:
            new_lis.append(replace[char])
        else:
            new_lis.append(char)
    return new_lis
# end
end = time.time()

print('Model process on %.2f s' % (end - start))
print(filter(lis[0]))

cv2.imshow('License Plate', image)
if cv2.waitKey(0) & 0xFF == ord('q'):
    exit(0)


cv2.destroyAllWindows()



