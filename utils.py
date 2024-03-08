import numpy as np
import cv2
import easyocr
reader = easyocr.Reader(['en'])
direction_refer_img = cv2.imread('direction_refer.jpg', 0)


def get_distance_area(img, rect=[40, 130, 100, 160]):
    return img[rect[1]:rect[3], rect[0]:rect[2], :]


def signal_logic(signals):
    if signals == [0, 0, 0, 0, 0, 0, 0]:
        return 0
    elif signals == [1, 1, 1, 1, 1, 1, 0]:
        return 0
    elif signals == [0, 0, 0, 0, 1, 1, 0]:
        return 1
    elif signals == [1, 0, 1, 1, 0, 1, 1]:
        return 2
    elif signals == [1, 0, 0, 1, 1, 1, 1]:
        return 3
    elif signals == [0, 1, 0, 0, 1, 1, 1]:
        return 4
    elif signals == [1, 1, 0, 1, 1, 0, 1]:
        return 5
    elif signals == [1, 1, 1, 1, 1, 0, 1]:
        return 6
    elif signals == [1, 1, 0, 0, 1, 1, 0]:
        return 7
    elif signals == [1, 1, 1, 1, 1, 1, 1]:
        return 8
    elif signals == [1, 1, 0, 1, 1, 1, 1]:
        return 9
    else:
        print('error signals:', signals)
        return None


def single_number_logic(img, thres=230):
    area1 = img[1:4, 4:20, :]
    area2 = img[6:18, 1:4, :]
    area3 = img[25:37, 1:4, :]
    area4 = img[40:43, 4:20, :]
    area5 = img[25:37, 21:25, :]
    area6 = img[6:18, 21:25, :]
    area7 = img[20:23, 4:20, :]
    # cv2.imwrite("im_digit.jpg", area1)

    signals = []
    for i, area in enumerate([area1, area2, area3, area4, area5, area6, area7]):
        # cv2.imwrite("im_digit_{}.jpg".format(i), area)
        # print(np.mean(area))
        signals.append(1 if np.mean(area) > thres else 0)
    num = signal_logic(signals)
    return num


def get_speed(img, rects=[[1688, 1147], [1719, 1147], [1750, 1147]], size=[25, 44]):
    nums = ''
    for i, rect in enumerate(rects):
        single_num_img = img[rect[1]:rect[1] + size[1], rect[0]:rect[0] + size[0], :]
        # cv2.imwrite("im_speed_{}.jpg".format(i), single_num_img)
        num = single_number_logic(single_num_img)
        # print(num)
        if num is None:
            cv2.imwrite('im_error_signals.jpg', single_num_img)
            return None
        else:
            nums += str(num)
    return int(nums)


def get_dist(img):
    img = get_distance_area(img)
    result = reader.readtext(img)
    if len(result):
        if result[0][1].isnumeric():
            return int(result[0][1])
    cv2.imwrite('im_error_dist.jpg', img)
    return None

def get_direction(img):
    roi = get_distance_area(img, rect=[1715, 1060, 1750, 1095])
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('direction.jpg', roi)
    # print(np.mean(roi / 255 - direction_refer_img / 255))
    if np.mean(roi / 255 - direction_refer_img / 255) < 0.01:
        direction = -1
    else:
        direction = 1
    return direction


if __name__ == '__main__':
    img = cv2.imread('im_opencv.jpg')

    # speed test
    print(get_speed(img))

    # dist test
    print(get_dist(img))

    # direction test
    print(get_direction(img))

    import time
    start = time.time()
    for i in range(1000):
        get_dist(img)
    end = time.time()
    print(end - start)