import numpy as np
import cv2
import easyocr
reader = easyocr.Reader(['en'])
edge_refer_img = cv2.imread('env_images/edge_refer.jpg')
start_refer_img = cv2.imread('env_images/start_refer.jpg')
restart_refer_img = cv2.imread('env_images/restart_refer.jpg')
restart_text_refer_img = cv2.imread('env_images/restart_text_refer.jpg')


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

def get_binary(img, thresh=220):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thres = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    return thres

def get_white_digit(img, thresh=230):
    b, g, r = cv2.split(img)
    _, b = cv2.threshold(b, thresh, 255, cv2.THRESH_BINARY)
    _, g = cv2.threshold(g, thresh, 255, cv2.THRESH_BINARY)
    _, r = cv2.threshold(r, thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(cv2.bitwise_and(b, g), r)
    return mask

def get_sobel(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    mask = cv2.add(sobelx, sobely)
    return mask

def get_dist(img, rect=[40, 130, 100, 160]):
    img = get_distance_area(img, rect)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thres = get_white_digit(img)
    # return None
    result = reader.readtext(thres)
    # cv2.imwrite('im_gray_dist.jpg', gray)
    # cv2.imwrite('im_thres_dist.jpg', thres)
    # cv2.imwrite('im_error_dist.jpg', img)
    # print(result[0][1], s)
    if len(result):
        s = result[0][1].lower().replace('m', '').strip()
        if s.isnumeric():
            cv2.imwrite('im_gray_dist_0.jpg', gray)
            cv2.imwrite('im_thres_dist_0.jpg', thres)
            cv2.imwrite('im_error_dist_0.jpg', img)
            return int(s)
    cv2.imwrite('im_gray_dist.jpg', gray)
    cv2.imwrite('im_thres_dist.jpg', thres)
    cv2.imwrite('im_error_dist.jpg', img)
    return None

# def get_direction(img):
#     roi = get_distance_area(img, rect=[1715, 1060, 1750, 1095])
#     roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     # cv2.imwrite('direction.jpg', roi)
#     # print(np.mean(roi / 255 - direction_refer_img / 255))
#     if np.mean(roi / 255 - direction_refer_img / 255) < 0.01:
#         direction = -1
#     else:
#         direction = 1
#     return direction

def get_edge(img, rect):
    roi = get_distance_area(img, rect=rect)
    # cv2.imwrite('edge.jpg', roi)
    res = cv2.matchTemplate(roi, edge_refer_img, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return True if min_val < 0.1 else False

def get_restart_text(img):
    res = cv2.matchTemplate(get_binary(img), get_binary(restart_text_refer_img), cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return [x + y // 2 for x, y in zip(min_loc, restart_text_refer_img.shape[:2][::-1])]

def get_restart(img):
    res = cv2.matchTemplate(get_binary(img), get_binary(restart_refer_img), cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return [x + y // 2 for x, y in zip(min_loc, restart_refer_img.shape[:2][::-1])]

def get_start(img):
    res = cv2.matchTemplate(get_binary(img), get_binary(start_refer_img), cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return [x + y // 2 for x, y in zip(min_loc, start_refer_img.shape[:2][::-1])]

if __name__ == '__main__':
    img = cv2.imread('test2.png')
    img = cv2.resize(img, (1920, 1080))
    #
    print(get_restart_text(img))
    # cv2.imwrite('start_refer.jpg', img)

    # img = cv2.imread('restart_text.png')
    # img = cv2.resize(img, (86, 33))
    # cv2.imwrite('restart_text_refer.jpg', img)
