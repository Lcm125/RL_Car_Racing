import time
import ctypes
import win32api
import win32ui
import win32gui
import win32con
import numpy as np
import cv2
from action import Action
from utils import get_speed, get_dist, get_direction, get_distance_area, get_edge


class Env():
    """
    Environment wrapper for WRCG
    """

    def __init__(self, handle):
        self.hwnd_target = handle
        self.action = Action()
        self.states = []
        self.action_spaces = ['w', 'a', 'd', '']
        self.size = (1920, 1200)
        self.repeat_nums = 0
        self.repeat_thres = 20
        win32gui.SetForegroundWindow(self.hwnd_target)
        # self.end_thres = None
        self.last_dist = None
        time.sleep(1)
        self.pause_game()
        time.sleep(1)

    def reset_car(self):
        self.action.press_key('r', internal=2)
        time.sleep(1)
        self.init_states()
        return np.stack(self.states, axis=0)

    def reset_game(self):
        self.pause_game()
        time.sleep(1)
        self.action.move_mouse((40, 440))
        time.sleep(1)
        self.action.left_click()
        time.sleep(1)
        self.action.move_mouse((620, 830))
        time.sleep(1)
        self.action.left_click()
        time.sleep(5)
        self.action.move_mouse((40, 310))
        time.sleep(1)
        self.action.left_click()
        time.sleep(1)
        self.init_states()
        self.init_run()
        self.init_states()
        return np.stack(self.states, axis=0)

    def init_states(self):
        self.states = []
        state = self.get_frame()
        gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (112, 112)) / 255.
        for i in range(4):
            self.states.append(gray)
        self.last_dist = self.get_dist(state)
        self.repeat_nums = 0

    def pause_game(self):
        time.sleep(1)
        self.action.press_key('esc', internal=0.1)
        time.sleep(3)

    def get_frame(self):
        left, top, right, bot = win32gui.GetWindowRect(self.hwnd_target)
        left, top, right, bot = [int(x * 1) for x in [left, top, right, bot]]
        w = right - left
        h = bot - top
        hdesktop = win32gui.GetDesktopWindow()
        hwndDC = win32gui.GetWindowDC(hdesktop)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
        saveDC.SelectObject(saveBitMap)
        result = saveDC.BitBlt((0, 0), (w, h), mfcDC, (left, top), win32con.SRCCOPY)
        signedIntsArray = saveBitMap.GetBitmapBits(True)
        im_opencv = np.frombuffer(signedIntsArray, dtype='uint8')
        im_opencv.shape = (h, w, 4)
        im_opencv = cv2.cvtColor(im_opencv, cv2.COLOR_BGRA2BGR)
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hdesktop, hwndDC)
        return im_opencv

    def get_dist(self, img):
        # print(self.last_dist)
        # if self.last_dist is None or self.last_dist >= 1000:
        return get_dist(img, rect=[int(20 / 1920 * self.size[0]), int(60 / 1200 * self.size[1]), int(110 / 1920 * self.size[0]), int(90 / 1200 * self.size[1])])
        # print(1)
        # return get_dist(img, rect=[40, 130, 90, 160])

    def get_speed(self, img):
        direction = get_direction(img)
        return get_speed(img) * direction

    def get_edge(self, img):
        return get_edge(img,
                        rect1=[int(1646 / 1920 * self.size[0]), int(235 / 1200 * self.size[1]), int(1664 / 1920 * self.size[0]), int(253 / 1200 * self.size[1])],
                        rect2=[int(1656 / 1920 * self.size[0]), int(235 / 1200 * self.size[1]), int(1674 / 1920 * self.size[0]), int(253 / 1200 * self.size[1])]
                        )

    def calc_reward(self, dist, dist_):
        return (-dist_ + dist) * 1 - 0.1

    def step(self, action):
        if action == 0:
            self.action.down_key(self.action_spaces[0])
            time.sleep(0.5)
            self.action.up_key(self.action_spaces[0])
        elif action == 1 or action == 2:
            self.action.down_key(self.action_spaces[0])
            self.action.down_key(self.action_spaces[action])
            time.sleep(0.25)
            self.action.up_key(self.action_spaces[action])
            self.action.up_key(self.action_spaces[0])
            time.sleep(0.25)

        else:
            time.sleep(0.5)

        state_ = self.get_frame()
        out_edge = self.get_edge(state_)
        gray = cv2.cvtColor(state_,  cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (112, 112)) / 255.
        self.states.pop(0)
        self.states.append(gray)

        dist_ = self.get_dist(state_)
        if dist_ is None or self.last_dist is None:
            print('None err', dist_, self.last_dist)
            return None, None, None, 1
        if abs(dist_ - self.last_dist) >= 100:
            print('dist diff err', dist_, self.last_dist)
            return None, None, None, 2
        if dist_ <= 40 and self.last_dist <= 40:
            print('small dist reset', dist_, self.last_dist)
            return None, None, None, 3
        r = self.calc_reward(self.last_dist, dist_)
        # if out_edge:
        #     r = -0.1

        if self.last_dist == dist_:
            self.repeat_nums += 1
            # print(self.repeat_nums)
        else:
            self.repeat_nums = 0

        self.last_dist = dist_

        done = True if (self.repeat_nums >= self.repeat_thres or out_edge) else False
        if done:
            r -= 20
        return np.stack(self.states, axis=0), r, done, 0

    def init_run(self):
        for i in range(1):
            self.step(0)
        time.sleep(5)

if __name__ == '__main__':
    wrcg_env = Env(0x000806FA)
    wrcg_env.reset_game()
    # while True:
        # img = wrcg_env.get_frame()
        # cv2.imwrite('im.jpg', img)
        # break
        # dist = wrcg_env.get_dist(img)
        # break
        # out_edge = wrcg_env.get_edge(img)
        # print(out_edge)
        # print(dist)
        # break
        # wrcg_env.last_dist = dist
        # break
        # if dist is None:
        #     break
        # print(dist)