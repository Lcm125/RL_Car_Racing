import time
import ctypes
import win32api
import win32ui
import win32gui
import win32con
import numpy as np
import cv2
from action import Action
from utils import get_speed, get_dist, get_direction, get_distance_area


class Env():
    """
    Environment wrapper for WRCG
    """

    def __init__(self, handle):
        self.hwnd_target = handle
        self.action = Action()
        self.states = []
        self.action_spaces = ['w', 'a', 'd', '']

        self.repeat_nums = 0
        self.repeat_thres = 20
        win32gui.SetForegroundWindow(self.hwnd_target)
        self.end_thres = None
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
        self.action.move_mouse((20, 220))
        time.sleep(1)
        self.action.left_click()
        time.sleep(1)
        self.action.move_mouse((300, 360))
        time.sleep(1)
        self.action.left_click()
        time.sleep(5)
        self.action.move_mouse((20, 155))
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
        left, top, right, bot = [int(x * 2.5) for x in [left, top, right, bot]]
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
        return get_dist(img)

    def get_speed(self, img):
        direction = get_direction(img)
        return get_speed(img) * direction

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
        gray = cv2.cvtColor(state_,  cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (112, 112)) / 255.
        self.states.pop(0)
        self.states.append(gray)

        dist_ = self.get_dist(state_)
        if dist_ is None or self.last_dist is None:
            return None, None, None, True
        r = self.calc_reward(self.last_dist, dist_)

        if self.last_dist == dist_:
            self.repeat_nums += 1
            # print(self.repeat_nums)
        else:
            self.repeat_nums = 0

        self.last_dist = dist_

        done = True if self.repeat_nums >= self.repeat_thres else False
        # if done:
        #     r = -10
        return np.stack(self.states, axis=0), r, done, False

    def init_run(self):
        for i in range(1):
            self.step(0)
        time.sleep(5)
