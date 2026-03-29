"""Borderless Win32 window setup and display helpers (Windows only)."""
import cv2
import numpy as np
import ctypes


class Display_Functions:
    """Static methods for creating and positioning a borderless OpenCV window via Win32 API."""
    @staticmethod
    def display_image(image_name, image, multiply_binary_255, resize_res, window_position = [None, None]):
        disp_image = image.copy()
        if (multiply_binary_255 == True):
            disp_image *= 255
        disp_image = cv2.resize(disp_image, resize_res, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(image_name, disp_image)
        if window_position:
            cv2.moveWindow(image_name, window_position[0], window_position[1])

    @staticmethod
    def init_borderless_window(window_name, display_res, position):
        blank = np.zeros((display_res[1], display_res[0], 3), dtype=np.uint8)
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window_name, blank)
        cv2.waitKey(50)
        hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
        if hwnd:
            GWL_STYLE   = -16
            GWL_EXSTYLE = -20
            WS_POPUP    = 0x80000000
            WS_VISIBLE  = 0x10000000
            WS_EX_WINDOWEDGE = 0x00000100
            WS_EX_CLIENTEDGE = 0x00000200
            SWP_NOMOVE       = 0x0002
            SWP_NOZORDER     = 0x0004
            SWP_FRAMECHANGED = 0x0020
            # Replace style with bare popup — no caption, no resize border
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_STYLE, WS_POPUP | WS_VISIBLE)
            # Strip raised/sunken edge extended styles
            ex = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex & ~(WS_EX_WINDOWEDGE | WS_EX_CLIENTEDGE))
            ctypes.windll.user32.SetWindowPos(hwnd, None, 0, 0,
                                              display_res[0], display_res[1],
                                              SWP_NOMOVE | SWP_NOZORDER | SWP_FRAMECHANGED)
            # Suppress DWM drop shadow
            try:
                DWMWA_NCRENDERING_POLICY = 2
                DWMNCRP_DISABLED = 1
                policy = ctypes.c_int(DWMNCRP_DISABLED)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, DWMWA_NCRENDERING_POLICY,
                    ctypes.byref(policy), ctypes.sizeof(policy))
            except Exception:
                pass
        cv2.moveWindow(window_name, position[0], position[1])

    @staticmethod
    def get_screen_resolution():
        w = ctypes.windll.user32.GetSystemMetrics(0)
        h = ctypes.windll.user32.GetSystemMetrics(1)
        return (w, h)