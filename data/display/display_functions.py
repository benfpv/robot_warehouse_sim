import cv2
import numpy as np

class Display_Functions:
    def display_image(image_name, image, multiply_binary_255, resize_res, window_position = [None, None]):
        #print("- display_image(): {}".format(image_name))
        disp_image = image.copy()
        if (multiply_binary_255 == True):
            disp_image *= 255
        disp_image = cv2.resize(disp_image, resize_res, interpolation=cv2.INTER_NEAREST)
        disp_image = cv2.imshow(image_name, disp_image)
        if window_position:
            disp_image = cv2.moveWindow(image_name, window_position[0], window_position[1])
        #print("- image_name: {}, disp_image: {}".format(image_name, disp_image))
        return disp_image