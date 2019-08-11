import cv2

def sobel_image(image):
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)#x方向导数
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)#y方向导数
    gradx = cv2.convertScaleAbs(grad_x)
    grady = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
    return grad