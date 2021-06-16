import cv2
import numpy as np


def get_contour_center(contour):
    m = cv2.moments(contour)
    if m["m00"] == 0:
        return [0, 0]
    x = int(m["m10"] / m["m00"])
    y = int(m["m01"] / m["m00"])
    return [x, y]


def process(image):
    _, thresh = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)  # Get Threshold
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Get contouro
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        # height, width = image.shape[:2]
        cv2.drawContours(image, [main_contour], -1, (150, 150, 150), 2)
        # middle_x = int(width / 2)
        # middle_y = int(height / 2)
        contour_center = get_contour_center(main_contour)
        cv2.circle(image, tuple(contour_center), 2, (150, 150, 150), 2)
        return image, contour_center
    else:
        return image, (0, 0)


def slice_out(im, num):
    cont_cent = list()

    height, width = im.shape[:2]
    sl = int(height / num)
    sliced_imgs = list()
    for i in range(num):
        part = sl * i
        crop_img = im[part:part + sl, 0:width]
        # images[i].image = crop_img
        processed = process(crop_img)
        # cv2.imshow(str(i), processed[0])
        sliced_imgs.append(processed[0])
        cont_cent.append(processed[1])
    # print(cont_cent)
    return sliced_imgs, cont_cent


def remove_background(image, b):
    up = 100
    lo = 0
    # create NumPy arrays from the boundaries
    lower = np.array([lo], dtype="uint8")
    upper = np.array([up], dtype="uint8")
    # ----------------COLOR SELECTION-------------- (Remove any area that is whiter than 'upper')
    if b is True:
        mask = cv2.inRange(image, lower, upper)
        image = cv2.bitwise_and(image, image, mask=mask)
        image = cv2.bitwise_not(image, image, mask=mask)
        image = (255 - image)
        return image
    else:
        return image


def repack(images):
    im = images[0]
    for i in range(len(images)):
        if i == 0:
            im = np.concatenate((im, images[1]), axis=0)
        if i > 1:
            im = np.concatenate((im, images[i]), axis=0)
    return im


def line(image, center, cont_cent):
    cv2.line(image, (0, 65), (480, 65), (30, 30, 30), 1)
    cv2.line(image, (0, 130), (480, 130), (30, 30, 30), 1)
    cv2.line(image, (0, 195), (480, 195), (30, 30, 30), 1)
    cv2.line(image, (240, 0), (240, 260), (30, 30, 30), 1)
    for i in range(len(cont_cent)):
        cv2.line(image, center, (cont_cent[i][0], cont_cent[i][1]+65*i), (100, 100, 100), 1)


def main():
    cap = cv2.VideoCapture('line.mp4')
    no_slice = 4
    center = (240, 227)
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is not None:
            frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = remove_background(img, True)
            slices, cont_cent = slice_out(img, no_slice)
            img = repack(slices)
            line(img, center, cont_cent)
            cv2.imshow('frame', img)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
