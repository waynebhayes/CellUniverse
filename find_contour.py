import cv2


def contour(find_contour_from, add_contour_to, save_to, i):
    print('begin to find contour in the output images...')
    for num in range(i):
        img1 = cv2.imread(find_contour_from + 'frame%03d.png' % num)
        gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        image, cnts, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img2 = cv2.imread(str(add_contour_to[num]))
        cv2.drawContours(img2, cnts, -1, (0, 0, 255), 1)
        cv2.imwrite(save_to + '/output_' + str(num) + '.png', img2)
    print('finish find contour')
