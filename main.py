import cv2
from PIL import Image
import numpy as np
import os

# creating or cleaning directories
try:
    os.mkdir('found')
except FileExistsError:
    for root, dirs, files in os.walk('found'):
        for f in files:
            os.unlink(os.path.join(root, f))
try:
    os.mkdir('found_pad')
except FileExistsError:
    for root, dirs, files in os.walk('found_pad'):
        for f in files:
            os.unlink(os.path.join(root, f))

# flags and changeable elements
height_pad = 5  # best possible choosed
width_pad = 2  # best possible choosed
threshold = 245  # best possible choosed

# searching all shapes in the input image
no_gray = cv2.imread(filename='trebuchet_bold.png', flags=cv2.IMREAD_GRAYSCALE)
pic2 = cv2.imread(filename='trebuchet_bold.png')
gray = cv2.imread(filename='trebuchet_bold.png', flags=cv2.IMREAD_GRAYSCALE)
gray = np.array(gray)
height, width = gray.shape
white = 0
for h in range(height):
    for w in range(width):
        if gray[h, w] == [255]:
            white = white + 1
if white < (height * width) / 2:
    gray = cv2.cv2.bitwise_not(gray)
ret, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_OTSU)
inverted_binary = ~binary
contours, hierarchy = cv2.findContours(inverted_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# deleting shapes included in bigger ones
k = 0
found_letter = []
found_best = []
found_best_r = []
for c in contours:
    found_letter.append(c)


def selection(contours_sel):
    iter1 = 0
    for cs in contours_sel:
        iter1 = iter1 + 1
        iter2 = 0
        xs, ys, ws, hs = cv2.boundingRect(cs)
        found_best.append(cs)
        for ds in contours_sel:
            iter2 = iter2 + 1
            if iter1 == iter2:
                pass
            else:
                xs2, ys2, ws2, hs2 = cv2.boundingRect(ds)
                if (xs >= xs2) and ((xs + ws) <= (xs2 + ws2)) and (ys >= ys2) and ((ys + hs) <= (ys2 + hs2)):
                    found_best.pop()
                    pass


pic1 = no_gray
selection(found_letter)
number_of_found_letters = len(found_best)  # number of found shapes
print("OCR have found ", number_of_found_letters, " letters")
found_better = []

for c in found_best:
    x, y, w, h = cv2.boundingRect(c)
    found_better.append(c)
    if (cv2.contourArea(c)) > 5:
        cv2.rectangle(pic1, (x, y), (x + w, y + h), (77, 22, 174), 2)
    else:
        found_better.pop()
cv2.imwrite('All contours with bounding box.png', pic1)
number_of_found_letters = len(found_better)  # number of found shapes
print("OCR have found ", number_of_found_letters, " letters")

found_letter_pad = []


def resize_with_pad(image, target_width, target_height, pr):  # function adding pads to found shapes
    background = Image.new('RGBA', (target_width, target_height), (255, 255, 255, 255))
    offset = (round((target_width - image.width) / 2), round((target_height - image.height) / 2))
    background.paste(image, offset)
    background.save('found_pad/found_letter_pad_' + str(pr) + '.png')
    img_s = cv2.imread('found_pad/found_letter_pad_' + str(pr) + '.png')
    found_letter_pad.append(img_s)


p = 0
for c in found_better:
    x, y, w, h = cv2.boundingRect(c)
    p = p + 1
    im = gray[y - height_pad:y + h + height_pad, x - width_pad:x + w + width_pad]  # cutting shapes from input image
    cv2.imwrite('found/found_letter_' + str(p) + '.png', im)
    im = Image.fromarray(im)
    im2 = cv2.imread('found/found_letter_' + str(p) + '.png')

    resize_with_pad(im, w + 40, h + 40, p)
