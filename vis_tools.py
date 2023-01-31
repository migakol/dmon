import cv2 as cv
import sqlite3
from PIL import Image, ImageDraw, ImageFont
import os
from copy import deepcopy

image_folder = '/Volumes/T7/Autobrains/AB_large'
sql_folder = '/Volumes/T7/Autobrains/db.sqlite'
connection = sqlite3.connect(sql_folder)
cursor = connection.cursor()

def get_image_filename(k):
    query = f'SELECT DISTINCT filename FROM auto_brains where id == 450*{k}'
    cursor.execute(query)
    distinct_filenames = []
    for signature_text in cursor.fetchall():
        distinct_filenames.append(signature_text[0])

    return distinct_filenames[0] + '.png'

k1 = 14544
k2 = 14547
# k2 = 31154
img1 = cv.imread(os.path.join(image_folder, get_image_filename(k1)))
img1 = cv.resize(img1, (int(img1.shape[1] / 4), int(img1.shape[0] / 4)))
img1_original = deepcopy(img1)
img2 = cv.imread(os.path.join(image_folder, get_image_filename(k2)))
img2 = cv.resize(img2, (int(img2.shape[1] / 4), int(img2.shape[0] / 4)))
img2_original = deepcopy(img2)

cell_shape = (img1.shape[0] / 15, img1.shape[1] / 30)

colors = [(0, 0, 255), (100, 0, 180), (180, 0, 100), (255, 0, 0), (22, 150, 22), (0, 255, 0)]

# Get the signature of a given position and a given image
def get_signature(imgk, row_pos, col_pos):
    query = f'SELECT signature FROM auto_brains where (id < 450*{imgk} + 1 and id > 450*({imgk}-1)) and pos1 = {row_pos} and pos2 = {col_pos}'
    cursor.execute(query)
    for signature_text in cursor.fetchall():
        signature_text = signature_text[0][1:-1].split(',')
        signature = [int(v) for v in signature_text]
        signature_set = set(signature)
        return signature_set

# Get all the signatures for the second image
img2_sigs = []
for row_pos in range(15):
    row_sig = []
    for col_pos in range(30):
        row_sig.append(get_signature(k2, row_pos, col_pos))
    img2_sigs.append(row_sig)

def jaccard_set(set1, set2):
    """Define Jaccard Similarity function for two sets"""
    intersection = len(set1.intersection(set2))
    union = (len(set1) + len(set2)) - intersection
    return float(intersection) / union

def start_end_point_from_cell(row_pos, col_pos):
    start_point = (int(cell_shape[1] * col_pos), int(cell_shape[0] * row_pos))
    end_point = (int(start_point[0] + cell_shape[1]), int(start_point[1] + cell_shape[0]))

    return start_point, end_point

def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, y)
        # compute the coordinates of the cell
        row = int(y / cell_shape[0])
        col = int(x / cell_shape[0])
        # start_point = (int(cell_shape[1] * col), int(cell_shape[0] * row))
        # end_point = (int(start_point[0] + cell_shape[1]), int(start_point[1] + cell_shape[0]))
        start_point, end_point = start_end_point_from_cell(row, col)

        img1 = deepcopy(img1_original)
        img_to_draw = cv.rectangle(img1, start_point, end_point, color=(255, 0, 0), thickness=2)
        cv.imshow("FIRST", img_to_draw)

        sig = get_signature(k1, row, col)

        # compute the coefficient between the sig to all the sigs from the second image
        img3 = deepcopy(img2_original)
        for row_pos in range(15):
            for col_pos in range(30):
                coef = jaccard_set(sig, img2_sigs[row_pos][col_pos])
                start_point, end_point = start_end_point_from_cell(row_pos, col_pos)
                if coef > 0.3:
                    rec_color = colors[0]
                elif coef > 0.2:
                    rec_color = colors[1]
                elif coef > 0.1:
                    rec_color = colors[2]
                elif coef > 0.05:
                    rec_color = colors[3]
                elif coef > 0.03:
                    rec_color = colors[4]
                else:
                    rec_color = colors[5]
                img3 = cv.rectangle(img3, start_point, end_point, color=rec_color, thickness=2)
        cv.imshow("SECOND", img3)
    #
    #
    # # grab references to the global variables
    # global refPt, cropping
    # # if the left mouse button was clicked, record the starting
	# # (x, y) coordinates and indicate that cropping is being
	# # performed
    # if event == cv2.EVENT_LBUTTONDOWN:
    #     refPt = [(x, y)]
    #     cropping = True
    # # check to see if the left mouse button was released
    # elif event == cv2.EVENT_LBUTTONUP:
	# 	# record the ending (x, y) coordinates and indicate that
	# 	# the cropping operation is finished
	# 	refPt.append((x, y))
	# 	cropping = False
	# 	# draw a rectangle around the region of interest
	# 	cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
	# 	cv2.imshow("image", image)
    #

def display_img(k1, k2):


    cv.imshow("FIRST", img1)
    cv.setMouseCallback("FIRST", mouse_callback)
    cv.imshow("SECOND", img2)
    cv.imshow("THIRD", img2_original)

    cv.waitKey(0)
    cv.destroyAllWindows()



if __name__ == "__main__":
    print('Start visualization')
    display_img(1000, 1000)