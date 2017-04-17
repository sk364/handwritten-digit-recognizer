"""
Test Script to extract and write digits from handwritten data provided in PDF
"""

import csv
import performRecognition
import sys
from PIL import Image
from wand.image import Image as WImage


# CONFIG
RESIZE_RATIO = 3
WIDTH, HEIGHT = 26*RESIZE_RATIO, 11*RESIZE_RATIO
PIXEL_WIDTH, PIXEL_HEIGHT = 1*RESIZE_RATIO, 1*RESIZE_RATIO
NUM_OF_COLS, NUM_OF_ROWS = 2, 2


def sort_digits(digits, key):
    """
    Returns updated list of digits sorted from left to right in the image

    Params:
        - digits : List of tuples containing digit and bounding rectangle of
          each digit
        - key : Sort the digits according to this key
    """

    if not digits:
        return []

    # sort according to x-coordinate of bounding rectangle
    digits = sorted(digits, key=key)

    return digits

def write_to_csv(box_regions, num_of_cols, num_of_rows, filename="out.csv"):
    """
    Writes the predicted digits in a CSV file

    Params:
        - box_regions : Regions cropped from image
        - num_of_cols : Number of columns in spreadsheet
        - num_of_rows : Number of rows in spreadsheet
        - filename : Output filename, Default : out.csv
    """

    cwriter = csv.writer(open(filename, 'w'))

    digit_list = []
    for region in box_regions:
        # performing Neural Network algorithm to get predicted digits
        digits = performRecognition.get_decimal_in_box(region)

        # sort the digits
        digits = sort_digits(digits, key=lambda digit: digit[1][0])

        print (digits)

        digit_str = ''
        for digit in digits:
            digit_str += str(digit[0])

        digit_list.append(digit_str)

    # writes the digits into the csv
    for i in range(0, num_of_rows*num_of_cols, num_of_cols):
        cwriter.writerow(digit_list[i:i+num_of_cols])

# driver section
if __name__ == '__main__':
    with WImage(filename="test.pdf") as img:
        img.save(filename="img/input.jpg")

    original_im = Image.open("img/input-0.jpg")

    big_box = (220, 241, 458, 647)
    im = original_im.crop(big_box)

    im_width, im_height = im.size

    im = im.rotate(90, expand=True)
    im = im.resize((im_height*RESIZE_RATIO, im_width*RESIZE_RATIO))

    """with open('img/crop_bb.jpg', 'wb') as crop_bb:
        im.save(crop_bb)"""

    box_regions = []

    for i in range(0, NUM_OF_ROWS*(HEIGHT+PIXEL_HEIGHT), HEIGHT+PIXEL_HEIGHT):
        for j in range(0, NUM_OF_COLS*(WIDTH+PIXEL_WIDTH), WIDTH+PIXEL_WIDTH):
            if i == 0:
                box = (j, i+(1*RESIZE_RATIO), j+WIDTH, i+HEIGHT)
            else:
                box = (j, i, j+WIDTH, i+HEIGHT)

            region = im.crop(box)
            box_regions.append(region)

    write_to_csv(box_regions, NUM_OF_COLS, NUM_OF_ROWS)

    print "\nComplete\n"
