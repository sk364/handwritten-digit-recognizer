"""
DigitDetector class
"""

import csv
import performRecognition
from PIL import Image
from wand.image import Image as WImage


class DigitDetector():
    """
    Detects digits from pdf files
    """

    def __init__(self, filename='test.pdf', bounding_rect=(), num_of_cols=0,
                 num_of_rows=0, resize_ratio=1.0, box_width=0, box_height=0,
                 ss_line_width=0, ss_line_height=0):
        """
        Creates a new object of the class

        Params:
            - filename : PDF file path
            - bounding_rect : area to crop out from PDF
            - num_of_cols : number of columns in the spreadsheet
            - num_of_rows : number of rows in the spreadsheet
            - resize_ratio : Ratio to scale the image
            - box_width : spreadsheet box width
            - box_height : spreadsheet box height
            - ss_line_width : spreadsheet line width between boxes
            - ss_line_height : spreadhsheet line height between boxes
        """

        self._filename = filename
        self._image = self._extract_image_from_pdf()

        if bounding_rect:
            self.bounding_rect = bounding_rect
        else:
            self.bounding_rect = (0, 0,
                                  self._image.size[0], self._image.size[1])

        self.num_of_cols = num_of_cols
        self.num_of_rows = num_of_rows
        self.resize_ratio = resize_ratio
        self.box_width = box_width
        self.box_height = box_height
        self.ss_line_width = ss_line_width
        self.ss_line_height = ss_line_height

    def detect(self):
        """
        Detects and writes the digits in the PDF to a CSV file
        """

        self._crop_image()
        self._rotate_image(angle=90)
        self._resize_image()

        box_regions = self._get_boxes()
        self._write_digits(box_regions)

    def _get_boxes(self):
        """
        Returns a list of box regions
        """

        row_limit = self.num_of_rows*(self.box_height + self.ss_line_height)
        col_limit = self.num_of_cols*(self.box_width + self.ss_line_width)

        box_regions = []

        for i in range(0, row_limit, self.box_height):
            for j in range(0, col_limit, self.box_width):
                if i == 0:
                    box = (j, i + (1 * self.resize_ratio), j + self.box_width,
                           i+self.box_height)
                else:
                    box = (j, i, j + self.box_width, i + self.box_height)

                region = self._image.crop(box)
                box_regions.append(region)

        return box_regions

    def _crop_image(self):
        """
        Crops the spreadsheet area in the image
        """

        self._image = self._image.crop(self.bounding_rect)

    def _rotate_image(self, angle=0):
        """
        Rotates the image at an angle

        Params :
            - angle : angle to rotate the image
        """

        self._image = self._image.rotate(90, expand=True)

    def _resize_image(self):
        """
        Scale the image according to resize ratio
        """

        width, height = self._image.size
        self._image = self._image.resize((height * self.resize_ratio,
                                          width * self.resize_ratio))

    def _extract_image_from_pdf(self):
        with WImage(filename=self._filename) as img:
            img.save(filename='img/input.jpg')

        original_im = Image.open("img/input-0.jpg")

        return original_im

    def _sort_digits(self, digits, key):
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

    def _write_digits(self, box_regions, filename="out.csv"):
        """
        Writes the predicted digits in a CSV file

        Params:
            - box_regions : Regions cropped from image
            - filename : Output filename, Default : out.csv
        """

        cwriter = csv.writer(open(filename, 'w'))

        digit_list = []
        for region in box_regions:
            # performing Neural Network algorithm to get predicted digits
            digits = performRecognition.get_decimal_in_box(region)

            # sort the digits
            digits = self._sort_digits(digits, key=lambda digit: digit[1][0])

            print (digits)

            digit_str = ''
            for digit in digits:
                digit_str += str(digit[0])

            digit_list.append(digit_str)

        # writes the digits into the csv
        for i in range(0, self.num_of_rows*self.num_of_cols, self.num_of_cols):
            cwriter.writerow(digit_list[i:i+self.num_of_cols])
