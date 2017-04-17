"""
Test program to implement handwritten digit recognition using neural networks
"""

from DigitDetector import DigitDetector

# CONFIG
BOUNDING_RECT = (220, 241, 458, 647)
RESIZE_RATIO = 3
WIDTH, HEIGHT = 26*RESIZE_RATIO, 11*RESIZE_RATIO
PIXEL_WIDTH, PIXEL_HEIGHT = 1*RESIZE_RATIO, 1*RESIZE_RATIO
NUM_OF_COLS, NUM_OF_ROWS = 2, 2

if __name__ == '__main__':
    dc = DigitDetector(filename='./test.pdf', bounding_rect=BOUNDING_RECT,
                       num_of_cols=NUM_OF_COLS, num_of_rows=NUM_OF_ROWS,
                       box_width=WIDTH, box_height=HEIGHT,
                       resize_ratio=RESIZE_RATIO, ss_line_width=PIXEL_WIDTH,
                       ss_line_height=PIXEL_HEIGHT)

    dc.detect()

    print "\n!! Complete !!\n"
