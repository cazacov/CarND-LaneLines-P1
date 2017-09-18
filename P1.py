# importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import math


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_poly(img, lines, color=[255, 0, 0]):
    pts = lines.reshape((-1, 1, 2))
    cv2.fillConvexPoly(img, pts, color)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


import os

os.listdir("test_images/")


def filter_low_slope(lines):
    result = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if math.fabs(slope) < math.tan(30.0 * math.pi / 180.0):
                # Skip lines with slope below 30 degrees
                continue
            else:
                result.append(line)

    return np.array(result)


# Choose the longest left and right line
def find_left_right(lines):
    left_lane = [0, 0, 0, 0]
    left_lane_len = 0
    right_lane = [0, 0, 0, 0]
    right_lane_len = 0

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)

        line_len = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
        if slope > 0:
            if line_len > right_lane_len:
                right_lane_len = line_len
                right_lane = line
        else:
            if line_len > left_lane_len:
                left_lane_len = line_len
                left_lane = line

    result = []
    if right_lane_len > 0:
        result.append(right_lane)
    if left_lane_len > 0:
        result.append(left_lane)

    return np.array(result)


def extend_lines_to_border(lines, xsize, ysize):
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            b = y1 - slope * x1
            xborder = (ysize - b) / slope
            if y2 > y1:
                line[0, 3] = ysize
                line[0, 2] = xborder
            else:
                line[0, 1] = ysize
                line[0, 0] = xborder

    return lines


def draw_poly(img, lines, color=[0, 255, 0]):
    if lines is not None:
        pts = lines.reshape((-1, 1, 2))
        cv2.fillConvexPoly(img, pts, color)


def cut_lines(lines, xsize, ysize):
    if lines.shape[0] != 2:
        # Can work only with exactly two lines
        return

    # find intersection point
    lineA = np.polyfit((lines[0][0][0], lines[0][0][2]), (lines[0][0][1], lines[0][0][3]), 1)
    lineB = np.polyfit((lines[1][0][0], lines[1][0][2]), (lines[1][0][1], lines[1][0][3]), 1)

    x = (lineB[1] - lineA[1]) / (lineA[0] - lineB[0])
    y = x * lineA[0] + lineA[1]

    cut_y = y + (ysize - y) * 0.1  # Take bottom 90% below the intersection point

    result = []

    result.append(
        [[
            (ysize - lineA[1]) / lineA[0],
            ysize,
            (cut_y - lineA[1]) / lineA[0],
            cut_y
        ]]
    )

    result.append(
        [[
            (cut_y - lineB[1]) / lineB[0],
            cut_y,
            (ysize - lineB[1]) / lineB[0],
            ysize,
        ]]
    )

    return np.array(result).astype(int)


def find_lines(gray, xsize, ysize):
    # Smooth
    blur = gaussian_blur(gray, kernel_size=5)

    # Detect edges
    low_threshold = 50
    high_threshold = 192
    edges = canny(blur, low_threshold, high_threshold)

    # Cut region of interest
    mask_verticles = np.array([[
        (0, ysize),
        (xsize * 0.4, ysize * 0.5),
        (xsize * 0.6, ysize * 0.5),
        (xsize, ysize)
    ]], dtype=np.int32)
    masked_image = region_of_interest(edges, mask_verticles)

    # Find lines
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 75  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 180  # minimum number of pixels making up a line
    max_line_gap = 150
    lines = cv2.HoughLinesP(masked_image, rho, theta, threshold, np.array([]), minLineLength=min_line_length,
                            maxLineGap=max_line_gap)

    # Skip those with low slope (tan < 0.3)
    filtered_lines = filter_low_slope(lines)

    # Choose the longest left and right lines
    filtered_lines = find_left_right(filtered_lines)

    return filtered_lines


def process_image(image):
    xsize = image.shape[1]
    ysize = image.shape[0]

    original_image = np.copy(image)

    # Convert to grayscale
    gray = grayscale(image)

    r, g, b = cv2.split(image)

    test_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(test_image)

    lines_gray = find_lines(gray, xsize, ysize)
    lines_b = find_lines(b, xsize, ysize)
    lines_s = find_lines(s, xsize, ysize)

    # Find the best color space
    # Try to use gray channel first. If no lines found, fallback to G channel of RGB, then fallbach to S channel of HLS
    if lines_gray.shape[0] >= 2:
        lines = lines_gray
    elif lines_b.shape[0] >= 2:
        lines = lines_b
    else:
        lines = lines_s

    lane_image = np.zeros((ysize, xsize, 3), dtype=np.uint8)

    lane = extend_lines_to_border(lines, xsize, ysize)
    nice_lane = cut_lines(lane, xsize, ysize)
    draw_lines(lane_image, lane, thickness=5)
    #draw_poly(lane_image, nice_lane)

    result = weighted_img(original_image, lane_image, 0.4, 1)

    return result


def pipeline(image_name):
    image = mpimg.imread(image_name)

    result = process_image(image)

    if not os.path.exists('test_images_output'):
        os.makedirs('test_images_output')

    output_name = 'test_images_output/' + os.path.basename(image_name)
    plt.imshow(result)
    #plt.savefig(output_name)
    plt.show()


pipeline("frame5.jpg")
# pipeline("test_images/solidWhiteCurve.jpg")
# pipeline("test_images/solidWhiteRight.jpg")
# pipeline("test_images/solidYellowCurve.jpg")
# pipeline("test_images/solidYellowCurve2.jpg")
# pipeline("test_images/solidYellowLeft.jpg")
# pipeline("test_images/whiteCarLaneSwitch.jpg")