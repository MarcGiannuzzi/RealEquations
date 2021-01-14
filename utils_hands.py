import cv2 as cv
from utils import *
import numpy as np

from utils import *
from utils_writing import *


CONTOUR_APPROXIMATION_MODE = cv.CHAIN_APPROX_SIMPLE  # Or CHAIN_APPROX_SIMPLE
NB_RECTANGLES_TO_GET_HAND_HISTOGRAM = 9
FRAME_SHAPE = (480, 640)
HIST_MASKING_CONVOLUTIONAL_KERNEL_SHAPE = (50, 50)
GAUSSIAN_BLUR_KERNEL_SIZE = 3  # Should be odd
MINIMUM_AREA_OF_MAX_CONTOUR = 8000
MARKER_SIZE = 25  # 20 works well, 30 more easy to use but performs less good

THRESHOLD_FACTOR_HIST_MASKING = 1.5
THRESHOLD_FACTOR_CONTOUR_AREA = 1.75


open_hand_cascade = cv.CascadeClassifier('cascades/open_hand_cascade.xml')
closed_hand_cascade = cv.CascadeClassifier('cascades/closed_hand_cascade.xml')
face_cascade = cv.CascadeClassifier('cascades/face_cascade.xml')


def get_comparison_contour_area_values(hand_hist):
    area_contour_when_hand_closed = 0
    got_area_contour_when_hand_closed = False

    area_contour_when_hand_writing = 0
    got_area_contour_when_hand_writing = False

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        else:
            if not got_area_contour_when_hand_closed:
                area_contour_when_hand_closed = get_area_contour_when_hand_closed(
                    cap, hand_hist)
                got_area_contour_when_hand_closed = True
                cv.destroyAllWindows()
            else:
                if not got_area_contour_when_hand_writing:
                    area_contour_when_hand_writing = get_area_contour_when_hand_writing(
                        cap, hand_hist, area_contour_when_hand_closed)
                    got_area_contour_when_hand_writing = True
                else:
                    break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    print("Area closed hand : ", area_contour_when_hand_closed)
    print("Area hand when writing : ", area_contour_when_hand_writing)
    return (area_contour_when_hand_closed, area_contour_when_hand_writing)


def get_area_contour_when_hand_closed(cap, hand_hist):
    print("Please close your hand.")
    contour_when_hand_closed = 0

    while True:
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)
        hist_mask_image, binary_thresholded = hist_masking(frame, hand_hist)
        head_removed = remove_head(frame, binary_thresholded)
        contours_list = contours(frame, hist_mask_image, binary_thresholded)
        cv.drawContours(frame, contours_list, -1, [0, 0, 0], 3)
        cv.imshow('Get Closed Hand Contour Area', frame)
        if head_removed:
            if len(contours_list) > 0:
                max_cont = max(contours_list, key=cv.contourArea)
                max_area_contour = cv.contourArea(max_cont)
                if max_area_contour > MINIMUM_AREA_OF_MAX_CONTOUR:
                    cnt_centroid = centroid(max_cont)
                    cv.circle(frame, cnt_centroid, 5, [255, 0, 0], -1)
                    contour_when_hand_closed = max_area_contour
        if cv.waitKey(1) == ord('o'):
            break
    return contour_when_hand_closed


def get_area_contour_when_hand_writing(cap, hand_hist, area_contour_when_hand_closed):
    print("Please use your hand as if you were writing.")
    contour_when_hand_writing = 0

    while True:
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)
        hist_mask_image, binary_thresholded = hist_masking(frame, hand_hist)
        head_removed = remove_head(frame, binary_thresholded)
        contours_list = contours(frame, hist_mask_image, binary_thresholded)
        cv.drawContours(frame, contours_list, -1, [255, 255, 255], 3)
        cv.imshow('Get Writing Hand Contour Area', frame)
        if head_removed:
            if len(contours_list) > 0:
                max_cont = max(contours_list, key=cv.contourArea)
                max_area_contour = cv.contourArea(max_cont)
                if max_area_contour > area_contour_when_hand_closed:
                    cnt_centroid = centroid(max_cont)
                    cv.circle(frame, cnt_centroid, 5, [0, 0, 0], -1)
                    contour_when_hand_writing = max_area_contour
        if cv.waitKey(1) == ord('o'):
            break
    return contour_when_hand_writing


def centroid(max_contour):
    """
        @params
        - max_contour : contour with the biggest area from all the contours calculated when trying 
        @returns
        - (cx, cy) : centroid tuple position 
        @description:
            Returns the position of the centroid of the hand.
    """
    moment = cv.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None


def contours(frame, hist_mask_image, binary_thresholded):
    """
        @params
        - hist_mask_image : masked frame containing only the hand
        @returns
        - contours : list of contours positions from the frame. 
        @description:
            Returns a list of contours positions from the frame.
    """
    # gray_hist_mask_image = cv.cvtColor(hist_mask_image, cv.COLOR_BGR2GRAY) # gray_hist_mask_image containing grayscale pixels
    # ret, thresh = cv.threshold(gray_hist_mask_image, minimum_threshold, 255, 0) # thresh containing only black and white pixels
    # cv.GaussianBlur(thresh, (GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_KERNEL_SIZE), 0, thresh) # In order to get full hand contour and not only a subset contour of it
    # cv.imshow("thresh contours", binary_thresholded)
    _, contours_found, hierarchy = cv.findContours(
        binary_thresholded, cv.RETR_EXTERNAL, CONTOUR_APPROXIMATION_MODE)
    return contours_found


def hist_masking(frame, hist):
    """
        @params
        - frame : original frame
        - hist : histogram of the hand
        @returns
        - hist_mask_image : frame containing only the hand 
        @description:
            Returns a masked frame containing the hand only.
    """
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # Depends A LOT on the lightning
    minimum_threshold = np.average(
        hsv[:, :, 2]) * THRESHOLD_FACTOR_HIST_MASKING

    # At each location (x, y) the function collects the values from the selected channels in the input images and finds the corresponding histogram bin. In terms of statistics, the function computes probability of each element value in respect with the empirical probability distribution represented by the histogram
    dst = cv.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
    # Returns a structuring element of the specified size and shape for morphological operations.
    convolution_kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE, HIST_MASKING_CONVOLUTIONAL_KERNEL_SHAPE)

    # Convolves an image with the kernel.
    cv.filter2D(dst, -1, convolution_kernel, dst)
    ret, binary_thresholded = cv.threshold(
        dst, minimum_threshold, 255, cv.THRESH_BINARY)

    binary_thresholded_for_bitwise = cv.merge(
        (binary_thresholded, binary_thresholded, binary_thresholded))
    head_removed = remove_head(frame, binary_thresholded)
    # if not head_removed:
    # closed_hands_kept = keep_only_closed_hands(frame, binary_thresholded)
    #     if not closed_hands_kept:
    #         open_hands_kept = keep_only_open_hands(frame, thresh)
    # remove_open_hands(frame, binary_thresholded)
    cv.imshow("thresh hist_masking", binary_thresholded)
    hist_mask_image = cv.bitwise_and(frame, binary_thresholded_for_bitwise)
    return hist_mask_image, binary_thresholded


def get_farthest_point(defects, max_contour, centroid):
    """
        @params
        - defects : each convexity defect is represented as [[start_index, end_index, farthest_pt_index, fixpt_depth]]. Any deviation of the object from this hull can be considered as convexity defect.
        - max_contour : contour that has the biggest area. Ex :  [[[272 233]], [[271 234]]]
        - centroid : centroid of the max contour area
        @returns
        - farthest_point : farthest point from the centroid 
        @description:
            Get the farthest point from the centroid (center of the hand). That point should be the logically the vertice of the index finger.
    """
    if defects is not None and centroid is not None:
        # np.array([[[1,2,3,4]], [[5,6,7,8]], [[9,10,11,12]]])   => x[:, 0][:, 0]  => array([1, 5, 9])
        contours_start_indexes_defects = defects[:, 0][:, 0]
        x_of_contour_points = np.array(
            max_contour[contours_start_indexes_defects][:, 0][:, 0], dtype=np.float)
        y_of_contour_points = np.array(
            max_contour[contours_start_indexes_defects][:, 0][:, 1], dtype=np.float)

        cx, cy = centroid

        xp = cv.pow(cv.subtract(x_of_contour_points, cx), 2)
        yp = cv.pow(cv.subtract(y_of_contour_points, cy), 2)
        dist = cv.sqrt(cv.add(xp, yp))

        index_max_distance = -1
        not_blocking_counter = 0
        # To focus the index, the position of the point needs to be above the centroid of the hand
        condition_above = False
        while condition_above == False and not_blocking_counter < 5:
            not_blocking_counter += 1
            # index of point from the contour which has the biggest distance from one point of the contour to the centroid of the hand
            index_max_distance = np.argmax(dist)
            # !! Don't change it, this is how it works !!
            if index_max_distance < len(y_of_contour_points):
                if y_of_contour_points[index_max_distance] < cy:
                    condition_above = True
                else:
                    dist[index_max_distance] = -1

        if condition_above == False:
            return None
        else:
            if index_max_distance < len(contours_start_indexes_defects):
                farthest_defect = contours_start_indexes_defects[index_max_distance]
                farthest_point = tuple(max_contour[farthest_defect][0])
                return farthest_point
            else:
                return None


def draw_circles(frame, blackboard, traversed_by_finger_points):
    """
        @params
        - frame : the original frame
        - traversed_by_finger_pointss : list of all the points on which the finger has been
        @returns
        - hist_mask_image : frame containing only the hand 
        @description:
            Main function to draw the circle on each position where the frame has been.
    """
    if traversed_by_finger_points is not None:
        for i in range(len(traversed_by_finger_points)):
            cv.circle(frame, traversed_by_finger_points[i], int(
                5 - (5 * i * 3) / 100), [0, 255, 255], -1)
            cv.circle(blackboard.content,
                      traversed_by_finger_points[i], MARKER_SIZE, 0, -1)


def get_biggest_y_coordinate_of_symbol_contour(blackboard):
    biggest_symbol_contour_y_coordinate = 0  # yes, y is the first coordinate
    ret, inv_blackboard = cv.threshold(blackboard.content.astype(
        np.uint8), 127, 255, cv.THRESH_BINARY_INV)
    _, blackboard_contours_list, hierarchy = cv.findContours(
        inv_blackboard, cv.RETR_EXTERNAL, CONTOUR_APPROXIMATION_MODE)
    if len(blackboard_contours_list) > 0:
        # Normally should have only one element, but for secury use max function on cv.contourArea
        symbol_contour = max(blackboard_contours_list, key=cv.contourArea)
        for coordinates in symbol_contour:
            symbol_contour_y_coordinate = coordinates[0][1]
            if symbol_contour_y_coordinate > biggest_symbol_contour_y_coordinate:
                biggest_symbol_contour_y_coordinate = symbol_contour_y_coordinate
    return biggest_symbol_contour_y_coordinate


def get_lowest_y_coordinate_of_symbol_contour(blackboard):
    # yes, y is the first coordinate
    lowest_symbol_contour_y_coordinate = FRAME_SHAPE[0]
    ret, inv_blackboard = cv.threshold(blackboard.content.astype(
        np.uint8), 127, 255, cv.THRESH_BINARY_INV)
    _, blackboard_contours_list, hierarchy = cv.findContours(
        inv_blackboard, cv.RETR_EXTERNAL, CONTOUR_APPROXIMATION_MODE)
    if len(blackboard_contours_list) > 0:
        # Normally should have only one element, but for secury use max function on cv.contourArea
        symbol_contour = max(blackboard_contours_list, key=cv.contourArea)
        for coordinates in symbol_contour:
            symbol_contour_y_coordinate = coordinates[0][1]
            if symbol_contour_y_coordinate < lowest_symbol_contour_y_coordinate:
                lowest_symbol_contour_y_coordinate = symbol_contour_y_coordinate
    return lowest_symbol_contour_y_coordinate


def get_biggest_x_coordinate_of_symbol_contour(blackboard):
    biggest_symbol_contour_x_coordinate = 0  # yes, y is the first coordinate
    ret, inv_blackboard = cv.threshold(blackboard.content.astype(
        np.uint8), 127, 255, cv.THRESH_BINARY_INV)
    _, blackboard_contours_list, hierarchy = cv.findContours(
        inv_blackboard, cv.RETR_EXTERNAL, CONTOUR_APPROXIMATION_MODE)
    if len(blackboard_contours_list) > 0:
        # Normally should have only one element, but for secury use max function on cv.contourArea
        symbol_contour = max(blackboard_contours_list, key=cv.contourArea)
        for coordinates in symbol_contour:
            symbol_contour_x_coordinate = coordinates[0][0]
            if symbol_contour_x_coordinate > biggest_symbol_contour_x_coordinate:
                biggest_symbol_contour_x_coordinate = symbol_contour_x_coordinate
    return biggest_symbol_contour_x_coordinate


def get_lowest_x_coordinate_of_symbol_contour(blackboard):
    # yes, x is the second coordinate
    lowest_symbol_contour_x_coordinate = FRAME_SHAPE[1]
    ret, inv_blackboard = cv.threshold(blackboard.content.astype(
        np.uint8), 127, 255, cv.THRESH_BINARY_INV)
    _, blackboard_contours_list, hierarchy = cv.findContours(
        inv_blackboard, cv.RETR_EXTERNAL, CONTOUR_APPROXIMATION_MODE)
    if len(blackboard_contours_list) > 0:
        # Normally should have only one element, but for secury use max function on cv.contourArea
        symbol_contour = max(blackboard_contours_list, key=cv.contourArea)
        for coordinates in symbol_contour:
            symbol_contour_x_coordinate = coordinates[0][0]
            if symbol_contour_x_coordinate < lowest_symbol_contour_x_coordinate:
                lowest_symbol_contour_x_coordinate = symbol_contour_x_coordinate
    return lowest_symbol_contour_x_coordinate


def manage_image_opr(frame, blackboard, hand_hist, contour_when_hand_closed, contour_when_hand_writing):
    """
        @params
        - frame : the original frame
        - hand_hist : histogram of the hand
        @returns
        - None
        @description:
            Operational function to follow the finger.
    """
    traversed_by_finger_points = []
    # need to find the tip of a finger with the convexity defect, which is furthest from the centroid of the contour.
    hist_mask_image, binary_thresholded = hist_masking(frame, hand_hist)
    contours_list = contours(frame, hist_mask_image, binary_thresholded)

    if len(contours_list) > 0:
        cv.drawContours(frame, contours_list, -1, [255, 0, 255])
        max_cont = max(contours_list, key=cv.contourArea)
        max_area_contour = cv.contourArea(max_cont)
        cnt_centroid = centroid(max_cont)
        cv.circle(frame, cnt_centroid, 5, [0, 0, 255], -1)

        # print("Current hand contour area : ", max_area_contour)
        if max_cont is not None and max_area_contour > contour_when_hand_closed / THRESHOLD_FACTOR_CONTOUR_AREA and max_area_contour < contour_when_hand_writing * THRESHOLD_FACTOR_CONTOUR_AREA:
            # hull calculated by cv.convexHull on the biggest contour
            hull = cv.convexHull(max_cont, returnPoints=False)
            # each convexity defect is represented as (start_index, end_index, farthest_pt_index, fixpt_depth)
            defects = cv.convexityDefects(max_cont, hull)
            farthest_point = get_farthest_point(
                defects, max_cont, cnt_centroid)
            if farthest_point is not None:
                # print("Centroid : " + str(cnt_centroid) + ", farthest Point : " + str(farthest_point))
                # cv.circle(frame, farthest_point, 5, [0, 0, 255], -1)

                if len(traversed_by_finger_points) < 20:
                    traversed_by_finger_points.append(farthest_point)
                else:
                    traversed_by_finger_points.pop(0)
                    traversed_by_finger_points.append(farthest_point)

                draw_circles(frame, blackboard, traversed_by_finger_points)
                return farthest_point


def remove_head(frame, gray):
    """
        @params
        - frame : Original frame
        @returns
        - frame : frame without head
        @description:
            Removes the head from the frame because head is same color than hand and disturbs hand detection
    """
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    gray_shape = gray.shape
    gray_max_x = gray.shape[1] - 1
    gray_max_y = gray.shape[0] - 1
    is_work_done = False
    for (x, y, w, h) in faces:
        can_stop = False
        for head_side_distance in [50, 25, 0]:
            for upper_head_distance in [200, 150, 100, 50, 25, 0]:
                if y - upper_head_distance > 0 and x + head_side_distance < gray_max_x and x - head_side_distance > 0 and can_stop == False:
                    gray[:, x: x + w + head_side_distance] = 0
                    gray[:, x - w - head_side_distance:x] = 0
                    can_stop = True
                    is_work_done = True

    return is_work_done


def find_biggest_region(regions):
    max_w = 0
    max_h = 0
    index_biggest_region = 0
    biggest_region = None
    for index_region, (x, y, w, h) in enumerate(regions):
        if w > max_w and h > max_h:
            index_biggest_region = index_region
            biggest_region = regions[index_biggest_region]

    return biggest_region


def keep_only_open_hands(frame, gray):
    open_hands = open_hand_cascade.detectMultiScale(frame, 1.3, 5)
    gray_max_x = gray.shape[1] - 1
    gray_max_y = gray.shape[0] - 1
    is_work_done = False
    if len(open_hands) != 0:
        can_stop = False
        biggest_hand_region = find_biggest_region(open_hands)
        (x, y, w, h) = biggest_hand_region
        for padding in [200, 150, 100, 50, 25, 0]:
            if y - padding > 0 and y + h + padding < gray_max_y and x - padding > 0 and x + w + padding < gray_max_x and can_stop == False:
                cv.rectangle(frame, (x - padding, y - padding),
                             (x + w + padding, y + h + padding), (0, 255, 0), 2)

                # Remove above region of hand
                gray[:y - padding, :] = 0

                # Remove upper region of hand
                gray[y + h + padding:, :] = 0

                # Remove left region of hand
                gray[:, :x - padding] = 0

                # Remove right region of hand
                gray[:, x + w + padding:] = 0

                can_stop = True
                is_work_done = True
    return is_work_done


def remove_open_hands(frame, gray):
    open_hands = open_hand_cascade.detectMultiScale(frame, 1.3, 5)
    gray_max_x = gray.shape[1] - 1
    gray_max_y = gray.shape[0] - 1
    is_work_done = False
    if len(open_hands) != 0:
        can_stop = False
        biggest_hand_region = find_biggest_region(open_hands)
        (x, y, w, h) = biggest_hand_region
        for padding in [200, 150, 100, 50, 25, 0]:
            if y - padding > 0 and y + h + padding < gray_max_y and x - padding > 0 and x + w + padding < gray_max_x and can_stop == False:
                cv.rectangle(frame, (x - padding, y - padding),
                             (x + w + padding, y + h + padding), (0, 255, 0), 2)

                # Remove all the hand
                gray[y - padding: y + h + padding,
                     x - padding: x + w + padding] = 0

                can_stop = True
                is_work_done = True
    return is_work_done


def keep_only_closed_hands(frame, gray):
    closed_hands = closed_hand_cascade.detectMultiScale(frame, 1.3, 5)
    gray_max_x = gray.shape[1] - 1
    gray_max_y = gray.shape[0] - 1
    is_work_done = False
    if len(closed_hands) != 0:
        can_stop = False
        biggest_hand_region = find_biggest_region(closed_hands)
        (x, y, w, h) = biggest_hand_region
        for padding in [200, 150, 100, 50, 25, 0]:
            if y - padding > 0 and y + h + padding < gray_max_y and x - padding > 0 and x + w + padding < gray_max_x and can_stop == False:
                cv.rectangle(frame, (x - padding, y - padding - 100),
                             (x + w + padding, y + h + padding), (0, 0, 255), 2)

                # Remove above region of hand and give some place to finger
                gray[:y - padding - 100, :] = 0

                # Remove upper region of hand
                gray[y + h + padding:, :] = 0

                # Remove left region of hand
                gray[:, :x - padding] = 0

                # Remove right region of hand
                gray[:, x + w + padding:] = 0

                can_stop = True
                is_work_done = True
    return is_work_done


def draw_rectangles_hand_recognition(frame):
    """
        @params
        - frame : original frame
        @returns
        - frame : original frame 
        - [hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y] to hold the coordinates of each rectangle for the histogram masking
        @description:
            Draws the rectangles that help the use to position it's hand in order to get hand histogram in hand_histogram
    """
    rows, cols, _ = frame.shape
    denominator = 20
    hand_rect_one_x = np.array(
        [6 * rows / denominator, 6 * rows / denominator, 6 * rows / denominator, 9 * rows / denominator, 9 * rows / denominator, 9 * rows / denominator, 12 * rows / denominator,
         12 * rows / denominator, 12 * rows / denominator], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / denominator, 10 * cols / denominator, 11 * cols / denominator, 9 * cols / denominator, 10 * cols / denominator, 11 * cols / denominator, 9 * cols / denominator,
         10 * cols / denominator, 11 * cols / denominator], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(NB_RECTANGLES_TO_GET_HAND_HISTOGRAM):
        cv.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                     (hand_rect_two_y[i], hand_rect_two_x[i]),
                     (0, 255, 0), 1)

    return frame, [hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y]


def get_hand_histogram():
    """
        @params
        @returns
        - hand_hist : the histogram of the hand  
        @description:
            Main function for us to get the hand histogram of the user.
    """
    hand_hist = None
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        else:
            frame, list_rectangles = draw_rectangles_hand_recognition(frame)
            cv.imshow('frame in get_hand_histogram', rescale_frame(frame))
            hand_rect_one_x = list_rectangles[0]
            hand_rect_one_y = list_rectangles[1]
            hand_hist = hand_histogram(frame, hand_rect_one_x, hand_rect_one_y)

        if cv.waitKey(1) == ord('o'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    return hand_hist


def hand_histogram(frame, hand_rect_one_x, hand_rect_one_y):
    """
        @params
        - frame : original frame
        - hand_rect_one_x, hand_rect_one_y : rectangles to get the colors from hand in order to create the hand histogram
        @returns
        - hand_hist_normalized : histogram of the hand normalized
        @description:
            Returns a normalized histogram of the hand.
    """
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    for i in range(NB_RECTANGLES_TO_GET_HAND_HISTOGRAM):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                                    hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    hand_hist = cv.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    hand_hist_normalized = cv.normalize(
        hand_hist, hand_hist, 0, 255, cv.NORM_MINMAX)
    return hand_hist_normalized


def are_contours_equal(contour1, contour2):
    bool_result = True
    if len(contour1) != len(contour2):
        return False
    else:
        for index_array in range(len(contour1)):
            equalities_arrays = contour1[index_array] == contour2[index_array]
            if not np.all(equalities_arrays == [True, True]):
                return False
    return bool_result


def remove_isolated_points_from_blackboard(blackboard):
    ret, inv_blackboard = cv.threshold(
        blackboard.content, 127, 255, cv.THRESH_BINARY_INV)  # white on black
    _, blackboard_contours, hierarchy = cv.findContours(
        inv_blackboard.astype(np.uint8), cv.RETR_EXTERNAL, CONTOUR_APPROXIMATION_MODE)
    max_contour = max(blackboard_contours, key=cv.contourArea)
    blackboard_content_shape = blackboard.content.shape
    new_blackboard_content = np.zeros(
        (blackboard_content_shape[0], blackboard_content_shape[1], 3)).astype(np.uint8)
    for contour in blackboard_contours:
        if not are_contours_equal(contour, max_contour):
            cv.drawContours(blackboard.content, [
                            contour], -1, (255, 255, 255), -1)

    # area_max_contour = cv.contourArea(max_contour)
    # if area_max_contour < blackboard_content_shape[0] * blackboard_content_shape[1] - 5000:
    #     cv.drawContours(new_blackboard_content, [max_contour], -1, (255, 255, 255), -1)
    #     new_blackboard_content_gray = cv.cvtColor(new_blackboard_content, cv.COLOR_BGR2GRAY)
    #     ret, blackboard.content = cv.threshold(new_blackboard_content_gray, 127, 255, cv.THRESH_BINARY_INV)


def create_image_to_predict(blackboard):
    remove_isolated_points_from_blackboard(blackboard)
    # cv.GaussianBlur(blackboard.content, (GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_KERNEL_SIZE), 0, blackboard.content) # blur just a bit in order to have values different than 0 and 255
    # dilate with black on white == erode with white on black
    blackboard.content = cv.dilate(blackboard.content, (15, 15), iterations=10)

    biggest_y_coordinate_of_symbol_contour = get_biggest_y_coordinate_of_symbol_contour(
        blackboard)
    lowest_y_coordinate_of_symbol_contour = get_lowest_y_coordinate_of_symbol_contour(
        blackboard)

    biggest_x_coordinate_of_symbol_contour = get_biggest_x_coordinate_of_symbol_contour(
        blackboard)
    lowest_x_coordinate_of_symbol_contour = get_lowest_x_coordinate_of_symbol_contour(
        blackboard)

    image_to_predict = blackboard.content[lowest_y_coordinate_of_symbol_contour:biggest_y_coordinate_of_symbol_contour,
                                          lowest_x_coordinate_of_symbol_contour:biggest_x_coordinate_of_symbol_contour]

    image_to_predict = cv.resize(image_to_predict, (45, 45)) / 255
    image_to_predict = np.expand_dims(image_to_predict, axis=2)
    return image_to_predict


def hand_operations():
    """
        @params
        - hand_hist : histogram of the hand
        @returns
        - None
        @description:
            Function to follow the finger and display the frame.
    """
    blackboard = Blackboard()
    model, history = load_model("64C_32C_16C_256D_Drop05_20epochs_cnn_model")

    hand_hist = get_hand_histogram()
    contour_when_hand_closed, contour_when_hand_writing = get_comparison_contour_area_values(
        hand_hist)

    string_computation = ""

    cap = cv.VideoCapture(0)
    last_keyboard = ''  # to avoid multi predictions when pressing r
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        else:
            frame = cv.flip(frame, 1)
            farthest_point = manage_image_opr(
                frame, blackboard, hand_hist, contour_when_hand_closed, contour_when_hand_writing)
            cv.imshow('Main Frame', rescale_frame(frame))
            cv.imshow('Blackboard', blackboard.content)

        if cv.waitKey(1) == ord('q'):
            break
        # There might be some delay (functions take time to finish before arriving to this point), so user should keep the key pressed.
        elif cv.waitKey(1) == ord('e'):
            blackboard.erase()
            last_keyboard = 'e'
        elif cv.waitKey(1) == ord('r') and last_keyboard != 'r':  # Try to predict the symbol
            image_to_predict = create_image_to_predict(blackboard)
            cv.imshow("Image to predict", image_to_predict)
            output_predictions = model.predict(np.array([image_to_predict]))
            print("\n\nOutput vector : ", output_predictions)
            math_element = TOTAL_MATH_SYMBOLS[np.argmax(output_predictions[0])]
            print("Predicted : ", math_element)
            string_computation += math_element
            last_keyboard = 'r'
        elif cv.waitKey(1) == ord('b'):
            cv.destroyAllWindows()
            hand_hist = get_hand_histogram()
            contour_when_hand_closed, contour_when_hand_writing = get_comparison_contour_area_values(
                hand_hist)
            cap = cv.VideoCapture(0)
            last_keyboard = 'b'
        elif cv.waitKey(1) == ord('c') and string_computation != "":
            int_computation = eval(string_computation)
            print("Result equation : ", int_computation)
            string_computation = ""

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    hand_operations()
