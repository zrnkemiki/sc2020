import numpy as np
import cv2
from matplotlib import pyplot as plt

def load_res():
    number_of_people_res = []
    with open('data/res.txt') as file:
        file.readline()
        for line in file:
            data = line.split(',')
            video_name = data[0].strip()
            people = data[1].strip()
            number_of_people_res.append(people)
    return number_of_people_res

def write_data(my_data):
    file = open('out.txt', "w")
    file.write("SW93-2017/Milos Zrnic")
    file.write("file,count")
    file.write("\n")
    for i in range (0,len(my_data)):
        data = "video" + str(i+1) + ".mp4," + str(my_data[i])
        file.write(data)
        file.write("\n")
    file.close()


def adaptive_threshold(image):
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 4)

# def dilate(image, iterations, kernel=kernel_size):
#    return cv2.dilate(image, kernel, iterations=iterations)


# def erode(image, iterations, kernel=kernel_size):
#    return cv2.erode(image, kernel, iterations=iterations)


# def closing(image, dilate_iterations, erode_iterations, dilate_kernel=kernel_size, erode_kernel=kernel_size):
#    return erode(dilate(image, dilate_iterations, dilate_kernel), erode_iterations, erode_kernel)


def plato_line(firstVideo):
    vid_name = "data/" + firstVideo + ".mp4"
    capture = cv2.VideoCapture(vid_name)
    frame_cnt = 0
    if not capture.isOpened():
        print("Something's off! Error while opening video")

    # kroz frejmove
    while capture.get(1) < capture.get(7) - 1:
        frame_cnt += 1
        ret_val, frame1 = capture.read()

    # Bitno nam je samo prvi frejm da iskoristimo za fiksiranje platoa zato sto kameru ne pomeramo
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

    low_threshold = 170
    high_threshold = 250

    edges = cv2.Canny(frame1, low_threshold, high_threshold)

    rho = 1
    theta = np.pi / 180
    min_line_length = 250  # Minimum line length. Line segments shorter than that are rejected.
    max_line_gap = 20
    line_image = np.copy(frame1)  # creating an image copy to draw lines on
    # Run Hough on the edge-detected image
    lines = cv2.HoughLinesP(edges, rho, theta, cv2.THRESH_BINARY, np.array([]),
                            min_line_length, max_line_gap)
    # Iterate over the output "lines" and draw lines on the image copy
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1 + 150), (x2, y2 + 150), (255, 0, 0), 2)
            cv2.imshow("resultimage", line_image)
            cv2.waitKey(2)
            # Spustamo liniju, optimalno sam pronasao da za vrednosti y=250 daje najbolje rezulate
    return x1, y1 + 149, x2, y2 + 149


def pedestrian_cross_detection(y, yy):
    if abs(yy - y) < 7.5:
        print("Pedestrian detected")
        return True


def count_people(video_path, y1, y2):
    number_of_people = 0

    # crossing line y cordinate
    line_y = (y1 + y2) / 2
    # Neuspeli pokusaj da brojim izmedju dve linije pesake
    #line_y2 = (y1 + y2 + 35) / 2

    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()
    current_frame = frame
    counter = -1

    while True:
        loaded, next_frame = cap.read()

        if not loaded:
            break

        counter += 1

        #Svaki 5frejm daje mi dobre rez
        if counter != 5:
            continue
        counter = 0

        # current_frame = imutils.resize(current_frame, width=500)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        # smoothing to remove noise, ne daje bolji rez
        # current_frame_gray = cv2.GaussianBlur(current_frame_gray, (21, 21), 0)

        # ext_frame = imutils.resize(next_frame, width=500)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
        # smoothing to remove noise, ne daje bolji rez
        # next_frame_gray = cv2.GaussianBlur(next_frame_gray, (21, 21), 0)
        current_frame = next_frame
        cop = next_frame.copy()

        # Now in order to detect motion, we will compare the pixel intensities of current and next frame
        difference = cv2.absdiff(current_gray, next_gray)
        # algorithm determines the threshold for a pixel based on a small region around it
        # threshold = cv2.adaptiveThreshold(difference, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 4)
        # dilation iter =1 , erode iter = 3
        # closing_segment = closing(diff_bin, 1, 3)
        # contours, _ = cv2.findContours(diff_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # v.2
        threshold = cv2.threshold(difference, 20, 255, cv2.THRESH_BINARY)[1]
        # threshold = closing(threshold, 1, 3)
        # cv2.imshow("Threshold", threshold)
        # cv2.waitKey(5)

        threshold = cv2.dilate(threshold, None, iterations=2)
        #cv2.imshow("Threshold", threshold)
        #cv2.waitKey(5)
        contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            rect = cv2.minAreaRect(contour)
            (x, y), (width, height), _ = rect
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # cv2.drawContours(cop, [box], 0, (0, 255, 0), 2)
            # cv2.imshow("resultimage", cop)
            # cv2.waitKey(1)
            if 10 < width < 70 or 10 < height < 70:
            #if cv2.contourArea(contour) > 150:
                if pedestrian_cross_detection(y, line_y):
                    number_of_people += 1
    cap.release()
    return number_of_people


def main():
    # Samo koristimo da bi pronasli liniju na platou u prvom frejmu
    x1, y1, x2, y2 = plato_line("video1")
    results = load_res()
    my_results = []
    err_sum = 0
    file = open("out.txt", "w")
    if not file:
        print("Somethings wrong with file writer!")
        return -1
    file.write("file,count\n")

    for vid_index in range(1, 11):
        people_counted = 0
        vid_name = "data/video" + str(vid_index) + ".mp4"

        people_counted = count_people(vid_name, y1, y2)
        my_results.append(people_counted)
        print("video number: " + str(vid_index))
        print("num of people: " + str(people_counted))
        err = abs(float(results[vid_index - 1]) - people_counted)
        err_sum += err
    print("\nMAE: ", err_sum / len(results))
    write_data(my_results)


main()
