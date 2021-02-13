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
    file.write("file,count")
    file.write("\n")
    for i in range(0, len(my_data)):
        data = "video" + str(i + 1) + ".mp4," + str(my_data[i])
        file.write(data)
        file.write("\n")
    file.write("ZRNIC MILOS SW93-2017")
    file.close()


def write_test_data(my_data):
    file = open('test_distance.txt', "a")
    file.write("Distance,Frame,MAE")
    file.write("\n")
    for i in my_data:
        data = str(i) + "  " + str(my_data[i])
        file.write(data)
        file.write("\n")
    file.close()


def plato_line(firstVideo):
    vid_name = "data/" + firstVideo + ".mp4"
    capture = cv2.VideoCapture(vid_name)
    frame_cnt = 0
    if not capture.isOpened():
        print("Error while opening video")

    # kroz frejmove, get(1) broj sledeceg frejma  get(7) ukupan broj frejmova
    while capture.get(1) < capture.get(7) - 1:
        frame_cnt += 1
        ret_val, frame1 = capture.read()
        if frame_cnt == capture.get(7) - 1:
            # Bitno nam je samo jedan  frejm da iskoristimo za fiksiranje platoa zato sto kameru ne pomeramo
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
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # cv2.line(line_image, (x1, y1 + 150), (x2, y2 + 150), (255, 0, 0), 2)
                    cv2.imshow("resultimage", line_image)
                    cv2.waitKey(2)
                    # Spustamo liniju, optimalno sam pronasao da za vrednosti otp y=250 daje najbolje rezulate
            return x1, y1 + 150, x2, y2 + 150


def pedestrian_cross_detection(pedestrian_position, line_position, distance=7.6):
    if abs(line_position - pedestrian_position) < distance:
        # print("Pedestrian detected")
        return True


def is_pedestrian(width, height, minWidth=10, maxWidth=80, minHeight=12, maxHeight=80):
    return minWidth < width < maxWidth or minHeight < height < maxHeight


def opening(image, kernel, it=2):
    eroded = cv2.erode(image, kernel, iterations=it)
    dilated = cv2.dilate(eroded, kernel, iterations=it)
    return dilated


def closing(image, kernel, it=2):
    dilated = cv2.dilate(image, kernel, iterations=it)
    eroded = cv2.erode(dilated, kernel, iterations=it)
    return eroded


def count_people(video_path, y1, y2):
    number_of_people = 0
    frame_test = 5
    # crossing line y cordinate
    line_y = (y1 + y2) / 2
    # Neuspeli pokusaj da brojim izmedju dve linije pesake
    # line_y2 = (y1 + y2 + 35) / 2

    capture = cv2.VideoCapture(video_path)
    _, frame = capture.read()
    current_frame = frame
    counter = -1

    while capture.get(1) < capture.get(7) - 1:
        counter += 1
        _, next_frame = capture.read()

        if counter % frame_test != 0:
            continue

        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        # smoothing to remove noise, ne daje bolji rez
        # current_frame_gray = cv2.GaussianBlur(current_frame_gray, (21, 21), 0)

        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
        # smoothing to remove noise, ne daje bolji rez
        # next_frame_gray = cv2.GaussianBlur(next_frame_gray, (21, 21), 0)
        current_frame = next_frame
        cop = next_frame.copy()

        # Now in order to detect motion, we will compare the pixel intensities of current and next frame
        difference = cv2.absdiff(current_gray, next_gray)

        # v.2
        threshold = cv2.threshold(difference, 20, 255, cv2.THRESH_BINARY)[1]
        # cv2.imshow("Threshold", threshold)
        # cv2.waitKey(5)

        # ostale sve su mi davale manju tacnost od obicne 3x3
        kernel = np.ones((3, 3))

        threshold = closing(threshold, kernel)
        # threshold = opening(threshold)
        # threshold = cv2.dilate(threshold, None, iterations=2)
        # threshold = cv2.erode(threshold, k, iterations=2)
        # Sa None vrednosti kernela, MAE 2.7
        # threshold = cv2.erode(threshold, None, iterations=2)
        # cv2.imshow("Threshold", threshold)
        # cv2.waitKey(40)

        # cv2.imshow("Threshold", threshold)
        # cv2.waitKey(5)
        contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        mostValues = []
        mostValuesH = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            (x, y), (width, height), _ = rect
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # cv2.drawContours(cop, [box], 0, (0, 255, 0), 2)
            # cv2.imshow("resultimage", cop)
            # cv2.waitKey(1)
            mostValues.append(width)
            mostValuesH.append(height)
            if is_pedestrian(width, height):
                # if cv2.contourArea(contour) > 130: IPAK NE
                # Korisceno za testiranje
                # pedestrian_cross_detection(y, line_y, distance_test):
                if pedestrian_cross_detection(y, line_y):
                    number_of_people += 1
    capture.release()
    return number_of_people


def test_distance(y1, y2):
    results = load_res()
    res = {}
    ran = [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]
    ran1 = [7.4, 7.5, 7.6, 7.7, 7.8, 7.9]
    f = [3, 4, 5, 6, 7, 8, 9, 10]
    # for distance in range(2,10):
    for distance in ran1:
        for frame in f:
            my_results = []
            err_sum = 0
            for vid_index in range(1, 11):
                people_counted = 0
                vid_name = "data/video" + str(vid_index) + ".mp4"
                people_counted = count_people(vid_name, y1, y2, frame, distance)
                my_results.append(people_counted)
                # print("video number: " + str(vid_index))
                # print("num of people: " + str(people_counted))
                err = abs(float(results[vid_index - 1]) - people_counted)
                err_sum += err
            res[str(distance) + " " + str(frame)] = " " + str(err_sum / len(results))
            print("Distance: " + str(distance) + "MAE:" + str(err_sum / len(results)) + "|Frame" + str(frame))

    write_test_data(res)


def main():
    # Samo koristimo da bi pronasli liniju na platou u jednom frejmu
    x1, y1, x2, y2 = plato_line("video1")
    results = load_res()
    my_results = []
    err_sum = 0
    file = open("out.txt", "w")
    if not file:
        print("Somethings wrong with file writer!")
        return -1
    file.write("file,count\n")
    # Pomocu pretrage nasao da za 7.5-7.8 daje najbolje rezultate
    # test_distance(y1, y2)
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
