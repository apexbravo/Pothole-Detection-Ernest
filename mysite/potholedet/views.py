from django.http import HttpResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import cv2 as cv
import numpy as np
# Rest of your pothole detection code
# importing necessary libraries
import cv2 as cv
import time
import geocoder
import os
from django.conf import settings
# add these lines at the beginning of the script
import numpy as np

from .models import Pothole


def index(request) -> HttpResponse:
    return render(request, 'potholedet/index.html')


def videotest(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        video_path = os.path.join(settings.BASE_DIR, 'media', video_file.name)
        with open(video_path, 'wb') as f:
            for chunk in video_file.chunks():
                f.write(chunk)

        # reading label name from obj.names file
        class_name = []
        with open(os.path.join(settings.BASE_DIR, "project_files", 'obj.names'), 'r') as f:
            class_name = [cname.strip() for cname in f.readlines()]

        # importing model weights and config file
        # defining the model parameters
        net1 = cv.dnn.readNet(os.path.join(settings.BASE_DIR, 'project_files', 'yolov4_tiny.weights'),
                              os.path.join(settings.BASE_DIR, 'project_files', 'yolov4_tiny.cfg'))
        net1.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net1.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
        model1 = cv.dnn_DetectionModel(net1)
        model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

        # defining the video source (0 for camera or file name for video)
        cap = cv.VideoCapture(video_path)
        width = cap.get(3)
        height = cap.get(4)
        result_path = os.path.join(
            settings.BASE_DIR, 'potholedet', 'static', 'result.mp4')
        result = cv.VideoWriter(result_path,
                                cv.VideoWriter_fourcc(*'avc1'),
                                10, (int(width), int(height)))

        # defining parameters for result saving and get coordinates
        # defining initial values for some parameters in the script
        g = geocoder.ip('me')
        result_path = os.path.join(
            settings.BASE_DIR, 'pothole_coordinates')
        starting_time = time.time()
        Conf_threshold = 0.5
        NMS_threshold = 0.4
        frame_counter = 0
        i = 0
        b = 0

        # detection loop
        while True:
            ret, frame = cap.read()
            frame_counter += 1
            if ret == False:
                break

            # lane detection
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            blurred = cv.GaussianBlur(gray, (5, 5), 0)
            edges = cv.Canny(blurred, 50, 150)
            roi = edges[300:480, 150:450]
            lines = cv.HoughLinesP(roi, rho=1, theta=1*np.pi/180,
                                   threshold=20, minLineLength=50, maxLineGap=5)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv.line(frame, (x1+150, y1+300),
                            (x2+150, y2+300), (0, 0, 255), 2)
            # analysis the stream with detection model
            classes, scores, boxes = model1.detect(
                frame, Conf_threshold, NMS_threshold)
            for (classid, score, box) in zip(classes, scores, boxes):
                label = "pothole"
                x, y, w, h = box
                recarea = w*h
                area = width*height
                # drawing detection boxes on frame for detected potholes and saving coordinates txt and photo
                if (len(scores) != 0 and scores[0] >= 0.7):
                    if ((recarea/area) <= 0.1 and box[1] < 600):
                        cv.rectangle(
                            frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                        cv.putText(frame, "%" + str(round(scores[0]*100, 2)) + " " + label,
                                   (box[0], box[1]-10), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
                        if (i == 0):
                            cv.imwrite(os.path.join(
                                result_path, 'pothole'+str(i)+'.jpg'), frame)
                            with open(os.path.join(result_path, 'pothole'+str(i)+'.txt'), 'w') as f:
                                f.write(str(g.latlng))
                                i = i+1
                        if (i != 0):
                            if ((time.time()-b) >= 2):
                                cv.imwrite(os.path.join(
                                    result_path, 'pothole'+str(i)+'.jpg'), frame)
                                with open(os.path.join(result_path, 'pothole'+str(i)+'.txt'), 'w') as f:
                                    f.write(str(g.latlng))
                                    latitude, longitude = g.latlng
                                    pothole = Pothole(
                                        latitude=latitude, longitude=longitude)
                                    pothole.save()
                                    b = time.time()
                                    i = i+1

            # writing fps on frame
            endingTime = time.time() - starting_time
            fps = frame_counter/endingTime
            cv.putText(frame, f'FPS: {fps}', (20, 50),
                       cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            # showing and saving result
            cv.imshow('frame', frame)
            result.write(frame)
            key = cv.waitKey(1)
            if key == ord('q'):
                break

        # end
        cap.release()
        result.release()
        cv.destroyAllWindows()

        # Return the processed video or display it on the page
        context = {'video_path': video_path}
        return render(request, 'potholedet/pages/result.html', context)

    return render(request, 'potholedet/pages/videotest.html')

    # add this function to the script


def detect_lanes(image):
    # convert image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # apply Gaussian blur to reduce noise
    blur = cv.GaussianBlur(gray, (5, 5), 0)

    # detect edges using Canny edge detection algorithm
    edges = cv.Canny(blur, 50, 150)

    # define a region of interest (ROI)
    height, width = edges.shape
    roi = np.array([[(0, height), (width // 2, height // 2),
                     (width, height), (0, height)]], dtype=np.int32)

    # apply ROI mask to the edges image
    masked_edges = np.zeros_like(edges)
    cv.fillPoly(masked_edges, roi, 255)
    masked_edges = cv.bitwise_and(edges, masked_edges)

    # detect lines using Hough line transform
    lines = cv.HoughLinesP(masked_edges, rho=2, theta=np.pi/180,
                           threshold=50, minLineLength=50, maxLineGap=100)

    # draw detected lines on the original image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return image


def result(request):
    # Process any necessary data or calculations
    # ...

    video_path = os.path.join(settings.BASE_DIR, 'media', 'result.mp4')

    # Render the result page with the video path
    return render(request, 'potholedet/pages/result.html', {'video_path': video_path})


def potholemap(request):
    potholes = Pothole.objects.all()
    coordinates = [(pothole.latitude, pothole.longitude)
                   for pothole in potholes]

    context = {'coordinates': coordinates}
    return render(request, 'potholedet/pages/potholemap.html', context)


def about(request):
    return render(request, 'potholedet/pages/about.html')
