import os
import cv2
import time
import darknet
import mjpegstreamer
from threading import Thread
import argparse


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0], \
                     detection[2][1], \
                     detection[2][2], \
                     detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        print(detection[0].decode(), str(round(detection[1] * 100, 2)) + "%")
        print("X:", round(x))
        print("Y:", round(y))
        print("W:", round(w))
        print("H:", round(h))
        print("\n")
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


def YOLO(width=640, height=480, ip='10.1.90.2', port="8190", source=0):
    # TODO: make these cmd line arguments, no default video
    configPath = "./model.cfg"
    weightPath = "./model.weights"
    metaPath = "./model.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath) + "`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath) + "`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath) + "`")
    netMain = darknet.load_net_custom(configPath.encode(
        "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    metaMain = darknet.load_meta(metaPath.encode("ascii"))

    if str(source).isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    cap.set(3, width)
    cap.set(4, height)
    print("Start YOLO processing...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                       darknet.network_height(netMain), 3)
    mjpeg = mjpegstreamer.MJPEGServer(ip)
    try:
        while True:
            prev_time = time.time()
            ret, frame_read = cap.read()
            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb,
                                       (darknet.network_width(netMain),
                                        darknet.network_height(netMain)),
                                       interpolation=cv2.INTER_LINEAR)

            darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
            image = cvDrawBoxes(detections, frame_resized)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mjpeg.send_image(image)
            if not mjpeg.started():
                mpjeg_server_thread = Thread(target=mjpeg.start, args=(port,), daemon=True)
                mpjeg_server_thread.start()
    except KeyboardInterrupt:
        import sys
        sys.exit()

    #         print(1/(time.time()-prev_time))
    cap.release()
    out.release()





if __name__ == "__main__":
    parser = argparse.ArgumentParser("WPILib implementation of darknet")
    parser.add_argument('--ip', dest="ip", default='10.1.90.2',
                        help="Change the ip address where the MJPEG is streamed")
    parser.add_argument('--height', dest="height", default=480, type=int, help="Change the height of the input frame")
    parser.add_argument('--width', dest="width", default=640, type=int, help="Change the width of the input frame")
    parser.add_argument('--source', dest="source", default='0',
                        help="Change the source of the video. Can either be a camera id (0, 1, ...) or a file location (any OpenCV supported file type, .mp4, .webm, etc.)")
    args = parser.parse_args()
    YOLO(ip=args.ip, height=args.height, width=args.width, source=args.source)
