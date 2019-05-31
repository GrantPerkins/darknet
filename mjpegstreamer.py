import cv2
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import time


class MJPGServer:
    def __init__(self):
        self.frames = []
        self._has_started = False

    def started(self):
        return self._has_started

    def start(self, port=1181):
        self._has_started = True
        try:
            CamHandler.set_capture(self)
            server = ThreadedHTTPServer(('localhost', port), CamHandler)
            print("server starting")
            server.serve_forever()

        except KeyboardInterrupt:
            # ctrl-c comes here but need another to end all
            print("KeyboardInterrpt in server - ending server")
            server.socket.close()

    def send_image(self, img):
        self.frames.append(img)

    def get_image(self):
        return self.frames.pop()


class CamHandler(BaseHTTPRequestHandler):
    capture = None

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=jpgboundary')
        self.end_headers()
        while True:
            try:
                # capture image from camera
                try:
                    img = CamHandler.capture.get_image()
                except IndexError:
                    print("Index error yikes")
                    continue

                img_str = cv2.imencode('.jpg', img)[1].tostring()  # change image to jpeg format
                self.send_header('Content-type', 'image/jpeg')
                self.end_headers()
                self.wfile.write(img_str)
                self.wfile.write(b"\r\n--jpgboundary\r\n")  # end of this part
            except KeyboardInterrupt:
                # end of the message - not sure how we ever get here, though
                print("KeyboardInterrpt in server loop - breaking the loop (server now hung?)")
                self.wfile.write(b"\r\n--jpgboundary--\r\n")
                break
            except ConnectionAbortedError:
                CamHandler.capture.release()
        return

    @staticmethod
    def set_capture(capture):
        CamHandler.capture = capture


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""


def main():
    capture = cv2.VideoCapture(0)  # connect openCV to camera 0

    # set desired camera properties
    time.sleep(.5)  # wait for auto exposure change to be set

    # a few other properties that can be set - not a complete list
    #	capture.set(cv2.CAP_PROP_BRIGHTNESS, .4); #1 is bright 0 or-1 is dark .4 is fairly dark default Brightness  0.5019607843137255
    #	capture.set(cv2.CAP_PROP_CONTRAST, 1);
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        CamHandler.set_capture(capture)
        server = ThreadedHTTPServer(('localhost', 1181), CamHandler)
        print("server starting")
        server.serve_forever()

    except KeyboardInterrupt:
        # ctrl-c comes here but need another to end all.  Probably should have terminated thread here, too.
        print("KeyboardInterrpt in server - ending server")
        capture.release()
        server.socket.close()

if __name__ == '__main__':
    main()
