import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn


class MJPEGServer:
    """
    The main MJPEG server class. Interfaces with the other classes in this thread.
    """
    def __init__(self, ip):
        """
        Constructs an MJPEGServer object
        :param ip:
        """
        self.frame = None
        self._has_started = False
        self._ip = ip

    def started(self):
        """
        Gets state of server
        :return: whether server has started
        """
        return self._has_started

    def start(self, port):
        """
        Starts MJPEG server. Should be done inside a thread.
        :param port: port to stream to. Use 8190, because team 190 is cool.
        """
        self._has_started = True
        CamHandler.set_capture(self)
        with ThreadedHTTPServer((self._ip, int(port)), CamHandler) as server:
            print("Done intializing")
            print("Server starting")
            print(server.server_port)
            server.serve_forever()

    def send_image(self, img):
        """
        Update the MJPEG stream with current image
        :param img: New image to put
        """
        self.frame = img

    def get_image(self):
        """
        Gets current image
        :return: Current image
        """
        return self.frame


class CamHandler(BaseHTTPRequestHandler):
    """
    Handles the get requests to the server. Overrides do_GET(). Just pushes current image to stream.
    """

    capture = None # MJPEGServer object, used to get frames.

    def do_GET(self):
        """
        Deals with get requests. Puts current frame.
        :return: None
        """
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=jpgboundary')
        self.end_headers()
        while True:
            try:
                # capture image from camera
                img = CamHandler.capture.get_image()
                img_str = cv2.imencode('.jpg', img)[1].tostring()  # change image to jpeg format
                self.send_header('Content-type', 'image/jpeg')
                self.end_headers()
                self.wfile.write(img_str)
                self.wfile.write(b"\r\n--jpgboundary\r\n")  # end of this part
            except ConnectionAbortedError:
                CamHandler.capture.release()

    @staticmethod
    def set_capture(capture):
        """
        Sets static capture variable. Should be the MJPEGServer.
        :param capture: MJPEGServer
        """
        CamHandler.capture = capture


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
