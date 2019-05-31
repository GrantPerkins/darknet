import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn


class MJPGServer:
    def __init__(self, ip='130.215.169.204'):
        self.frame = None
        self._has_started = False
        self._ip = ip

    def started(self):
        return self._has_started

    def start(self, port=8091):
        self._has_started = True
        try:
            CamHandler.set_capture(self)
            with ThreadedHTTPServer((self._ip, int(port)), CamHandler) as server:
                print("Done intializing")
                print("Server starting")
                print(server.server_port)
                server.serve_forever()
        except KeyboardInterrupt:
            # ctrl-c comes here but need another to end all
            print("KeyboardInterrpt in server - ending server")
            server.socket.close()

    def send_image(self, img):
        self.frame = img

    def get_image(self):
        return self.frame


class CamHandler(BaseHTTPRequestHandler):
    capture = None

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=jpgboundary')
        self.end_headers()
        while True:
            img = None
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
        CamHandler.capture = capture


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
