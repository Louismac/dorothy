from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server

dispatcher = Dispatcher()
dispatcher.map("/fft", print)

server = osc_server.ThreadingOSCUDPServer(
    ("127.0.0.1", 5005), dispatcher)
print("Serving on {}".format(server.server_address))
server.serve_forever()