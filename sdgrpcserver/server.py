import argparse, os, sys, threading, signal
from concurrent import futures

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from twisted.web import server, resource, static
from twisted.web.wsgi import WSGIResource 
from twisted.internet import reactor, endpoints

import grpc
import hupper
from sdgrpcserver.sonora.wsgi import grpcWSGI
from wsgicors import CORS

# Google protoc compiler is dumb about imports (https://github.com/protocolbuffers/protobuf/issues/1491)
# TODO: Move to https://github.com/danielgtaylor/python-betterproto
generatedPath = os.path.join(os.path.dirname(__file__), "generated")
sys.path.append(generatedPath)

import generation_pb2_grpc, dashboard_pb2_grpc, engines_pb2_grpc

from sdgrpcserver.manager import EngineMode, EngineManager
from sdgrpcserver.services.dashboard import DashboardServiceServicer
from sdgrpcserver.services.generate import GenerationServiceServicer
from sdgrpcserver.services.engines import EnginesServiceServicer

class DartGRPCCompatibility(object):
    """Fixes a couple of compatibility issues between Dart GRPC-WEB and Sonora

    - Dart GRPC-WEB doesn't set HTTP_ACCEPT header, but Sonora needs it to build Content-Type header on response
    - Sonora sets Access-Control-Allow-Origin to HTTP_HOST, and we need to strip it out so CORSWSGI can set the correct value
    """
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        def wrapped_start_response(status, headers):
            headers = [header for header in headers if header[0] != 'Access-Control-Allow-Origin']
            return start_response(status, headers)
        
        if environ.get("HTTP_ACCEPT") == "*/*":
            environ["HTTP_ACCEPT"] = "application/grpc-web+proto"

        return self.app(environ, wrapped_start_response)

class GrpcServer(object):
    def __init__(self, args):
        host = "[::]" if args.listen_to_all else "localhost"
        port = args.grpc_port

        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        self._server.add_insecure_port(f"{host}:{port}")

    @property
    def grpc_server(self):
        return self._server

    def start(self):
        self._server.start()

    def block(self):
        self._server.wait_for_termination()

    def stop(self, grace=10):
        self._server.stop(grace)

class HttpServer(object):
    def __init__(self, args):
        host = "" if args.listen_to_all else "127.0.0.1"
        port = args.http_port

        # Build the WSGI layer for GRPC-WEB handling
        self._grpcapp = wsgi_app = grpcWSGI(None)
        wsgi_app = DartGRPCCompatibility(wsgi_app)
        wsgi_app = CORS(wsgi_app, headers="*", methods="*", origin="*")

        wsgi_resource = WSGIResource(reactor, reactor.getThreadPool(), wsgi_app)

        # Build the web handler
        controller = RoutingController(args.http_file_root, wsgi_resource)

        # Connect to an endpoint
        site = server.Site(controller)
        endpoint = endpoints.TCP4ServerEndpoint(reactor, port, interface=host)
        endpoint.listen(site)

    @property
    def grpc_server(self):
        return self._grpcapp
    
    def start(self, block=False):
        # Run the Twisted reactor
        self._thread = threading.Thread(target=reactor.run, args=(False,))
        self._thread.start()

    def stop(self, grace=10):
        reactor.callFromThread(reactor.stop)
        self._thread.join(timeout=grace)

class ServerDetails(resource.Resource):
    isLeaf = True
    def render_GET(self, request):
        host = request.getHost()
        request.setHeader(b"Content-type", b"application/json; charset=utf-8")
        return bytes(f'{{"host": "{host.host}", "port": "{host.port}"}}', encoding='utf-8')

class RoutingController(resource.Resource):
    def __init__(self, fileroot, wsgiapp):
        super().__init__()

        self.details = ServerDetails()
        self.fileroot=fileroot
        self.files = static.File(fileroot) if fileroot else None
        self.wsgi=wsgiapp

    def getChild(self, child, request):        
        request.prepath.pop()
        request.postpath.insert(0, child)

        filepath = os.path.join(self.fileroot, *[x.decode() for x in request.postpath])

        if request.postpath[0] == b"server.json":
            return self.details
        elif self.fileroot and os.path.exists(filepath):
            return self.files
        else:
            return self.wsgi

    def render(self, request):
        return self.files.render(request) if self.files else self.wsgi.render(request)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--enginecfg", "-E", type=str, default=os.environ.get("SD_ENGINECFG", "./engines.yaml"), help="Path to the engines.yaml file"
    )
    parser.add_argument(
        "--listen_to_all", "-L", action='store_true', help="Accept requests from the local network, not just localhost" 
    )
    parser.add_argument(
        "--grpc_port", type=int, default=os.environ.get("SD_GRPC_PORT", 50051), help="Set the port for GRPC to run on"
    )
    parser.add_argument(
        "--http_port", type=int, default=os.environ.get("SD_HTTP_PORT", 5000), help="Set the port for HTTP (GRPC-WEB and static files if configured) to run on"
    )
    parser.add_argument(
        "--enable_mps", action="store_true", help="Use MPS on MacOS where available"
    )
    parser.add_argument(
        "--vram_optimisation_level", "-V", type=int, default=os.environ.get("SD_VRAM_OPTIMISATION_LEVEL", 2), help="How much to trade off performance to reduce VRAM usage (0 = none, 2 = max)"
    )
    parser.add_argument(
        "--nsfw_behaviour", "-N", type=str, default=os.environ.get("SD_NSFW_BEHAVIOUR", "block"), choices=["block", "flag"], help="What to do with images detected as NSFW"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Auto-reload on source change"
    )
    parser.add_argument(
        "--weight_root", "-W", type=str, default=os.environ.get("SD_WEIGHT_ROOT", "./weights"), help="Path that local weight in engine.yaml are relative to"
    )
    parser.add_argument(
        "--http_file_root", type=str, default=os.environ.get("SD_HTTP_FILE_ROOT", ""), help="Set this to the root of a filestructure to serve that via the HTTP server (in addition to the GRPC-WEB handler)"
    )
    
    args = parser.parse_args()

    if args.reload:
        # start_reloader will only return in a monitored subprocess
        reloader = hupper.start_reloader('sdgrpcserver.server.main', reload_interval=10)

    grpc = GrpcServer(args)
    grpc.start()

    http = HttpServer(args)
    http.start()

    prevHandler = None
    def shutdown_reactor_handler(*args):
        print("Waiting for server to shutdown...")
        http.stop()
        grpc.stop()
        print("All done. Goodbye.")
        sys.exit(0)

    prevHandler = signal.signal(signal.SIGINT, shutdown_reactor_handler)

    with open(os.path.normpath(args.enginecfg), 'r') as cfg:
        engines = yaml.load(cfg, Loader=Loader)
        manager = EngineManager(
            engines, 
            weight_root=args.weight_root,
            mode=EngineMode(vram_optimisation_level=args.vram_optimisation_level, enable_cuda=True, enable_mps=args.enable_mps), 
            nsfw_behaviour=args.nsfw_behaviour
        )

        generation_pb2_grpc.add_GenerationServiceServicer_to_server(GenerationServiceServicer(manager), grpc.grpc_server)
        dashboard_pb2_grpc.add_DashboardServiceServicer_to_server(DashboardServiceServicer(), grpc.grpc_server)
        engines_pb2_grpc.add_EnginesServiceServicer_to_server(EnginesServiceServicer(manager), grpc.grpc_server)

        generation_pb2_grpc.add_GenerationServiceServicer_to_server(GenerationServiceServicer(manager), http.grpc_server)
        dashboard_pb2_grpc.add_DashboardServiceServicer_to_server(DashboardServiceServicer(), http.grpc_server)
        engines_pb2_grpc.add_EnginesServiceServicer_to_server(EnginesServiceServicer(manager), http.grpc_server)

        print(f"GRPC listening on port {args.grpc_port}, HTTP listening on port {args.http_port}. Start your engines....")

        manager.loadPipelines()

        print("All engines ready")

        # Block until termination
        grpc.block()

