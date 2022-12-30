import argparse
import hashlib
import os
import re
import secrets
import shutil
import signal
import sys
import threading
import time
from concurrent import futures

import yaml

from sdgrpcserver.pipeline.xformers_utils import xformers_mea_available
from sdgrpcserver.ram_monitor import RamMonitor

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import grpc
import hupper
import torch
from twisted.internet import endpoints, protocol, reactor
from twisted.web import resource, server, static
from twisted.web.resource import ForbiddenResource, NoResource
from twisted.web.wsgi import WSGIResource
from wsgicors import CORS

from sdgrpcserver.sonora.wsgi import grpcWSGI

# Google protoc compiler is dumb about imports (https://github.com/protocolbuffers/protobuf/issues/1491)
# TODO: Move to https://github.com/danielgtaylor/python-betterproto
generatedPath = os.path.join(os.path.dirname(__file__), "generated")
sys.path.append(generatedPath)

# Inject the nonfree projects if they exist
nonfree_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nonfree")
if os.path.exists(nonfree_path):
    sys.path.append(os.path.join(nonfree_path, "ToMe"))

import dashboard_pb2_grpc
import engines_pb2_grpc
import generation_pb2_grpc

from sdgrpcserver.debug_recorder import DebugNullRecorder, DebugRecorder
from sdgrpcserver.http.grpc_gateway import GrpcGatewayRouter
from sdgrpcserver.manager import BatchMode, EngineManager, EngineMode
from sdgrpcserver.services.dashboard import DashboardServiceServicer
from sdgrpcserver.services.engines import EnginesServiceServicer
from sdgrpcserver.services.generate import GenerationServiceServicer


class DartGRPCCompatibility(object):
    """Fixes a couple of compatibility issues between Dart GRPC-WEB and Sonora

    - Dart GRPC-WEB doesn't set HTTP_ACCEPT header, but Sonora needs it to build Content-Type header on response
    - Sonora sets Access-Control-Allow-Origin to HTTP_HOST, and we need to strip it out so CORSWSGI can set the correct value
    """

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        def wrapped_start_response(status, headers):
            headers = [
                header
                for header in headers
                if header[0] != "Access-Control-Allow-Origin"
            ]
            return start_response(status, headers)

        if environ.get("HTTP_ACCEPT") == "*/*":
            environ["HTTP_ACCEPT"] = "application/grpc-web+proto"

        return self.app(environ, wrapped_start_response)


class CheckAuthHeaderMixin(object):
    def _checkAuthHeader(self, value):
        token = re.match("Bearer\s+(.*)", value, re.IGNORECASE)
        if token and token[1] == self.access_token:
            return True
        return False


class GrpcServerTokenChecker(grpc.ServerInterceptor, CheckAuthHeaderMixin):
    def __init__(self, key):
        self.access_token = key

        def deny(_, context):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid key")

        self._deny = grpc.unary_unary_rpc_method_handler(deny)

    def intercept_service(self, continuation, handler_call_details):
        metadatum = handler_call_details.invocation_metadata

        for meta in metadatum:
            if meta.key == "authorization":
                if self._checkAuthHeader(meta.value):
                    return continuation(handler_call_details)

        return self._deny


class GrpcServer(object):
    def __init__(self, args):
        host = "[::]" if args.listen_to_all else "localhost"
        port = args.grpc_port

        interceptors = []
        if args.access_token:
            interceptors.append(GrpcServerTokenChecker(args.access_token))

        maxMsgLength = 30 * 1024 * 1024  # 30 MB

        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=4),
            interceptors=interceptors,
            options=[
                ("grpc.max_message_length", maxMsgLength),
                ("grpc.max_send_message_length", maxMsgLength),
                ("grpc.max_receive_message_length", maxMsgLength),
            ],
        )
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
        self.controller = RoutingController(
            args.http_file_root, wsgi_resource, access_token=args.access_token
        )

        # Connect to an endpoint
        site = server.Site(self.controller)
        endpoint = endpoints.TCP4ServerEndpoint(reactor, port, interface=host)
        endpoint.listen(site)

    @property
    def grpc_server(self):
        return self._grpcapp

    @property
    def grpc_gateway(self):
        return self.controller.grpc_gateway

    def start(self, block=False):
        # Run the Twisted reactor
        self._thread = threading.Thread(target=reactor.run, args=(False,))
        self._thread.start()

    def stop(self, grace=10):
        reactor.callFromThread(reactor.stop)
        self._thread.join(timeout=grace)


class LocaltunnelServer(object):
    class LTProcessProtocol(protocol.ProcessProtocol):
        def __init__(self, access_token):
            self.access_token = (
                access_token  # Just used to print out with address later
            )
            self.received_address = False

        def connectionMade(self):
            self.transport.closeStdin()

        def outReceived(self, data):
            data = data.decode("utf-8")
            print("Received unexpected output from localtunnel:")
            print("  ", data)

        def outReceived(self, err):
            err = err.decode("utf-8")
            m = re.search("url is: https://(.*)$", err, re.M)
            if m:
                self.received_address = True
                print(f"Localtunnel started. Use these settings to connect:")
                print(f"    Server '{m[1]}'")
                print(f"    Port '443'")
                print(f"    Key '{self.access_token}'")

            else:
                print("Received unexpected error from localtunnel:")
                print("  ", err)

        def processExited(self, status):
            if not self.received_address:
                print(
                    "Didn't receive an address from localtunnel before it shut down. Please check your installation."
                )

    def __init__(self, args):
        self.access_token = args.access_token
        self.internal_port = args.http_port

    def start(self):
        npx_path = shutil.which("npx")
        if not npx_path:
            raise NotImplementedError(
                "You need an npx install in your path to run localtunnel"
            )

        self.proc = reactor.spawnProcess(
            LocaltunnelServer.LTProcessProtocol(self.access_token),
            executable=npx_path,
            args=["npx", "localtunnel", "--port", str(self.internal_port)],
            env=os.environ,
        )

    def stop(self, grace=10):
        self.proc.signalProcess("TERM")

        for _ in range(grace):
            if not self.proc.pid:
                return
            time.sleep(1)

        print("Hard killing LT")
        self.proc.signalProcess("KILL")


class ServerDetails(resource.Resource):
    isLeaf = True

    def render_GET(self, request):
        host = request.getHost()
        request.setHeader(b"Content-type", b"application/json; charset=utf-8")
        return bytes(
            f'{{"host": "{host.host}", "port": "{host.port}"}}', encoding="utf-8"
        )


class RoutingController(resource.Resource, CheckAuthHeaderMixin):
    def __init__(self, fileroot, wsgiapp, access_token=None):
        super().__init__()

        self.details = ServerDetails()
        self.fileroot = fileroot
        self.files = static.File(fileroot) if fileroot else None
        self.grpc_gateway = GrpcGatewayRouter()
        self.wsgi = wsgiapp

        self.access_token = access_token

    def _checkAuthorization(self, request):
        if not self.access_token:
            return True
        if request.method == b"OPTIONS":
            return True

        authHeader = request.getHeader("authorization")
        if authHeader:
            if self._checkAuthHeader(authHeader):
                return True

        return False

    def getChild(self, child, request):
        if not self._checkAuthorization(request):
            return ForbiddenResource()

        # -- These handlers are all nested

        # Hardcoded handler for service discovery
        if child == b"server.json":
            return self.details

        # Pass off stability REST API
        if child == b"v1alpha":
            print("Stability REST API not implemented yet")
            return NoResource()

        if child == b"grpcgateway":
            return self.grpc_gateway

        # -- These handler are all overlapped on root

        # Resert path to include the first section of the URL
        request.prepath.pop()
        request.postpath.insert(0, child)

        # Pass off GRPC-WEB requests (detect via content-type header)
        content_type = request.getHeader("content-type")
        if content_type and content_type.startswith("application/grpc-web"):
            return self.wsgi

        # If we're serving files, check to see if the request is for a served file
        if self.fileroot:
            return self.files

        return NoResource()

    def render(self, request):
        if not self._checkAuthorization(request):
            return ForbiddenResource().render(request)

        print("render", request)

        if self.files:
            return self.files.render(request)

        return NoResource().render(request)


def git_object_hash(bs: bytes):
    hasher = hashlib.sha1()
    hasher.update(b"blob ")
    hasher.update(bytes(str(len(bs)), "utf-8"))
    hasher.update(b"\0")
    hasher.update(bs)
    return hasher.hexdigest()


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    generation_opts = parser.add_argument_group("generation")
    networking_opts = parser.add_argument_group("networking")
    util_opts = parser.add_argument_group("utility")
    batch_opts = parser.add_argument_group("generation batch control")
    debug_opts = parser.add_argument_group("debugging")

    networking_opts.add_argument(
        "--listen_to_all",
        "-L",
        action="store_true",
        help="Accept requests from the local network, not just localhost",
    )
    networking_opts.add_argument(
        "--grpc_port",
        type=int,
        default=os.environ.get("SD_GRPC_PORT", 50051),
        help="Set the port for GRPC to run on",
    )
    networking_opts.add_argument(
        "--http_port",
        type=int,
        default=os.environ.get("SD_HTTP_PORT", 5000),
        help="Set the port for HTTP (GRPC-WEB and static files if configured) to run on",
    )
    networking_opts.add_argument(
        "--access_token",
        type=str,
        default=os.environ.get("SD_ACCESS_TOKEN", None),
        help="Set a single access token that must be provided to access this server",
    )
    networking_opts.add_argument(
        "--localtunnel",
        action="store_true",
        help="Expose HTTP to public internet over localtunnel.me. If you don't specify an access token, setting this option will add one for you.",
    )

    generation_opts.add_argument(
        "--enginecfg",
        "-E",
        type=str,
        default=os.environ.get("SD_ENGINECFG", "./engines.yaml"),
        help="Path to the engines.yaml file",
    )
    generation_opts.add_argument(
        "--weight_root",
        "-W",
        type=str,
        default=os.environ.get("SD_WEIGHT_ROOT", "./weights"),
        help="Path that local weights in engine.yaml are relative to",
    )
    generation_opts.add_argument(
        "--refresh_models",
        "-r",
        type=str,
        default=os.environ.get("SD_REFRESH_MODELS", None),
        help="'*' or a comma-seperated list of model path globs to refresh even if a local cache exists (missing models will always be downloaded)",
    )
    generation_opts.add_argument(
        "--dont_refresh_on_error",
        action="store_true",
        help="If a model exists in cache but fails to load, don't try redownloading (to fix a potentially corrupted download.)",
    )
    generation_opts.add_argument(
        "--vram_optimisation_level",
        "-V",
        type=int,
        default=os.environ.get("SD_VRAM_OPTIMISATION_LEVEL", 2),
        help="How much to trade off performance to reduce VRAM usage (0 = none, 2 = max)",
    )
    generation_opts.add_argument(
        "--nsfw_behaviour",
        "-N",
        type=str,
        default=os.environ.get("SD_NSFW_BEHAVIOUR", "block"),
        choices=["block", "flag", "ignore"],
        help="What to do with images detected as NSFW",
    )
    generation_opts.add_argument(
        "--supress_metadata",
        action="store_true",
        help="Supress storing request metadata in returned PNGs",
    )
    generation_opts.add_argument(
        "--enable_mps", action="store_true", help="Use MPS on MacOS where available"
    )

    batch_opts.add_argument(
        "--batch_autodetect",
        action="store_true",
        help="Determine the maximum batch size automatically",
    )
    batch_opts.add_argument(
        "--batch_autodetect_margin",
        type=float,
        default=os.environ.get("SD_BATCH_AUTODETECT_MARGIN", 0.2),
        help="The fraction of memory that should be reserved when autodetecting batch max",
    )
    batch_opts.add_argument(
        "--batch_points",
        type=str,
        default=os.environ.get("SD_BATCH_POINTS", None),
        help="A JSON string of (pixels, batch_max) points, usually the output of running batch_autodetect",
    )
    batch_opts.add_argument(
        "--batch_max",
        type=int,
        default=os.environ.get("SD_BATCH_MAX", 1),
        help="A fixed maximum number of generations to run in a batch. Overriden by batch_points or batch_autodetect if provided.",
    )

    util_opts.add_argument(
        "--reload", action="store_true", help="Auto-reload on source change"
    )
    util_opts.add_argument(
        "--http_file_root",
        type=str,
        default=os.environ.get("SD_HTTP_FILE_ROOT", ""),
        help="Set this to the root of a filestructure to serve that via the HTTP server (in addition to the GRPC-WEB handler)",
    )
    util_opts.add_argument(
        "--enable_debug_recording",
        action="store_true",
        help="Enable collection of debug information for reporting with later. This collection is local only, until you deliberately choose to submit a sample.",
    )
    util_opts.add_argument(
        "--save_safetensors",
        type=str,
        default=None,
        help="'*' or a comma-seperated list of engine IDs to save as safetensors. All the models used in that engine will also be saved. Server will exit after completing.",
    )

    debug_opts.add_argument(
        "--vram_fraction",
        type=float,
        default=os.environ.get("SD_VRAM_FRACTION", 1.0),
        help="The fraction of memory that we should restrict ourselves to",
    )
    debug_opts.add_argument(
        "--monitor_ram",
        action=argparse.BooleanOptionalAction,
        help="Enable or disable monitoring of RAM and VRAM usage",
    )

    args = parser.parse_args()

    args.listen_to_all = args.listen_to_all or "SD_LISTEN_TO_ALL" in os.environ
    args.enable_mps = args.enable_mps or "SD_ENABLE_MPS" in os.environ
    args.reload = args.reload or "SD_RELOAD" in os.environ
    args.localtunnel = args.localtunnel or "SD_LOCALTUNNEL" in os.environ
    args.batch_autodetect = args.batch_autodetect or "SD_BATCH_AUTODETECT" in os.environ
    args.enable_debug_recording = (
        args.enable_debug_recording or "SD_ENABLE_DEBUG_RECORDING" in os.environ
    )
    args.supress_metadata = args.supress_metadata or "SD_SUPRESS_METADATA" in os.environ
    args.dont_refresh_on_error = (
        args.dont_refresh_on_error or "SD_DONT_REFRESH_ON_ERROR" in os.environ
    )

    refresh_models = None
    if args.refresh_models:
        refresh_models = re.split("\s*,\s*", args.refresh_models.strip())

    save_safetensor_patterns = None
    if args.save_safetensors:
        save_safetensor_patterns = re.split("\s*,\s*", args.save_safetensors.strip())
        args.reload = False

    if args.localtunnel and not args.access_token:
        args.access_token = secrets.token_urlsafe(16)

    if args.reload:
        # start_reloader will only return in a monitored subprocess
        reloader = hupper.start_reloader("sdgrpcserver.server.main", reload_interval=10)

    debug_recorder = DebugNullRecorder()

    if args.enable_debug_recording:
        debug_recorder = DebugRecorder()
        print(
            "You have enabled debug telemetry. "
            f"This will keep a local recording of all generation actions in the last 10 minutes in the folder '{debug_recorder.storage_path}'. "
            "See the README.md for how to submit a debug sample for troubleshooting."
        )

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction)

    ram_monitor = None
    if args.monitor_ram:
        ram_monitor = RamMonitor()
        ram_monitor.start()

    grpc = GrpcServer(args)
    grpc.start()

    http = HttpServer(args)
    http.start()

    localtunnel = None
    if args.localtunnel:
        localtunnel = LocaltunnelServer(args)
        localtunnel.start()

    prevHandler = None

    def shutdown_reactor_handler(*args):
        print("Waiting for server to shutdown...")
        if ram_monitor:
            ram_monitor.stop()
        if localtunnel:
            localtunnel.stop()
        http.stop()
        grpc.stop()
        print("All done. Goodbye.")
        sys.exit(0)

    prevHandler = signal.signal(signal.SIGINT, shutdown_reactor_handler)

    enginecfg_path = os.path.normpath(args.enginecfg)
    distcfg_path = os.path.join(os.path.dirname(__file__), "config")
    disthashes_path = os.path.join(distcfg_path, "dist_hashes")
    distenginecfg_path = os.path.join(distcfg_path, "engines.yaml")

    install_enginecfg = False

    if not os.path.exists(enginecfg_path):
        print("No existing engines.yaml - initialising with default version.")
        install_enginecfg = True
    else:
        # Read in existing config as byte-string, and calculate git hash
        current_cfg = open(enginecfg_path, "rb").read()
        current_hash = git_object_hash(current_cfg)
        # Read in a list of historic hashes, one per line
        *dist_hashes, newest_hash = open(disthashes_path, "r").read().splitlines()

        # If current_hash matches a distro hash, but not the _newest_ hash, update
        if current_hash in set(dist_hashes):
            print("Out of date default engines.yaml found - updating.")
            install_enginecfg = True
        elif current_hash == newest_hash:
            print("Current default engines.yaml found.")
        else:
            print("Your engines.yaml has been modified, so we won't change it.")

    if install_enginecfg:
        shutil.copyfile(distenginecfg_path, enginecfg_path)
        print(
            f"Edit {enginecfg_path} to enable, disable or add additional models and engines."
        )

    if xformers_mea_available():
        print("Xformers defaults to on")

    with open(os.path.normpath(args.enginecfg), "r") as cfg:
        engines = yaml.load(cfg, Loader=Loader)

        manager = EngineManager(
            engines,
            weight_root=args.weight_root,
            refresh_models=refresh_models,
            refresh_on_error=not args.dont_refresh_on_error,
            mode=EngineMode(
                vram_optimisation_level=args.vram_optimisation_level,
                enable_cuda=True,
                enable_mps=args.enable_mps,
            ),
            batchMode=BatchMode(
                autodetect=args.batch_autodetect,
                points=args.batch_points,
                simplemax=args.batch_max,
                safety_margin=args.batch_autodetect_margin,
            ),
            nsfw_behaviour=args.nsfw_behaviour,
            ram_monitor=ram_monitor,
        )

        print("Manager loaded")

        if save_safetensor_patterns:
            manager.save_engine_as_safetensor(save_safetensor_patterns)
            shutdown_reactor_handler()

        if ram_monitor:
            ram_monitor.print()

        # Create Generation Servicer and attach to all the servers

        generation_servicer = GenerationServiceServicer(
            manager,
            supress_metadata=args.supress_metadata,
            debug_recorder=debug_recorder,
            ram_monitor=ram_monitor,
        )

        generation_pb2_grpc.add_GenerationServiceServicer_to_server(
            generation_servicer, grpc.grpc_server
        )
        generation_pb2_grpc.add_GenerationServiceServicer_to_server(
            generation_servicer, http.grpc_server
        )
        http.grpc_gateway.add_GenerationServiceServicer(generation_servicer)

        # Create Engines Servicer and attach to all the servers

        engines_servicer = EnginesServiceServicer(manager)

        engines_pb2_grpc.add_EnginesServiceServicer_to_server(
            engines_servicer, grpc.grpc_server
        )
        engines_pb2_grpc.add_EnginesServiceServicer_to_server(
            engines_servicer, http.grpc_server
        )
        http.grpc_gateway.add_EnginesServiceServicer(engines_servicer)

        # Create Dashobard Servicer and attach to all the servers

        dashboard_servicer = DashboardServiceServicer()

        dashboard_pb2_grpc.add_DashboardServiceServicer_to_server(
            dashboard_servicer, grpc.grpc_server
        )
        dashboard_pb2_grpc.add_DashboardServiceServicer_to_server(
            DashboardServiceServicer(), http.grpc_server
        )

        print(
            f"GRPC listening on port {args.grpc_port}, HTTP listening on port {args.http_port}. Start your engines...."
        )

        loads_time = time.time()

        manager.loadPipelines()

        print(
            f"All engines ready, loading took {time.time()-loads_time:.1f}s, total startup {time.time()-start_time:.1f}s"
        )

        if ram_monitor:
            ram_monitor.print()

        # Block until termination
        grpc.block()
