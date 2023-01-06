import dashboard_pb2
import dashboard_pb2_grpc

from gyre.services.exception_to_grpc import exception_to_grpc


class DashboardServiceServicer(dashboard_pb2_grpc.DashboardServiceServicer):
    def __init__(self):
        pass

    @exception_to_grpc
    def GetMe(self, request, context):
        user = dashboard_pb2.User()
        user.id = "0000-0000-0000-0001"
        return user
