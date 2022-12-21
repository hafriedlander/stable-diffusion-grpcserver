import inspect
import os
import re
import traceback

import grpc

return_traceback = os.environ.get("SD_ENV", "dev").lower().startswith("dev")


def _handle_exception(func, e, context, mappings):
    stack = [f"Exception in handler {func.__name__}. "]

    for block in traceback.format_exception(e):
        stack.append(block)

    stack = "".join(stack)
    print(stack, end=None)

    code, message = grpc.StatusCode.INTERNAL, "Internal Error"
    for exception_class, grpc_code in mappings.items():
        if isinstance(e, exception_class):
            code = grpc_code
            message = str(e)
            break

    context.abort(
        code,
        stack if return_traceback else message,
    )


def _exception_to_grpc_generator(func, mappings):
    def wrapper(*args, **kwargs):
        if "context" in kwargs:
            context = kwargs["context"]
        else:
            context = args[-1]

        try:
            yield from func(*args, **kwargs)
        except grpc.RpcError as e:
            # Allow grpc / whatever-called-Servicer to receive RpcError
            raise e
        except BaseException as e:
            _handle_exception(func, e, context, mappings)

    return wrapper


def _exception_to_grpc_unary(func, mappings):
    def wrapper(*args, **kwargs):
        if "context" in kwargs:
            context = kwargs["context"]
        else:
            context = args[-1]

        try:
            return func(*args, **kwargs)
        except grpc.RpcError as e:
            # Allow grpc / whatever-called-Servicer to receive RpcError
            raise e
        except BaseException as e:
            _handle_exception(func, e, context, mappings)

    return wrapper


def exception_to_grpc(mapping):
    def decorator(func):
        if inspect.isgeneratorfunction(func):
            return _exception_to_grpc_generator(func, mapping)
        else:
            return _exception_to_grpc_unary(func, mapping)

    if callable(mapping):
        func, mapping = mapping, {}
        return decorator(func)
    else:
        return decorator
