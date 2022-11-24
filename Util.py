import functools
import sys
import time
import types
import importlib
from pathlib import Path


class TimeLogger(types.ModuleType):
    """
    Timeprint can be used as a context manager or decorator
    to print how much time things take to run.

    It pretends to be a module so it can be used directly,
    rather than making users import or reference something
    below the timeprint module.

    reference : https://github.com/raymondbutcher/python-timeprint

    """

    # Keep a reference to these because messing with sys.modules
    # causes them to disappear from the module/global namespace.
    functools = functools
    sys = sys
    time = time
    __name__ = __name__

    def __init__(self, name: str, logger=None):
        self._name = name
        self.logger = logger
        self._stack = []
        super(self.__class__, self).__init__(self.__name__)

    def __enter__(self):
        self._stack.append(
            (
                self._name,
                self.time.time(),
            )
        )

    def __exit__(self, *args, **kwargs):
        name, start = self._stack.pop()
        elapsed = self.time.time() - start
        log_msg = self.format_message(name=name, seconds=elapsed)
        if self.logger:
            self.logger.info(log_msg)
        else:
            print(log_msg)

    def __call__(self, *args):
        func = args and args[0]
        if hasattr(func, "__call__") and self.logger is None:
            return self.decorator(func)
        else:
            return self.contextmanager(*args)

    def format_message(self, name, seconds):
        if seconds < 0.001:
            return f"{name} {seconds * 1000:.4f} ms"
        elif seconds < 0.1:
            return f"{name} {seconds * 1000:.2f} ms"
        elif seconds < 1:
            return f"{name} {seconds * 1000:.0f} ms"
        else:
            return f"{name} {seconds:.1f} s"

    def contextmanager(self, name=None):
        return self.__class__(name or self._name)

    def decorator(self, func):

        context = self.contextmanager(self._name or func.__name__)

        @self.functools.wraps(func)
        def decorator(*args, **kwargs):
            with context:
                return func(*args, **kwargs)

        return decorator

class common:
    @staticmethod
    def get_module_from_file(module_name, file_path):
        spec = importlib.util.spec_from_file_location(
            module_name, str(Path(file_path) / f"{module_name}.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def get_module_class(module_name, class_name, path):
        m = common.get_module_from_file(module_name, path)
        c = getattr(m, class_name)
        return c

    @staticmethod
    def to_camel_case(snake_str):
        components = snake_str.split("_")
        return "".join(x.title() for x in components[0:])

    @staticmethod
    def usage():
        print("usage: command [OPTIONS] [MODULE]")
        print("    -m, --module   module name")
        print("    -d, --datapath   train data directory")
        print("    -g, --gpu   gpu number")
