from queue import Queue
from threading import Thread

import torch.multiprocessing as mp


class EOFSignal:
    pass


class ExceptionWithQueueMixIn:

    to_thread_queue: Queue

    from_thread_queue: Queue

    args: tuple = ()

    def __init__(self, target, pass_self=False):
        self.user_func = target
        self.pass_self = pass_self

    def user_func_runner(self):
        try:
            if self.user_func is not None:
                if self.pass_self:
                    self.user_func(self, *self.args)
                else:
                    self.user_func(*self.args)
        except BaseException as e:
            self.from_thread_queue.put(
                ("exception", e), block=True, timeout=None
            )
        finally:
            self.from_thread_queue.put(("eof", None), block=True, timeout=None)

    def send_msg_to_mainthread(self, value):
        self.from_thread_queue.put(("user", value), block=True, timeout=None)

    def get_msg_from_thread(self, timeout=None):
        msg, value = self.from_thread_queue.get(block=True, timeout=timeout)
        if msg == "eof":
            return EOFSignal
        if msg == "exception":
            raise Exception("Processing signal data failed") from value

        return value

    def send_msg_to_thread(self, value):
        self.to_thread_queue.put(("user", value), block=True, timeout=None)

    def notify_to_end_thread(self):
        self.to_thread_queue.put(("eof", None), block=True, timeout=None)

    def get_msg_from_mainthread(self):
        msg, value = self.to_thread_queue.get(block=True, timeout=None)
        if msg == "eof":
            return EOFSignal

        return value


class ThreadWithException(ExceptionWithQueueMixIn):

    def __init__(self, target, args=(), **kwargs):
        super().__init__(target=target, **kwargs)
        self.to_thread_queue = Queue(maxsize=0)
        self.from_thread_queue = Queue(maxsize=0)
        self.args = args
        self.thread = Thread(target=self.user_func_runner)

    def start(self):
        self.thread.start()

    def join(self, timeout=None):
        self.thread.join(timeout=timeout)


class ProcessWithException(ExceptionWithQueueMixIn):

    process: mp.Process = None

    def __init__(self, target, args=(), **kwargs):
        super().__init__(target=target, **kwargs)
        ctx = mp.get_context("spawn")
        self.to_thread_queue = ctx.Queue(maxsize=0)
        self.from_thread_queue = ctx.Queue(maxsize=0)
        self.process = ctx.Process(target=self.user_func_runner)

        self.args = args

    def start(self):
        self.process.start()

    def join(self, timeout=None):
        self.process.join(timeout=timeout)
