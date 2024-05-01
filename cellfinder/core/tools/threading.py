from queue import Queue
from threading import Thread


class EOFSignal:
    pass


class ThreadWithException(Thread):

    def __init__(self, target=None, args=(), kwargs=None):
        super().__init__(
            target=self.user_func_runner, args=args, kwargs=kwargs
        )
        self.to_thread_queue = Queue(maxsize=0)
        self.from_thread_queue = Queue(maxsize=0)
        self.user_func = target

    def user_func_runner(self, *args, **kwargs):
        try:
            if self.user_func is not None:
                self.user_func(*args, **kwargs)
        except BaseException as e:
            self.from_thread_queue.put(
                ("exception", e), block=True, timeout=None
            )
        finally:
            self.from_thread_queue.put(("eof", None), block=True, timeout=None)

    def send_msg_to_mainthread(self, value):
        self.from_thread_queue.put(("user", value), block=True, timeout=None)

    def get_msg_from_thread(self):
        msg, value = self.from_thread_queue.get(block=True, timeout=None)
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
