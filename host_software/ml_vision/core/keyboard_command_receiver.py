"""Keyboard-driven stand-in for AudioCommandReceiverONNX
(host_software/src/audio_receiver_onnx.py), for testing state-machine/target-switching
logic in a live entry point without a working microphone or the trained audio model.

Same public interface (get_latest_command(), stop(), latest_inference_time_ms), so it's
a drop-in swap at the call site -- nothing downstream (TargetStateMachine.process_command(),
the status print line) needs to know which one is active.

Accepted commands match what TargetStateMachine._apply_command() understands
(host_software/src/state_machine.py): hold, stop, go_<color> (e.g. go_blue, go_black),
forward, backward, left, right. Type a command and press Enter.
"""

import queue
import threading


class KeyboardCommandReceiver:
    def __init__(self) -> None:
        self.command_queue: queue.Queue = queue.Queue(maxsize=1)
        self.running = True
        # Kept only for call-site compatibility with AudioCommandReceiverONNX's status print.
        self.latest_inference_time_ms = 0.0

        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        print(
            "Keyboard command receiver active -- type a command and press Enter "
            "(e.g. go_blue, go_black, hold, stop, forward, backward, left, right)."
        )

    def _read_loop(self) -> None:
        # input() blocks until Enter/EOF -- fine on a daemon thread, it just never gets
        # explicitly interrupted by stop() (the thread dies with the process instead).
        while self.running:
            try:
                line = input()
            except EOFError:
                break

            command = line.strip()
            if not command:
                continue

            if self.command_queue.full():
                try:
                    self.command_queue.get_nowait()
                except queue.Empty:
                    pass
            self.command_queue.put(command)

    def get_latest_command(self):
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        self.running = False
