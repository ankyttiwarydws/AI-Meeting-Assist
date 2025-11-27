
"""overlay.py
PySide6 overlay with a scrollable conversation panel.
Answers are appended slowly to emulate "pouring in".
"""
import logging
import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit
from PySide6.QtCore import Qt, QTimer

logger = logging.getLogger(__name__)

class OverlayApp:
    def __init__(self, display_queue):
        self.display_queue = display_queue
        self.app = QApplication(sys.argv)
        logger.info('Initialising overlay window')
        self.win = QWidget()
        self.win.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.win.setAttribute(Qt.WA_TranslucentBackground)
        self.win.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.win.setFixedSize(600, 300)
        self.win.move(50, 50)

        layout = QVBoxLayout()
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setStyleSheet('background: rgba(0,0,0,0.6); color: white; font-size: 14px;')
        self.text.setPlainText('Waiting for meeting audio...')
        layout.addWidget(self.text)
        self.win.setLayout(layout)

        # timer to poll the queue
        self.timer = QTimer()
        self.timer.timeout.connect(self._poll)
        self.timer.start(200)

        # slow appender state
        self._append_buffer = ''
        self._append_timer = QTimer()
        self._append_timer.timeout.connect(self._append_tick)
        self._append_timer.start(120)  # pour-in speed
        self._last_status = None
        logger.info('Overlay timers started')

    def _poll(self):
        try:
            while True:
                item = self.display_queue.get_nowait()
                role, text = item
                if role == 'assistant':
                    # append slowly
                    self._append_buffer = '\nAssistant: ' + text + '\n'
                    logger.debug('Queue -> assistant message (%s chars)', len(text))
                elif role == 'user':
                    self.text.append('\nUser: ' + text + '\n')
                    logger.debug('Queue -> user message (%s chars)', len(text))
                elif role == 'status':
                    if text != self._last_status:
                        self.text.append(f"\nStatus: {text}\n")
                        self._last_status = text
                        logger.info('Status update: %s', text)
        except Exception:
            pass

    def _append_tick(self):
        if not self._append_buffer:
            return
        # take a small slice
        take = int(max(1, len(self._append_buffer) * 0.08))
        piece = self._append_buffer[:take]
        self.text.insertPlainText(piece)
        self._append_buffer = self._append_buffer[take:]
        # scroll slowly
        self.text.verticalScrollBar().setValue(self.text.verticalScrollBar().maximum())

    def run(self):
        logger.info('Showing overlay window')
        self.win.show()
        sys.exit(self.app.exec())
