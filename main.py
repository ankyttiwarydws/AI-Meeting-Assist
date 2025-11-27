
"""Entry point for the local AI meeting assistant.
Run on Windows. This orchestrates audio capture -> transcription -> LLM -> overlay.
"""
import logging
import threading
import queue
import time

import numpy as np
import yaml

from transcriber import Transcriber
from llm_client import LLMClient
from overlay import OverlayApp
from audio_listener import AudioListener

# load config
with open('config.yaml','r') as f:
    cfg = yaml.safe_load(f)

log_level_name = str(cfg.get('log_level', 'INFO')).upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(
    level=log_level,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)
logger.info('AI Meeting Assistant starting (log level=%s)', log_level_name)

AUDIO_SAMPLE_RATE = cfg.get('audio_samplerate', 48000)
AUDIO_CHUNK_SECONDS = cfg.get('audio_chunk_seconds', 3.0)


def stt_worker(audio_q, text_q, stop_event, silence_seconds):
    stt_logger = logging.getLogger('stt_worker')
    trans = Transcriber(
        model_name=cfg.get('whisper_model', 'small'),
        use_gpu=cfg.get('gpu', True),
        input_sample_rate=AUDIO_SAMPLE_RATE,
        cfg=cfg,
    )
    stt_logger.info(
        'Transcriber initialised (model=%s, gpu=%s, sample_rate=%s)',
        cfg.get('whisper_model', 'small'),
        trans.use_gpu,
        AUDIO_SAMPLE_RATE,
    )
    buffer = ''
    last_text_time = None
    chunks_received = 0
    chunks_with_audio = 0
    while not stop_event.is_set():
        try:
            audio = audio_q.get(timeout=0.25)
            chunks_received += 1
            # Check if audio has actual signal
            audio_level = np.abs(audio).mean() if audio.size > 0 else 0.0
            if audio_level > 0.001:
                chunks_with_audio += 1
            if chunks_received % 10 == 0:  # Log every 10 chunks
                stt_logger.info(
                    'Audio chunks: received=%s, with_audio=%s, buffer_len=%s',
                    chunks_received,
                    chunks_with_audio,
                    len(buffer),
                )
        except queue.Empty:
            if buffer and last_text_time and (time.time() - last_text_time) >= silence_seconds:
                final_text = buffer.strip()
                if final_text:
                    stt_logger.info('Silence detected -> finalising utterance (%s chars)', len(final_text))
                    text_q.put(('final', final_text))
                buffer = ''
                last_text_time = None
            continue
        # transcribe chunk
        stt_logger.debug('Transcribing audio chunk (%s samples, RMS=%.4f)', len(audio), audio_level)
        try:
            txt = trans.transcribe_audio_chunk(audio)
        except Exception as trans_err:
            stt_logger.error('Transcription exception (non-fatal): %s', trans_err, exc_info=True)
            continue
        if not txt:
            if chunks_received % 5 == 0:  # Log every 5 empty transcriptions to avoid spam
                stt_logger.info('No transcription text returned (RMS=%.4f, chunk=%d)', audio_level, chunks_received)
            continue
        stt_logger.info('Transcription result: "%s"', txt)
        buffer = (buffer + ' ' if buffer else '') + txt.strip()
        last_text_time = time.time()
        stt_logger.debug('Updated partial transcript (%s chars)', len(buffer))
        text_q.put(('partial', buffer))

    # flush remaining buffered text when stopping
    if buffer:
        final_text = buffer.strip()
        if final_text:
            stt_logger.info('Flushing buffered utterance on shutdown (%s chars)', len(final_text))
            text_q.put(('final', final_text))


def llm_worker(text_q, display_q, stop_event):
    llm_logger = logging.getLogger('llm_worker')
    client = LLMClient(cfg)
    llm_logger.info('LLM worker online (backend=%s)', client.backend)
    while not stop_event.is_set():
        try:
            typ, payload = text_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if typ == 'partial':
            continue
        elif typ == 'final':
            user_text = payload.strip()
            if not user_text:
                continue
            llm_logger.info('Submitting %s chars of user text to LLM', len(user_text))
            display_q.put(('user', user_text))
            display_q.put(('status', 'Querying assistant...'))
            response = client.query(user_text)
            llm_logger.info('Received response (%s chars)', len(response or ''))
            display_q.put(('assistant', response))
            display_q.put(('status', 'Listening...'))
        elif typ == 'shutdown':
            llm_logger.info('Shutdown signal received')
            break


if __name__ == '__main__':
    audio_q = queue.Queue()
    text_q = queue.Queue()
    display_q = queue.Queue()
    stop_event = threading.Event()

    # start audio listener (uses system loopback on Windows)
    listener = AudioListener(
        chunk_seconds=AUDIO_CHUNK_SECONDS,
        samplerate=AUDIO_SAMPLE_RATE,
        out_queue=audio_q,
        device_index=cfg.get('audio_device_index'),
    )
    logger.info(
        'Audio listener initialised (device=%s, chunk=%ss, samplerate=%s)',
        listener.device,
        AUDIO_CHUNK_SECONDS,
        AUDIO_SAMPLE_RATE,
    )
    t_listener = threading.Thread(target=listener.run, daemon=True)
    t_listener.start()
    logger.info('Audio listener thread started')

    t_stt = threading.Thread(
        target=stt_worker,
        args=(audio_q, text_q, stop_event, cfg.get('silence_seconds', 2.0)),
        daemon=True,
    )
    t_llm = threading.Thread(target=llm_worker, args=(text_q, display_q, stop_event), daemon=True)
    t_stt.start(); t_llm.start()
    logger.info('STT and LLM worker threads started')

    # launch overlay (runs on main thread)
    app = OverlayApp(display_q)
    display_q.put(('status', 'Waiting for meeting audio...'))
    try:
        logger.info('Launching overlay UI')
        app.run()
    finally:
        stop_event.set()
        listener.stop()
        text_q.put(('shutdown', None))
        logger.info('Shutdown complete')
