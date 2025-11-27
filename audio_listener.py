
"""audio_listener.py
Captures system audio via sounddevice (WASAPI loopback on Windows).
"""
import logging
import threading
import time

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioListener:
    def __init__(self, chunk_seconds=3.0, samplerate=48000, channels=2, out_queue=None, device_index=None):
        self.chunk_seconds = chunk_seconds
        self.samplerate = samplerate
        self.channels = channels
        self.out_queue = out_queue
        self._running = True
        self._stream = None
        self._buffer = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self.device, self._using_wasapi = self._resolve_device(device_index)
        self._last_audio_time = None

    def _resolve_device(self, device_index):
        if device_index is not None:
            try:
                device_index = int(device_index)
            except ValueError:
                logger.warning('Invalid audio_device_index %s; falling back to auto-detect', device_index)
                device_index = None
        if device_index is not None:
            try:
                device_info = sd.query_devices(device_index)
                host = sd.query_hostapis()[device_info.get('hostapi', 0)]
                is_wasapi = host.get('type') == 'Windows WASAPI'
                logger.info(
                    'Using configured audio device index %s (%s, host=%s)',
                    device_index,
                    device_info.get('name'),
                    host.get('name'),
                )
                return device_index, is_wasapi
            except Exception as exc:
                logger.error('Failed to use configured audio_device_index %s: %s', device_index, exc)
        return self._find_loopback_device()

    def _find_loopback_device(self):
        """Attempt to find a WASAPI loopback device automatically.

        Returns a tuple of (device_index or None, bool using_wasapi).
        For WASAPI loopback, we need to use the OUTPUT device index.
        """
        try:
            host_apis = sd.query_hostapis()
            wasapi_ids = [idx for idx, api in enumerate(host_apis) if api.get('type') == 'Windows WASAPI']
        except Exception:
            wasapi_ids = []

        try:
            devices = sd.query_devices()
        except Exception as exc:
            logger.error('Audio device query failed: %s', exc)
            return None, False

        # First, try to find explicit loopback devices (Stereo Mix, etc.)
        for idx, device in enumerate(devices):
            name = device.get('name', '').lower()
            # Look for loopback devices (can be on any host API)
            if device.get('max_output_channels', 0) > 0 and (
                'loopback' in name or 'stereo mix' in name or 'mix (' in name or 'what u hear' in name
            ):
                host_api = host_apis[device.get('hostapi', 0)]
                is_wasapi = host_api.get('type') == 'Windows WASAPI'
                logger.info('Auto-selected loopback device %s (%s, host=%s)', idx, device.get('name'), host_api.get('name'))
                return idx, is_wasapi
        
        # Second, try WASAPI output devices (can use loopback mode)
        for idx, device in enumerate(devices):
            if device.get('hostapi') not in wasapi_ids:
                continue
            # WASAPI output devices can be opened as loopback inputs
            if device.get('max_output_channels', 0) > 0:
                logger.info('Auto-selected WASAPI output device %s (%s) for loopback', idx, device.get('name'))
                return idx, True
        
        # Third, fallback to default output device
        try:
            default_output = sd.default.device[1]
            if default_output is not None:
                device = devices[default_output]
                host_api = host_apis[device.get('hostapi', 0)]
                is_wasapi = host_api.get('type') == 'Windows WASAPI'
                logger.info('Using default output device %s (%s, host=%s)', 
                          default_output, device.get('name'), host_api.get('name'))
                return default_output, is_wasapi
        except Exception as exc:
            logger.warning('Falling back to default output device failed: %s', exc)
        
        logger.error('No suitable loopback device found. Enable Stereo Mix in Windows Sound settings.')
        return None, False

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning('Audio stream status: %s', status)
        if not self._running:
            return
        
        # Calculate RMS to detect if audio is actually coming through
        audio_level = np.abs(indata).mean()
        if audio_level > 0.001:  # Threshold for "real" audio
            self._last_audio_time = time.time()
            logger.debug('Audio detected: RMS=%.4f, frames=%s', audio_level, frames)
        
        mono = indata
        if mono.ndim == 2:
            mono = np.mean(mono, axis=1)
        mono = mono.astype(np.float32, copy=False)

        with self._lock:
            self._buffer = np.concatenate((self._buffer, mono))
            required = int(self.samplerate * self.chunk_seconds)
            while self._buffer.size >= required:
                chunk = self._buffer[:required]
                self._buffer = self._buffer[required:]
                if self.out_queue is not None:
                    try:
                        chunk_rms = np.abs(chunk).mean()
                        self.out_queue.put_nowait(chunk.copy())
                        logger.info('Enqueued audio chunk (%s samples, RMS=%.4f)', len(chunk), chunk_rms)
                    except Exception as e:
                        logger.warning('Failed to enqueue audio chunk: %s', e)

    def run(self):
        frames_per_buffer = max(1024, int(self.samplerate * self.chunk_seconds / 4))
        extra_settings = None
        if self._using_wasapi:
            try:
                extra_settings = sd.WasapiSettings(loopback=True)
            except TypeError:
                # Older sounddevice versions expose WasapiSettings but without keyword args
                try:
                    extra_settings = sd.WasapiSettings()
                    extra_settings.loopback = True
                except Exception:
                    extra_settings = None
            except AttributeError:
                # Very old sounddevice releases may not expose WasapiSettings at all
                extra_settings = None

        try:
            logger.info(
                'Opening audio stream (device=%s, samplerate=%s, chunk=%ss, using_wasapi=%s)',
                self.device,
                self.samplerate,
                self.chunk_seconds,
                self._using_wasapi,
            )
            self._stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                dtype='float32',
                callback=self._audio_callback,
                device=self.device,
                blocksize=frames_per_buffer,
                extra_settings=extra_settings,
            )
            self._stream.start()
            logger.info('Audio stream started')
            
            # Log a warning after 5 seconds if no audio detected
            startup_time = time.time()
            last_warning_time = startup_time
            while self._running:
                time.sleep(0.1)
                elapsed = time.time() - startup_time
                if elapsed > 5.0:
                    if self._last_audio_time is None or (time.time() - self._last_audio_time) > 5.0:
                        if (time.time() - last_warning_time) > 10.0:  # Warn every 10 seconds max
                            logger.warning(
                                'No audio detected. Ensure: '
                                '1) Stereo Mix is enabled in Windows Sound settings (Recording tab), '
                                '2) System audio is playing, '
                                '3) Device %s is the correct loopback device',
                                self.device,
                            )
                            last_warning_time = time.time()
        except Exception as e:
            logger.error('Audio capture error: %s', e)
            if not self._using_wasapi:
                logger.warning(
                    'Hint: Enable Windows WASAPI loopback by selecting a "Speakers (loopback)" device in Windows Sound settings.'
                )
            else:
                logger.warning(
                    'Hint: Ensure the loopback device is enabled in Windows Sound settings or try updating the sounddevice package.'
                )
        finally:
            if self._stream is not None:
                try:
                    self._stream.stop()
                except Exception:
                    pass
                try:
                    self._stream.close()
                except Exception:
                    pass
            logger.info('Audio listener stopped')

    def stop(self):
        self._running = False
        logger.debug('Stop requested for audio listener')
