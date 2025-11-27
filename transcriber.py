
"""transcriber.py
Wrapper around faster-whisper. This file expects the model to be installed on the local machine.
"""
import contextlib
import logging
import os
import sys

import numpy as np

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_stderr():
    """Temporarily suppress stderr output."""
    with open(os.devnull, 'w', encoding='utf-8') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


class Transcriber:
    def __init__(self, model_name='small', use_gpu=True, input_sample_rate=48000, cfg=None):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.input_sample_rate = input_sample_rate
        self.model = None
        self._warned = False
        # Load filter thresholds from config
        self.min_audio_level = (cfg or {}).get('min_audio_level', 0.005)
        self.min_language_probability = (cfg or {}).get('min_language_probability', 0.5)
        self.min_transcription_length = (cfg or {}).get('min_transcription_length', 3)
        try:
            from faster_whisper import WhisperModel

            device = 'cuda' if use_gpu else 'cpu'
            if use_gpu:
                # Try GPU first, but suppress stderr to avoid scary CUDA error messages
                try:
                    with suppress_stderr():
                        self.model = WhisperModel(model_name, device='cuda', compute_type='auto')
                    # Test if it actually works by doing a quick check
                    logger.info('Loaded faster-whisper model=%s device=cuda', model_name)
                except Exception as gpu_err:
                    logger.warning('GPU initialization failed, falling back to CPU: %s', gpu_err)
                    try:
                        self.model = WhisperModel(model_name, device='cpu', compute_type='auto')
                        self.use_gpu = False
                        logger.info('Loaded faster-whisper model=%s device=cpu (GPU unavailable)', model_name)
                    except Exception as cpu_err:
                        logger.error('CPU initialization also failed: %s', cpu_err)
                        self.model = None
            else:
                self.model = WhisperModel(model_name, device='cpu', compute_type='auto')
                logger.info('Loaded faster-whisper model=%s device=cpu', model_name)
        except ImportError as err:
            logger.error('faster-whisper is not installed. Install with "pip install faster-whisper". %s', err)
        except Exception as err:  # pylint: disable=broad-except
            logger.error('faster-whisper failed to initialize: %s', err)

    def _resample_to_16k(self, audio_np):
        target_rate = 16000
        if self.input_sample_rate == target_rate or self.input_sample_rate <= 0:
            return audio_np
        ratio = target_rate / float(self.input_sample_rate)
        new_length = int(len(audio_np) * ratio)
        if new_length <= 0:
            return audio_np
        x_old = np.linspace(0, 1, num=len(audio_np), endpoint=False)
        x_new = np.linspace(0, 1, num=new_length, endpoint=False)
        return np.interp(x_new, x_old, audio_np).astype(np.float32)

    def transcribe_audio_chunk(self, audio_np):
        # audio_np: 1-D numpy float32 PCM at samplerate self.input_sample_rate
        if self.model is None:
            if not self._warned:
                logger.error('Transcriber unavailable; skipping transcription.')
                self._warned = True
            return ''
        if audio_np is None or audio_np.size == 0:
            return ''
        
        # Check audio level - if too quiet, skip transcription
        audio_level = np.abs(audio_np).mean()
        if audio_level < self.min_audio_level:
            logger.debug('Audio too quiet (RMS=%.6f < %.4f), skipping transcription', audio_level, self.min_audio_level)
            return ''
        
        audio_16k = self._resample_to_16k(audio_np.astype(np.float32, copy=False))
        try:
            segments, info = self.model.transcribe(audio_16k, beam_size=5, temperature=0.2)
            
            # Check language detection confidence - if too low, it's probably noise
            if hasattr(info, 'language_probability') and info.language_probability < self.min_language_probability:
                logger.debug('Language detection confidence too low (%.2f < %.2f), likely noise - skipping', 
                          info.language_probability, self.min_language_probability)
                return ''
            
            texts = []
            segment_count = 0
            empty_segments = 0
            for segment in segments:
                segment_count += 1
                text = segment.text.strip()
                if text:
                    # Filter out very short transcriptions (likely false positives)
                    if len(text) < self.min_transcription_length:
                        logger.debug('Skipping very short transcription: "%s" (length=%d < %d)', 
                                  text, len(text), self.min_transcription_length)
                        empty_segments += 1
                        continue
                    # Filter out common false positive patterns
                    text_lower = text.lower()
                    if text_lower in ['thank you', 'thanks', 'uh', 'um', 'hmm', 'ah', 'oh', 'yeah', 'yes', 'no', 'ok', 'okay']:
                        # These are often false positives from noise - only keep if audio is loud enough
                        if audio_level < 0.01:
                            logger.debug('Skipping likely false positive: "%s" (audio too quiet)', text)
                            empty_segments += 1
                            continue
                    texts.append(text)
                    logger.debug('Whisper segment %d: "%s" (start=%.2f, end=%.2f)', segment_count, text, segment.start, segment.end)
                else:
                    empty_segments += 1
                    logger.debug('Whisper segment %d: empty text (start=%.2f, end=%.2f)', segment_count, segment.start, segment.end)
            result = ' '.join(texts).strip()
            if result:
                lang_prob = info.language_probability if hasattr(info, 'language_probability') else None
                logger.info('Transcription successful: "%s" (%d segments with text, %d empty, language=%s, lang_prob=%.2f, audio_level=%.4f)', 
                          result, segment_count - empty_segments, empty_segments, 
                          info.language if info else 'unknown', lang_prob or 0.0, audio_level)
            else:
                lang_prob = info.language_probability if hasattr(info, 'language_probability') else None
                logger.debug('Transcription returned empty (processed %d segments, %d empty, language=%s, lang_prob=%.2f, audio_level=%.4f)', 
                          segment_count, empty_segments, info.language if info else 'unknown', lang_prob or 0.0, audio_level)
            return result
        except (RuntimeError, OSError) as err:
            # CUDA/cuDNN errors - try to fallback to CPU if we're on GPU
            error_str = str(err).lower()
            if self.use_gpu and ('cuda' in error_str or 'cudnn' in error_str or 'gpu' in error_str):
                logger.warning('CUDA error during transcription, falling back to CPU: %s', err)
                try:
                    # Reinitialize on CPU
                    from faster_whisper import WhisperModel
                    self.model = WhisperModel(self.model_name, device='cpu', compute_type='auto')
                    self.use_gpu = False
                    logger.info('Reinitialized transcriber on CPU')
                    # Retry transcription
                    segments, info = self.model.transcribe(audio_16k, beam_size=5, temperature=0.2)
                    texts = []
                    for segment in segments:
                        text = segment.text.strip()
                        if text:
                            texts.append(text)
                    result = ' '.join(texts).strip()
                    logger.info('CPU fallback transcription: "%s"', result if result else '(empty)')
                    return result
                except Exception as cpu_err:
                    logger.error('CPU fallback also failed: %s', cpu_err)
                    return ''
            else:
                logger.error('Transcription error: %s', err)
                return ''
        except Exception as err:  # pylint: disable=broad-except
            logger.error('Transcription error: %s', err)
            return ''
