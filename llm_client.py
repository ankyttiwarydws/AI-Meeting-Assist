
"""llm_client.py
Supports Ollama and LM Studio via simple HTTP APIs. Choose backend in config.yaml.
"""
import logging

import requests

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = cfg.get('llm_backend', 'lmstudio')
        self.ollama_url = cfg.get('ollama_url', 'http://localhost:11434')
        self.lmstudio_url = cfg.get('lmstudio_url', 'http://localhost:1234')
        self.model = cfg.get('llm_model', '')
        self.max_tokens = cfg.get('llm_max_tokens', 200)
        self.temperature = cfg.get('llm_temperature', 0.2)
        self.timeout = cfg.get('llm_timeout', 15)
        logger.info(
            'LLMClient configured (backend=%s, model=%s, timeout=%ss)',
            self.backend,
            self.model or '<default>',
            self.timeout,
        )

    def query(self, prompt):
        logger.debug('Dispatching query to backend=%s (prompt chars=%s)', self.backend, len(prompt or ''))
        if self.backend == 'ollama':
            return self._query_ollama(prompt)
        return self._query_lmstudio(prompt)

    def _query_ollama(self, prompt):
        try:
            url = self.ollama_url.rstrip('/') + '/api/generate'
            payload = {
                'model': self.model or 'llama2',
                'prompt': prompt,
                'max_tokens': self.max_tokens
            }
            # Ollama may expect different payload shape; adjust per your install.
            logger.debug('POST %s payload=%s', url, payload)
            r = requests.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            # try typical fields
            if isinstance(data, dict) and 'text' in data:
                return data['text']
            return str(data)
        except Exception as e:
            logger.error('Ollama request error: %s', e)
            return '[LLM error]'

    def _query_lmstudio(self, prompt):
        try:
            url = self.lmstudio_url.rstrip('/') + '/v1/chat/completions'
            payload = {
                'model': self.model,
                'messages': [
                    {'role': 'system', 'content': self.cfg.get('system_prompt', 'You are a helpful AI meeting assistant.')},
                    {'role': 'user', 'content': prompt},
                ],
                'max_tokens': self.max_tokens,
                'temperature': self.temperature,
                'stream': False,
            }
            # Remove empty model to let LM Studio fallback to default
            if not payload['model']:
                payload.pop('model')

            logger.debug('POST %s payload_keys=%s', url, list(payload.keys()))
            r = requests.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            choices = data.get('choices') or []
            if choices:
                message = choices[0].get('message') or {}
                content = message.get('content')
                if content:
                    return content.strip()
            # fallback to completion/text fields
            return data.get('text') or data.get('completion') or str(data)
        except Exception as e:
            logger.error('LM Studio request error: %s', e)
            return '[LLM error]'
