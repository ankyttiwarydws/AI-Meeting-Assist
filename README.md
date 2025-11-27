
# AI Meeting Assistant (Windows)

This project is a local Windows prototype that:
- captures system audio (WASAPI loopback)
- transcribes with **faster-whisper** (GPU-enabled)
- sends text to a local LLM (Ollama or LM Studio) via HTTP
- displays a transparent always-on-top overlay (PySide6) with a scrollable conversation panel

**Important:** This Colab helper only generates the project files. Run this project locally on Windows (not in Colab).

## Quick start (on Windows)
1. Extract the zip into a folder.
2. Create a Python 3.11+ virtualenv and activate it.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
pip install faster-whisper
```

4. **Set up audio capture** (IMPORTANT):
   - Run the audio device tester: `python test_audio_devices.py`
   - This will list all available devices and test which ones work
   - If no loopback devices are found, enable "Stereo Mix":
     - Right-click speaker icon → Sounds
     - Recording tab → Right-click empty space → Show Disabled Devices
     - Enable "Stereo Mix" or "What U Hear"
   - Update `config.yaml` with the working device index: `audio_device_index: X`

5. Configure `config.yaml`:
   - Choose LLM backend (`ollama` or `lmstudio`)
   - Set `audio_device_index` from the test results
   - Adjust `gpu: false` if you have CUDA issues

6. Start your local LLM (Ollama or LM Studio) and ensure the HTTP API is running.

7. Run:

```powershell
python main.py
```

## Audio Capture Setup

**Windows System Audio Capture Options:**

1. **WASAPI Loopback** (Best option):
   - Automatically uses your default output device
   - Works with most modern Windows systems
   - No additional setup needed if using default speakers

2. **Stereo Mix** (Fallback):
   - Enable in Windows Sound settings → Recording tab
   - Right-click empty space → Show Disabled Devices
   - Enable "Stereo Mix" or "What U Hear"
   - Use the device index from `test_audio_devices.py`

3. **Troubleshooting**:
   - Run `python test_audio_devices.py` to discover and test devices
   - Check RMS levels - should be > 0.01 when audio is playing
   - If RMS is always 0.0000, the device isn't capturing audio
   - Try different device indices until you find one that works

## Notes
- This project assumes a Windows environment for audio loopback and overlay.
- Default silence trigger: **2 seconds** (you can change in `config.yaml`).
- Overlay shows a scrollable full conversation; answers "pour in" slowly.
- Audio capture requires either WASAPI loopback or Stereo Mix to be enabled.

## Choosing LLM backend
- **Ollama**: easy local setup, solid performance.
- **LM Studio**: also popular; may be simpler if you already use it.

The project supports both via `llm_client.py` and `config.yaml`.
