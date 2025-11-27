#!/usr/bin/env python3
"""Quick test for device 11 (WASAPI Stereo Mix)"""
import sounddevice as sd
import numpy as np
import time

def test_device_11():
    device_index = 11
    duration = 3
    
    print(f"Testing device {device_index} for {duration} seconds...")
    
    try:
        device_info = sd.query_devices(device_index)
        print(f"Device: {device_info.get('name')}")
        print(f"Host: {sd.query_hostapis()[device_info.get('hostapi', 0)].get('name')}")
        print(f"Channels: {device_info.get('max_input_channels', 0)} in")
        print(f"Sample rate: {device_info.get('default_samplerate', 48000)} Hz")
        
        samplerate = int(device_info.get('default_samplerate', 48000))
        channels = min(device_info.get('max_input_channels', 2), 2)
        
        # Use WASAPI settings for loopback
        extra_settings = None
        try:
            extra_settings = sd.WasapiSettings(loopback=True)
            print("Using WASAPI loopback mode")
        except (TypeError, AttributeError):
            try:
                extra_settings = sd.WasapiSettings()
                extra_settings.loopback = True
                print("Using WASAPI loopback mode (legacy)")
            except Exception:
                print("WASAPI settings not available")
        
        print("\nRecording... (play some audio to test)")
        frames = int(samplerate * duration)
        recording = sd.rec(
            frames,
            samplerate=samplerate,
            channels=channels,
            device=device_index,
            dtype='float32',
            extra_settings=extra_settings,
        )
        sd.wait()
        
        # Convert to mono and analyze
        if recording.ndim == 2:
            mono = np.mean(recording, axis=1)
        else:
            mono = recording
        
        rms = np.abs(mono).mean()
        max_level = np.abs(mono).max()
        
        print(f"\nResults:")
        print(f"RMS (average level): {rms:.6f}")
        print(f"Peak level: {max_level:.6f}")
        
        if rms < 0.001:
            print("⚠ WARNING: Very low audio level - device may not be working")
        elif rms < 0.005:
            print("⚠ Low audio level - may work but might miss quiet speech")
        else:
            print("✓ Good audio level detected!")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == '__main__':
    test_device_11()