"""Test script to discover and test audio capture devices on Windows.
Run this to find the best device for system audio capture.
"""
import sounddevice as sd
import numpy as np
import time

def list_audio_devices():
    """List all available audio devices with details."""
    print("\n" + "="*80)
    print("AUDIO DEVICE DISCOVERY")
    print("="*80)
    
    devices = sd.query_devices()
    host_apis = sd.query_hostapis()
    
    print(f"\nFound {len(devices)} audio devices:\n")
    
    wasapi_devices = []
    loopback_candidates = []
    
    for idx, device in enumerate(devices):
        host_api = host_apis[device.get('hostapi', 0)]
        host_name = host_api.get('name', 'Unknown')
        host_type = host_api.get('type', 'Unknown')
        
        name = device.get('name', 'Unknown')
        max_input = device.get('max_input_channels', 0)
        max_output = device.get('max_output_channels', 0)
        default_samplerate = device.get('default_samplerate', 0)
        
        # Check if this is a WASAPI device
        is_wasapi = host_type == 'Windows WASAPI'
        
        # Check if it might be a loopback device
        is_loopback_candidate = (
            max_output > 0 and 
            ('loopback' in name.lower() or 
             'stereo mix' in name.lower() or 
             'mix' in name.lower() or
             'what u hear' in name.lower())
        )
        
        marker = ""
        if idx == sd.default.device[0]:
            marker += " [DEFAULT INPUT]"
        if idx == sd.default.device[1]:
            marker += " [DEFAULT OUTPUT]"
        if is_loopback_candidate:
            marker += " [LOOPBACK CANDIDATE]"
            loopback_candidates.append((idx, device, host_api))
        if is_wasapi:
            marker += " [WASAPI]"
            wasapi_devices.append((idx, device, host_api))
        
        print(f"Device {idx}: {name}")
        print(f"  Host: {host_name} ({host_type})")
        print(f"  Channels: {max_input} in, {max_output} out")
        print(f"  Sample rate: {default_samplerate} Hz{marker}\n")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if loopback_candidates:
        print("\n✓ Found loopback candidates:")
        for idx, device, host_api in loopback_candidates:
            print(f"  → Device {idx}: {device.get('name')} ({host_api.get('name')})")
            print(f"    Use this in config.yaml: audio_device_index: {idx}")
    else:
        print("\n⚠ No obvious loopback devices found.")
        print("  You may need to enable 'Stereo Mix' in Windows Sound settings:")
        print("  1. Right-click speaker icon → Sounds")
        print("  2. Recording tab → Right-click empty space → Show Disabled Devices")
        print("  3. Enable 'Stereo Mix'")
    
    if wasapi_devices:
        print(f"\n✓ Found {len(wasapi_devices)} WASAPI devices (best for loopback)")
        for idx, device, host_api in wasapi_devices:
            if device.get('max_output_channels', 0) > 0:
                print(f"  → Device {idx}: {device.get('name')} (output device)")
    
    return loopback_candidates, wasapi_devices


def test_device_capture(device_index, duration=3):
    """Test capturing audio from a specific device."""
    print(f"\n{'='*80}")
    print(f"TESTING DEVICE {device_index}")
    print(f"{'='*80}\n")
    
    try:
        device_info = sd.query_devices(device_index)
        print(f"Device: {device_info.get('name')}")
        print(f"Channels: {device_info.get('max_input_channels', 0)} in")
        print(f"Sample rate: {device_info.get('default_samplerate', 48000)} Hz")
        print(f"\nRecording for {duration} seconds...")
        print("Play some audio (YouTube, music, etc.) to test capture...\n")
        
        samplerate = int(device_info.get('default_samplerate', 48000))
        channels = min(device_info.get('max_input_channels', 2), 2)
        
        # Try to use WASAPI loopback if available
        extra_settings = None
        try:
            host_apis = sd.query_hostapis()
            device = sd.query_devices(device_index)
            host_api = host_apis[device.get('hostapi', 0)]
            if host_api.get('type') == 'Windows WASAPI':
                try:
                    extra_settings = sd.WasapiSettings(loopback=True)
                    print("Using WASAPI loopback mode")
                except (TypeError, AttributeError):
                    try:
                        extra_settings = sd.WasapiSettings()
                        extra_settings.loopback = True
                        print("Using WASAPI loopback mode (legacy)")
                    except Exception:
                        pass
        except Exception:
            pass
        
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
        
        # Convert to mono
        if recording.ndim == 2:
            mono = np.mean(recording, axis=1)
        else:
            mono = recording
        
        # Calculate statistics
        rms = np.abs(mono).mean()
        max_level = np.abs(mono).max()
        std_dev = np.std(mono)
        
        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        print(f"RMS (average level): {rms:.6f}")
        print(f"Peak level: {max_level:.6f}")
        print(f"Standard deviation: {std_dev:.6f}")
        
        if rms < 0.001:
            print("\n⚠ WARNING: Very low audio level detected!")
            print("  This device may not be capturing system audio.")
            print("  Try:")
            print("  1. Playing audio (YouTube, music, etc.)")
            print("  2. Checking Windows Sound settings")
            print("  3. Testing a different device")
        elif rms < 0.005:
            print("\n⚠ Low audio level - may work but might miss quiet speech")
        else:
            print("\n✓ Good audio level detected!")
            print("  This device should work for meeting capture.")
        
        return rms, max_level
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nThis device cannot be used for capture.")
        return None, None


if __name__ == '__main__':
    print("\n" + "="*80)
    print("WINDOWS SYSTEM AUDIO CAPTURE TESTER")
    print("="*80)
    
    # List all devices
    loopback_candidates, wasapi_devices = list_audio_devices()
    
    # Test loopback candidates
    if loopback_candidates:
        print("\n" + "="*80)
        print("TESTING LOOPBACK CANDIDATES")
        print("="*80)
        
        for idx, device, host_api in loopback_candidates:
            rms, peak = test_device_capture(idx, duration=3)
            if rms and rms > 0.005:
                print(f"\n✓ Device {idx} works well! Use: audio_device_index: {idx}")
                break
    else:
        print("\n" + "="*80)
        print("MANUAL DEVICE TESTING")
        print("="*80)
        print("\nNo automatic loopback candidates found.")
        print("Enter a device index to test (or press Enter to skip):")
        try:
            device_str = input("Device index: ").strip()
            if device_str:
                device_idx = int(device_str)
                test_device_capture(device_idx, duration=3)
        except (ValueError, KeyboardInterrupt):
            print("\nSkipping manual test.")

