#!/usr/bin/env python3
"""
Debug script to test configuration loading.
"""

import sys
from pathlib import Path

# Add the project to the path
sys.path.insert(0, str(Path(__file__).parent))

from phonebooth_vision.config.manager import set_settings_path, get_settings
from phonebooth_vision.config.models import Settings

def test_config_loading():
    """Test configuration loading."""
    
    print("=== CONFIGURATION DEBUG ===")
    
    # Test 1: Check if config file exists
    config_path = Path("config.toml")
    print(f"1. Config file exists: {config_path.exists()}")
    print(f"   Config file path: {config_path.absolute()}")
    
    # Test 2: Try to load settings without explicit path
    print("\n2. Testing default settings (no config file):")
    try:
        settings = get_settings()
        print(f"   Clothing enabled: {settings.app.clothing.enabled}")
        print(f"   Model type: {settings.app.clothing.model_type}")
        print(f"   Model name: {settings.app.clothing.model_name}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Try to load settings with explicit path
    print("\n3. Testing settings with explicit config path:")
    try:
        set_settings_path(config_path)
        settings = get_settings()
        print(f"   Clothing enabled: {settings.app.clothing.enabled}")
        print(f"   Model type: {settings.app.clothing.model_type}")
        print(f"   Model name: {settings.app.clothing.model_name}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 4: Try to load settings directly
    print("\n4. Testing direct settings loading:")
    try:
        settings = Settings.load(config_path)
        print(f"   Clothing enabled: {settings.app.clothing.enabled}")
        print(f"   Model type: {settings.app.clothing.model_type}")
        print(f"   Model name: {settings.app.clothing.model_name}")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_config_loading()
