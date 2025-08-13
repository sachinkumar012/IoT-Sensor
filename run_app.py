#!/usr/bin/env python3
"""
IoT Sensor Data RAG System - Application Launcher

This script launches the Streamlit application for the IoT Sensor Data RAG system.
It handles setup, configuration, and launching the web interface.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    # Map package names to their import names
    package_imports = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'sentence-transformers': 'sentence_transformers',
        'plotly': 'plotly'
    }
    
    missing_packages = []
    
    for package, import_name in package_imports.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed")
    return True

def setup_environment():
    """Set up the environment for the application"""
    print("🔧 Setting up environment...")
    
    # Create necessary directories
    directories = [
        "data/sensors",
        "data/manuals",
        "data/specs",
        "data/chroma_db",
        "models",
        "logs",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created directory: {directory}")
    
    # Check if configuration exists
    config_file = "config/config.yaml"
    if not os.path.exists(config_file):
        print(f"   Configuration file not found: {config_file}")
        print("   Using default configuration")
    else:
        print(f"   Configuration file found: {config_file}")
    
    print("✅ Environment setup complete")

def launch_streamlit(port=8501, host="localhost"):
    """Launch the Streamlit application"""
    print(f"🚀 Launching Streamlit application on {host}:{port}")
    print("📱 The application will open in your default web browser")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 50)
    
    # Get the path to the main application
    app_path = Path("app/main.py")
    
    if not app_path.exists():
        print(f"❌ Application file not found: {app_path}")
        print("Please ensure the application files are in the correct location")
        return False
    
    try:
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.port", str(port),
            "--server.address", host,
            "--server.headless", "false"
        ]
        
        subprocess.run(cmd, check=True)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  Application stopped by user")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def run_tests():
    """Run the test suite"""
    print("🧪 Running system tests...")
    
    test_file = Path("tests/test_system.py")
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        cmd = [sys.executable, str(test_file)]
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="IoT Sensor Data RAG System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_app.py                    # Launch with default settings
  python run_app.py --port 8502       # Launch on port 8502
  python run_app.py --host 0.0.0.0    # Launch on all interfaces
  python run_app.py --test            # Run tests only
  python run_app.py --setup           # Setup environment only
        """
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8501,
        help="Port to run the application on (default: 8501)"
    )
    
    parser.add_argument(
        "--host", 
        type=str, 
        default="localhost",
        help="Host to bind the application to (default: localhost)"
    )
    
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Run tests only"
    )
    
    parser.add_argument(
        "--setup", 
        action="store_true",
        help="Setup environment only"
    )
    
    parser.add_argument(
        "--check-deps", 
        action="store_true",
        help="Check dependencies only"
    )
    
    args = parser.parse_args()
    
    print("🏗️  IoT Sensor Data RAG for Smart Buildings")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup environment
    if args.setup or not args.test:
        setup_environment()
    
    # Run tests if requested
    if args.test:
        if run_tests():
            print("✅ All tests passed!")
            sys.exit(0)
        else:
            print("❌ Some tests failed!")
            sys.exit(1)
    
    # Check dependencies only
    if args.check_deps:
        print("✅ Dependency check complete")
        sys.exit(0)
    
    # Launch application
    if launch_streamlit(args.port, args.host):
        print("✅ Application completed successfully")
        sys.exit(0)
    else:
        print("❌ Application failed to launch")
        sys.exit(1)

if __name__ == "__main__":
    main()
