import sys
import os
import subprocess
import importlib.util
import pkg_resources

def check_file_exists(filename):
    if not os.path.exists(filename):
        print(f"[Error] Missing required file: {filename}")
        return False
    return True

def get_installed_packages():
    return {pkg.key for pkg in pkg_resources.working_set}

def check_dependencies(requirements_file='requirements.txt'):
    if not os.path.exists(requirements_file):
        print(f"[Warning] {requirements_file} not found. Skipping dependency check.")
        return []

    missing_packages = []
    installed_packages = get_installed_packages()
    
    with open(requirements_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse package name (handle version specifiers like package>=1.0)
            # Simple parsing: take everything before >=, ==, <, >, etc.
            pkg_name = line
            for op in ['>=', '<=', '==', '>', '<', '~=']:
                if op in line:
                    pkg_name = line.split(op)[0]
                    break
            
            pkg_name = pkg_name.strip().lower()
            if pkg_name not in installed_packages:
                # Some packages have different import names vs install names (e.g. opencv-python -> cv2)
                # This simple check might have false positives, but it's a good first line of defense.
                # We can try importlib to be sure for common ones if needed, but pkg_resources is usually reliable for installed names.
                missing_packages.append(line)
    
    return missing_packages

def install_packages(packages):
    print(f"Installing missing packages: {', '.join(packages)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        print("Installation complete.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Error] Failed to install packages: {e}")
        return False

def main():
    print("Initializing Multi-Camera Timelapse Analyzer...")
    
    # 1. Check Critical Files
    required_files = ['multicam_timelapse_analyzer.py', 'yolo_tracker_v2.py']
    for f in required_files:
        if not check_file_exists(f):
            input("Press Enter to exit...")
            sys.exit(1)
            
    # 2. Check Dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print("Missing dependencies found:")
        for dep in missing_deps:
            print(f" - {dep}")
        
        choice = input("Do you want to install them now? (y/n): ").lower()
        if choice == 'y':
            if not install_packages(missing_deps):
                print("Please install dependencies manually.")
                input("Press Enter to exit...")
                sys.exit(1)
        else:
            print("Cannot proceed without dependencies.")
            input("Press Enter to exit...")
            sys.exit(1)
    else:
        print("All dependencies checked. OK.")

    # 3. Launch GUI
    print("Starting GUI...")
    try:
        import multicam_timelapse_analyzer
        multicam_timelapse_analyzer.main()
    except Exception as e:
        print(f"[Critical Error] Failed to launch application: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()
