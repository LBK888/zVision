"""PyInstaller entry point; freeze support must precede argument parsing."""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from uvc_lab.__main__ import main
    main()
