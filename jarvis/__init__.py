"""Jarvis - a local, agentic desktop assistant.

perceive (screenshot + UI tree)  ->  think (local LLM brain)  ->  act (mouse/keys/os)
"""

__version__ = "0.1.0"

# Configure DPI awareness for GUI automation on Windows
import sys
if sys.platform == "win32":
    try:
        import ctypes
        # Set process DPI awareness to Per Monitor DPI Aware (value 2) or System DPI Aware (value 1)
        ctypes.windll.shcore.SetProcessDpiAwareness(2) # 2 = PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
