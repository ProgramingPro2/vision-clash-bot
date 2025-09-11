import sys
from pathlib import Path
import platform

from cx_Freeze import Executable, setup

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

PROJECT_NAME = "py-clash-bot"
AUTHOR = "Matthew Miglio, Martin Miglio"
DESCRIPTION = "Automated Clash Royale"
KEYWORDS = "clash of clans bot"
COPYRIGHT = "2023 Matthew Miglio"
ENTRY_POINT = ROOT_DIR / "pyclashbot" / "__main__.py"
ICON_PATH = ROOT_DIR / "assets" / "pixel-pycb.ico"
GUI = True
UPGRADE_CODE = "{494bebef-6fc5-42e5-98c8-d0b2e339750e}"


try:
    VERSION = sys.argv[sys.argv.index("--target-version") + 1]
except (ValueError, IndexError):
    VERSION = "v0.0.0"

version_file = ROOT_DIR / "pyclashbot" / "__version__"
if not version_file.exists():
    version_file.touch()
with version_file.open("w", encoding="utf-8") as f:
    f.write(VERSION)


build_exe_options = {
    "excludes": [
        "test", 
        "setuptools", 
        "tkinter", 
        "unittest", 
        "doctest",
        "distutils",
        "lib2to3"
    ],
    "packages": [
        "pyclashbot",
        "pyclashbot.ai",
        "pyclashbot.detection",
        "pyclashbot.config",
        "pyclashbot.bot",
        "pyclashbot.emulators",
        "pyclashbot.interface",
        "pyclashbot.utils",
        "torch",
        "sklearn",
        "scipy",
        "matplotlib",
        "pytesseract",
        "cv2",
        "numpy",
        "PIL",
        "psutil",
        "pymemuc",
        "freesimplegui",
        "pygetwindow",
    ],
    "include_files": [
        ROOT_DIR / "assets" / "pixel-pycb.ico",
        ROOT_DIR / "pyclashbot" / "detection" / "reference_images",
        ROOT_DIR / "pyclashbot" / "__version__",
        # Include model directories for movement bot (only if they exist)
        *[ROOT_DIR / dir_name for dir_name in ["models", "data", "config"] if (ROOT_DIR / dir_name).exists()],
        # Include Windows-specific files (only if they exist)
        *[("pyclashbot/emulators/configs/memu_config.json", "pyclashbot/emulators/configs/memu_config.json") 
          if (ROOT_DIR / "pyclashbot" / "emulators" / "configs" / "memu_config.json").exists() else ()],
    ],
    "include_msvcr": True,
    "zip_include_packages": "*",
    "zip_exclude_packages": "",
    "optimize": 0,  # Don't optimize to avoid import issues
    "silent": False,  # Show build progress
    "replace_paths": [
        ("*", ""),  # Remove path prefixes
    ],
    "build_exe": "build/exe.win-amd64-3.12",  # Explicit build directory
    "constants": [
        "BUILD_EXE=True",  # Add build constant
    ],
}

bdist_msi_options = {
    "upgrade_code": UPGRADE_CODE,
    "add_to_path": False,
    "initial_target_dir": f"[ProgramFilesFolder]\\{PROJECT_NAME}",
    "summary_data": {
        "author": AUTHOR,
        "comments": DESCRIPTION,
        "keywords": KEYWORDS,
    },
}

# Force Windows compilation for cross-platform building
# This will work with Wine or when building on Windows
base = "Win32GUI" if GUI else None
uac_admin = True
shortcut_name = f"{PROJECT_NAME} {VERSION}"
shortcut_dir = "DesktopFolder"
target_name = f"{PROJECT_NAME}.exe"

exe = Executable(
    script=ENTRY_POINT,
    base=base,
    uac_admin=uac_admin,
    shortcut_name=shortcut_name,
    shortcut_dir=shortcut_dir,
    target_name=target_name,
    copyright=COPYRIGHT,
    icon=ICON_PATH,
)

setup(
    name=PROJECT_NAME,
    description=DESCRIPTION,
    executables=[exe],
    options={
        "bdist_msi": bdist_msi_options,
        "build_exe": build_exe_options,
    },
)
