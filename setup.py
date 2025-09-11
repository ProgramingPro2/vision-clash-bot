"""Setup script for vision-clash-bot."""

from setuptools import setup, find_packages

setup(
    name="pyclashbot",
    version="v0.0.0",
    description="Enhanced Clash Royale bot with AI and computer vision",
    packages=find_packages(include=["pyclashbot", "pyclashbot.*"]),
    python_requires=">=3.12.0,<3.13",
    install_requires=[
        "opencv-python>=4.9.0,<5",
        "numpy>=2.2.1,<3",
        "pymemuc>=0.6.0,<0.7",
        "psutil>=6.1.0,<7",
        "freesimplegui>=5.2.0,<6",
        "pygetwindow>=0.0.9,<0.0.10",
        "pillow>=11.3.0,<12.0.0",
        "torch>=2.0.0,<3.0.0",
        "scikit-learn>=1.3.0,<2.0.0",
        "scipy>=1.11.0,<2.0.0",
        "matplotlib>=3.7.0,<4.0.0",
        "pytesseract>=0.3.10,<1.0.0",
    ],
)
