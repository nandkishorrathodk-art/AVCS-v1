"""Standalone Distribution Setup for the 'avc' (Artificial Visual Cortex) package."""
from setuptools import find_packages, setup

setup(
    name="avc",
    version="1.0.0",
    description="Artificial Visual Cortex (AVC) — 1.14M Biomimetic Neural Network for Real-Time Screen Perception",
    long_description="""# AVC — Artificial Visual Cortex (Neural Network)

A biomimetic visual cortex for screens and computer-use AI agents.

## Architecture
- **Retinal Foveation**: Saccadic eye movement with high-detail foveal (256x256) and peripheral (256x256) streams.
- **Cortical Backbone**: Dual-path Inverted Residual (MBConv) + Squeeze-and-Excitation stages fusing into a 128-channel Conscious Visual Field.
- **Dorsal Stream ('Where / How')**: Dense 5-class per-pixel affordance heatmap (static, clickable, typeable, scrollable, draggable).
- **Ventral Stream ('What')**: 8-way UI application category classification and text density regression.
- **Language Modulation**: Zero-generation spatial text grounding vectors.
- **Predictive Coding Engine**: 527K-parameter recurrent GRU temporal world model computing optical surprise and stability states.
- **Biomechanical Servoing**: Closed-loop Fitts' Law velocity field steering.

## Quickstart
```python
import avc

# 1. Load pretrained cortex (bundled weights loaded automatically)
cortex = avc.load_cortex()

# 2. Perceive any screen frame
state = cortex.perceive(screen_frame)
print("UI Category:", state.primary_category)
print("Stability:", state.stability_state)
print("Affordance Map:", state.affordance_map.shape)
print("Hotspots:", state.action_hotspots)
```
""",
    long_description_content_type="text/markdown",
    author="Parth Labs",
    author_email="contact@parthlabs.in",
    url="https://github.com/Parth-labs-in/AVCS",
    packages=find_packages(include=["avc*"]),
    package_data={"avc": ["py.typed", "weights/*.pt"]},
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
        "Pillow>=9.0.0",
    ],
    extras_require={
        "desktop": ["mss>=9.0.0"],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Operating System :: OS Independent",
    ],
)
