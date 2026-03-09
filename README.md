# USpaceNet: AI-Powered Autonomous Star Tracker

I built this project to bridge the gap between classical orbital mechanics and Deep Learning. It handles everything from propagating a satellite's orbit to using a **U-Net** to figure out exactly where the spacecraft is pointing in space.

## What it actually does
* **Orbital Sim:** Propagates a 2-body orbit (TBP) using `solve_ivp` and handles the LVLH frame transformations so the "camera" actually follows the orbit.
* **Synthetic Imaging:** Instead of just drawing simple dots, it projects star catalog vectors onto a 2D sensor plane. I added Gaussian noise and PSF (Point Spread Function) blurring to make the data look like it's coming from a real CMOS sensor.
* **The AI Bit:** I used a custom U-Net architecture (**USpaceNet**) to extract star centroids from the noisy images. It’s way more reliable than basic thresholding when the stars are faint or the noise is high.
* **Triangle Handshake:** It picks 3 detected stars, calculates their relative geometry, and matches them against a local catalog to identify which stars they actually are.
* **Attitude Determination:** Once identified, it solves the **Wahba problem** to calculate the final rotation matrix and outputs the pointing accuracy in degrees.

## Optimization (Killing the Lag)

**To fix the real-time performance, I:**
1. **Ditced Disk I/O:** Everything now stays in RAM. NumPy arrays are converted directly into PyTorch tensors.
2. **Hardware Acceleration:** Added a check to automatically use **NVIDIA CUDA** or **Apple MPS** (Metal) depending on the machine.

## How to Run
1. Make sure you have `torch`, `numpy`, `scipy`, and `ipywidgets` installed.
2. Open `star_tracker.ipynb`.
3. Use the **Orbit Step** slider to move the satellite along its path and the **FOV** slider to change the lens zoom.
4. The AI will update the centroid detections and attitude error instantly.


