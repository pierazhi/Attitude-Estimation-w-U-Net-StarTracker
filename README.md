It handles everything from propagating a satellite's orbit to using a **U-Net** to figure out exactly where the spacecraft is pointing in space.

## What it actually does
* **Orbital Sim:** Propagates a 2-body orbit (TBP) using `solve_ivp` and handles the LVLH frame transformations so the "camera" actually follows the orbit.
* **Synthetic Imaging:** Instead of just drawing simple dots, it projects star catalog vectors onto a 2D sensor plane. I added Gaussian noise and PSF blurring to make the data look like it's coming from a real CMOS sensor.
* **Centroid Detection:** I used a custom U-Net architecture (**USpaceNet**) to extract star centroids from the noisy images. It’s way more reliable than basic thresholding when the stars are faint or the noise is high.
* **Triangle Linking** It picks 3 detected stars, calculates their relative geometry, and matches them against a local catalog to identify which stars they actually are.
* **Attitude Determination:** Once identified, it solves the **Wahba problem** to calculate the final rotation matrix and outputs the pointing accuracy in degrees.

## How to Run
1. Make sure you have `torch`, `numpy`, `scipy`, and `ipywidgets` installed.
2. Open `star_tracker.ipynb`. Make sure that the .pth is in the same folder of the ipynb.
4. Use the **Orbit Step** slider to move the satellite along its path and the **FOV** slider to change the lens zoom.
5. The AI will update the centroid detections and attitude error instantly.

<img width="2130" height="700" alt="1" src="https://github.com/user-attachments/assets/1eee0a59-beed-45ee-af6d-f33f938cde5d" />
<img width="800" height="800" alt="2" src="https://github.com/user-attachments/assets/6d4e1fe8-ea4e-4c06-9edd-f5967c60857a" />
<img width="1200" height="500" alt="3" src="https://github.com/user-attachments/assets/1728985b-f68b-41eb-b98e-0e6963b4b2c5" />
<img width="2130" height="600" alt="4" src="https://github.com/user-attachments/assets/06adb430-d7c3-4186-be1f-83d7f27e45c2" />

## Future Ideas

1. Use an already built CNN
2. Train a better CNN
3. Implement attitude disturbances to S/C orientation

