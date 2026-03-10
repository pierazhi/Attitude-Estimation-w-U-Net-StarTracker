import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as pexp
from ipywidgets import interact, FloatSlider, FloatLogSlider, Dropdown
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os
from scipy import ndimage

def tbp(t, y, mu, case):
    x = y[0:3]
    v = y[3:6]

    if case == 'no':
        drdt = v
        dvdt = - mu * x / np.linalg.norm(x)**3

    elif case == 'j2':
        J2 = 0.1082626925638815e-2
        we = np.deg2rad(15.04/3600)
        w_e = np.array([0, 0, we])
        R_e = 6378.137  # km

        xx = np.linalg.norm(x)
        c = 1.5 * J2 * mu * R_e**2 / xx**4

        a_j2_x = c * x[0] / xx * (5*x[2]**2 / xx**2 - 1)
        a_j2_y = c * x[1] / xx * (5*x[2]**2 / xx**2 - 1)
        a_j2_z = c * x[2] / xx * (5*x[2]**2 / xx**2 - 3)

        drdt = v

        dvdt_x = - mu * x[0] / np.linalg.norm(x)**3 + a_j2_x
        dvdt_y = - mu * x[1] / np.linalg.norm(x)**3 + a_j2_y
        dvdt_z = - mu * x[2] / np.linalg.norm(x)**3 + a_j2_z

        dvdt = np.array([dvdt_x, dvdt_y, dvdt_z])

    return np.concatenate([drdt, dvdt])


def car2kep(R, V, mu):
    """
    Computes the Keplerian orbital elements from the state vector (R, V).
    
    Inputs:
    R  : Position vector [km]
    V  : Velocity vector [km/s]
    mu : Gravitational parameter [km^3/s^2]
    
    Outputs:
    a, e, i, OM, om, theta
    """
    eps = 1e-10
    
    # 1. Magnitudes and Radial Velocity
    r = np.linalg.norm(R)
    v = np.linalg.norm(V)
    vr = np.dot(R, V) / r
    
    # 2. Angular Momentum
    H = np.cross(R, V)
    h = np.linalg.norm(H)
    
    # 3. Inclination (rad)
    # H[2] is the Z-component (MATLAB H(3))
    i = np.acos(H[2] / h)
    
    # 4. Node Vector
    N = np.cross([0, 0, 1], H)
    n = np.linalg.norm(N)
    
    # 5. Right Ascension of Ascending Node (OM)
    if n != 0:
        OM = np.acos(N[0] / n) # N[0] is X-component
        if N[1] < 0:           # N[1] is Y-component
            OM = 2 * np.pi - OM
    else:
        OM = 0
        
    # 6. Eccentricity Vector and Magnitude
    E = 1 / mu * ((v**2 - mu / r) * R - r * vr * V)
    e = np.linalg.norm(E)
    
    # 7. Argument of Perigee (om)
    if n != 0:
        if e > eps:
            om = np.acos(np.dot(N, E) / (n * e))
            if E[2] < 0: # E[2] is Z-component
                om = 2 * np.pi - om
        else:
            om = 0
    else:
        om = 0
        
    dum = np.dot(E, R) / (e * r)

    # Safety Check: ensure dum is within [-1, 1] to avoid math errors with acos
    dum = np.clip(dum, -1.0, 1.0)

    if e > eps:
        theta = np.acos(dum)
        
        # Quadrant check using radial velocity (vr)
        if vr < 0:
            theta = 2 * np.pi - theta
            
    else:
        # Circular orbit case: use the node vector N as a reference
        cp = np.cross(N, R)
        
        # (Note: dum is recalculated here in your MATLAB script, 
        # but it's the same math as above)
        if cp[2] >= 0: # cp[2] is the Z-component (MATLAB's cp(3))
            theta = np.acos(dum)
        else:
            theta = 2 * np.pi - np.acos(dum)

    # 9. Semi-major Axis
    a = h**2 / mu / (1 - e**2)

    kep_state = np.array([a, e, i, OM, om, theta])
    
    return kep_state

def kep2car(kep_state, mu):
    """
    Converts Keplerian orbital elements to Cartesian state vectors.
    
    Inputs:
    kep_state : Array of Keplerian elements [a, e, i, OM, om, theta]
    mu        : Gravitational parameter [km^3/s^2]
    
    Outputs:
    R : Position vector [km]
    V : Velocity vector [km/s]
    """
    a, e, i, OM, om, theta = kep_state

    # 1. Semi-latus rectum
    p = a * (1 - e**2)

    # 2. Position in perifocal coordinates
    r_perifocal = (p / (1 + e * np.cos(theta))) * np.array([np.cos(theta), np.sin(theta), 0])

    # 3. Velocity in perifocal coordinates
    v_perifocal = np.sqrt(mu / p) * np.array([-np.sin(theta), e + np.cos(theta), 0])

    # 4. Rotation matrices
    R3_OM = np.array([[np.cos(OM), -np.sin(OM), 0],
                      [np.sin(OM),  np.cos(OM), 0],
                      [0,           0,          1]])

    R1_i = np.array([[1, 0,           0],
                     [0, np.cos(i), -np.sin(i)],
                     [0, np.sin(i),  np.cos(i)]])

    R3_om = np.array([[np.cos(om), -np.sin(om), 0],
                      [np.sin(om),  np.cos(om), 0],
                      [0,           0,          1]])

    # Combined rotation matrix
    Q_pX = R3_OM @ R1_i @ R3_om

    # 5. Position and velocity in inertial frame
    R = Q_pX @ r_perifocal
    V = Q_pX @ v_perifocal

    return R, V

def add_earth_surface(fig, Re, opacity=0.18, n=50):
    phi, theta = np.mgrid[0:2*np.pi:complex(n), 0:np.pi:complex(n)]
    x = Re * np.cos(phi) * np.sin(theta)
    y = Re * np.sin(phi) * np.sin(theta)
    z = Re * np.cos(theta)
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale='Greens',
        opacity=opacity,
        showscale=False,
        hoverinfo='skip',
        name='Earth'
    ))

def project_combined(star_direction_unit, A0, fov_deg, image_size, sizes, YY, current_idx, where):
    # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -
    # Creates a camera model which points towards the same line of the Nadir but opposite direction (+ Z)
    # Masks all the stars outside the FOV of the camera, highlights only the one inside it
    # Projects the 3D stars onto 2D camera plane
    # Plots the 3D stars and highlights the ones inside the FOV of the camera, plots the 2D camera view of the same space
    # Outputs the stars inside the FOV, with their respective position in the camera frame
    # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -

    boresight_B = np.array([0, 0, 1])
    boresight_I = A0.T @ boresight_B
    fov_rad = np.deg2rad(fov_deg)
    f = 1 / (np.tan(fov_rad / 2))
    
    visibility = star_direction_unit @ boresight_I
    mask = visibility > np.cos(fov_rad / 2)
    mask_bool = mask.astype(bool)

    number_of_stars = star_direction_unit[mask_bool, 0].shape[0]
    
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        specs=[[{'type': 'scene'}, {'type': 'xy'}]],
        subplot_titles=(f"3D ECI Space | True Stars in View: {number_of_stars}", "2D Sensor View")
    )
    # Project to 2D
    stars_B = A0 @ star_direction_unit[mask_bool, :].T
    x_coords = f * stars_B[0] / stars_B[2]
    y_coords = f * stars_B[1] / stars_B[2]      
    px = ((- x_coords + 1) / 2) * image_size
    py = ((- y_coords + 1) / 2) * image_size

    # --- 3D TRACES (Column 1) ---

    # Trajectory

    global_scale = np.mean(np.linalg.norm(YY, axis=1)) / 0.6
    scaled_YY = YY / global_scale

    fig.add_trace(go.Scatter3d(
        x=scaled_YY[:, 0],
        y=scaled_YY[:, 1],
        z=scaled_YY[:, 2],
        mode='lines',
        name="Trajectory",
        line=dict(color='yellow', width=3) # Corretto qui
    ))

    # Earth

    R_Earth = 6378.15e3 / global_scale
    add_earth_surface(fig, R_Earth, opacity=0.5, n=50)

    r_current = scaled_YY[current_idx, :] 

    # 2. Define the boresight length (how far the red line extends in the 3D plot)
    # Since stars are at radius 1.0, 0.4 is a good length to show direction
    b_len = 0.4 

    # --- Boresight Line starting from Satellite position ---
    fig.add_trace(go.Scatter3d(
        x=[r_current[0], r_current[0] + boresight_I[0] * b_len], 
        y=[r_current[1], r_current[1] + boresight_I[1] * b_len], 
        z=[r_current[2], r_current[2] + boresight_I[2] * b_len],
        mode='lines', 
        line=dict(color='purple', width=5), 
        name='Boresight'
    ), row=1, col=1)

    # --- Cone (Arrowhead) at the tip of the boresight line ---
    fig.add_trace(go.Cone(
        x=[r_current[0] + boresight_I[0] * b_len], 
        y=[r_current[1] + boresight_I[1] * b_len], 
        z=[r_current[2] + boresight_I[2] * b_len],
        u=[boresight_I[0]], 
        v=[boresight_I[1]], 
        w=[boresight_I[2]],
        sizemode="absolute", 
        sizeref=0.1, 
        showscale=False,
        colorscale=[[0, 'purple'], [1, 'red']], 
        anchor="tail",
        showlegend=False, 
    ), row=1, col=1)

    # Add Body Axes

    body_fram_len = 0.2
    colors = ['red', 'green', 'blue']
    body_axes = ['X_B', 'Y_B', 'Z_B']
    for i in range(3):
        fig.add_trace(go.Scatter3d(
        x=[r_current[0], r_current[0] + A0[i, 0] * body_fram_len], 
        y=[r_current[1], r_current[1] + A0[i, 1] * body_fram_len], 
        z=[r_current[2], r_current[2] + A0[i, 2] * body_fram_len],
        mode='lines', 
        line=dict(color=colors[i], width=5), 
        name=body_axes[i],
        ), row=1, col=1)

        # --- Cone (Arrowhead) at the tip of the boresight line ---
        fig.add_trace(go.Cone(
            x=[r_current[0] + A0[i, 0] * body_fram_len], 
            y=[r_current[1] + A0[i, 1] * body_fram_len], 
            z=[r_current[2] + A0[i, 2] * body_fram_len],
            u=[A0[i, 0]], 
            v=[A0[i, 1]], 
            w=[A0[i, 2]],
            sizemode="absolute", 
            sizeref=0.1, 
            showscale=False,
            colorscale=[[0, colors[i]], [1, colors[i]]], 
            anchor="tail",
            showlegend=False, 
        ), row=1, col=1)

    # Optional: Add a marker for the Satellite itself so it's visible at the start of the line
    fig.add_trace(go.Scatter3d(
        x=[r_current[0]], y=[r_current[1]], z=[r_current[2]],
        mode='markers',
        marker=dict(size=4, color='yellow'),
        name='Satellite'
    ), row=1, col=1)

    # Stars Outside FOV

    fig.add_trace(go.Scatter3d(
        x=star_direction_unit[~mask_bool, 0], 
        y=star_direction_unit[~mask_bool, 1], 
        z=star_direction_unit[~mask_bool, 2],
        mode='markers', marker=dict(color='white', size=sizes[~mask_bool], opacity=0.3),
        name='Fuori FOV'
    ), row=1, col=1)

    # Stars Inside FOV
    fig.add_trace(go.Scatter3d(
        x=star_direction_unit[mask_bool, 0], 
        y=star_direction_unit[mask_bool, 1], 
        z=star_direction_unit[mask_bool, 2],
        mode='markers', marker=dict(color='red', size=sizes[mask_bool]),
        name='Nel FOV (3D)'
    ), row=1, col=1)

    # --- 2D TRACES (Column 2) ---
    # First, add the black background "image"
    img = np.zeros((image_size, image_size))
    # We use Heatmap instead of imshow here because imshow creates its own figure
    fig.add_trace(go.Heatmap(
        z=img, 
        colorscale=[[0, 'black'], [1, 'black']], # Forza tutto a nero indipendentemente dai dati
        zmin=0, 
        zmax=1, 
        showscale=False, 
        hoverinfo='skip'
    ), row=1, col=2)

    # Add the 2D Star Markers
    fig.add_trace(go.Scatter(
        x=px, y=py, mode='markers',
        marker=dict(color='yellow', size=sizes[mask_bool], symbol='circle'),
        name='Nel FOV (2D)'
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=[np.round(image_size / 2)], y=[np.round(image_size / 2)], mode='markers',
        marker=dict(color='red', size=3, symbol='cross'),
        name='Boresight Center'
    ), row=1, col=2)
    
    # --- LAYOUT CONFIGURATION ---
    fig.update_layout(
        template="plotly_dark",
        height=700,
        showlegend=True,
        paper_bgcolor="black",
        plot_bgcolor="black"
    )

    # Fix 2D Axis (Col 2)
    fig.update_xaxes(range=[0, image_size], visible=False, row=1, col=2)
    fig.update_yaxes(range=[image_size, 0], visible=False, row=1, col=2) # Inverted Y

    # Fix 3D Scene (Col 1)
    fig.update_scenes(
        xaxis=dict(backgroundcolor="black"),
        yaxis=dict(backgroundcolor="black"),
        zaxis=dict(backgroundcolor="black"),
        row=1, col=1
    )
    if where == 'browser':
        pio.renderers.default = 'browser'
    elif where == 'vscode':
        pio.renderers.default = 'vscode'
    fig.show()

    # return fig.data
    return mask_bool, px, py

def create_catalogo(num_stars, scale, smallest, biggest):
    # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -
    # Creates a catalogue of N stars, with random size between the smallest value and the biggest one
    # Outputs the K divisions, the sorted pairs and distances and the star unit vectors as well as their sizes
    # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -

    star_direction = np.random.normal(0, 1, (num_stars, 3))
    star_direction_unit = star_direction / np.linalg.norm(star_direction, axis = 1, keepdims=True)
    sizes = np.random.exponential(scale, size=num_stars) + smallest
    sizes = np.clip(sizes, smallest, biggest)
    
    distances = []
    pairs = []

    for i, j in itertools.combinations(range(num_stars), 2):
        d = np.dot(star_direction_unit[i], star_direction_unit[j])
        distances.append(d)
        pairs.append((i, j))

    # 2. Convert to NumPy arrays for speed
    distances = np.array(distances)
    pairs = np.array(pairs)

    # 3. SORT THEM by distance
    sort_idx = np.argsort(distances)
    distances_sorted = distances[sort_idx]
    pairs_sorted = pairs[sort_idx]

    N_pairs = len(pairs_sorted)
    slope = (N_pairs - 1) / (distances_sorted[-1] - distances_sorted[0])
    q = -slope * distances_sorted[0]

    i = np.arange(N_pairs)
    thresholds = (i - q) / slope           
    K = np.searchsorted(distances_sorted, thresholds, side="left").astype(float)

    return K, pairs_sorted, distances_sorted, slope, q, N_pairs, star_direction_unit, sizes

def lvlh_frame(r0, v0):
    """Updated to Zenith-pointing frame"""
    # Z is Zenith (pointing away from Earth)
    Z_b = r0 / np.linalg.norm(r0)              
    # Y is the Orbit Normal
    Y_b = np.cross(r0, v0)
    Y_b = Y_b / np.linalg.norm(Y_b)            
    # X is Along-track (X = Y cross Z)
    X_b = np.cross(Y_b, Z_b)
    X_b = X_b / np.linalg.norm(X_b)             
    return np.vstack([X_b, Y_b, Z_b])

def create_real_image(sizes, px, py, noise_std, image_size):
    # Genera l'immagine direttamente come array numpy senza salvarla
    canvas = np.zeros((image_size, image_size))
    x_grid = np.arange(0, image_size)
    y_grid = np.arange(0, image_size)
    xx, yy = np.meshgrid(x_grid, y_grid)

    for k in range(px.shape[0]):
        # Calcolo della PSF (Point Spread Function) stellare
        exponent = -((xx - px[k])**2 + (yy - py[k])**2) / (2 * sizes[k]**2)
        canvas += np.exp(exponent) 

    # Aggiunta rumore gaussiano
    noise = np.random.normal(0, noise_std, (image_size, image_size))
    canvas = np.clip(canvas + noise, 0, 1)
    
    return canvas

def extract_all_stars(heatmap, threshold=0.2):
    # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -
    # Extracts all the centroids from the stars inside the FOV (that the model has identified, this may be different to the real s actually present)
    # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -

    # 1. Convert to a binary mask (1 for star, 0 for space)
    mask = (heatmap > threshold).numpy().astype(int)
    
    # 2. Label each 'island' of pixels with a unique number
    # labels will be a grid where Star 1 is all 1s, Star 2 is all 2s, etc.
    labels, n_stars = ndimage.label(mask)
    
    star_coords = []
    
    # 3. Find the center of each island
    # center_of_mass calculates the weighted center for each label ID
    centers = ndimage.center_of_mass(heatmap.numpy(), labels, range(1, n_stars + 1))
    
    for y, x in centers:
        star_coords.append((x, y))
        
    return np.array(star_coords)


def get_candidates(K, d_measured, slope, q, N_pairs, tolerance=1e-4):
    z_min = int(np.floor(slope * (d_measured - tolerance) + q))
    z_max = int(np.ceil(slope * (d_measured + tolerance) + q))

    z_min = max(0, min(N_pairs - 1, z_min))
    z_max = max(0, min(N_pairs - 1, z_max))
    start_idx = int(K[z_min])
    end_idx = int(K[z_max])

    return start_idx, end_idx

def solve_wahba(obs_vectors, ref_vectors, weights=None):
    # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -
    # Solves Wahba's problem using Singular Value Decomposition.
    
    # Parameters:
    # obs_vectors: (N, 3) array of unit vectors in Body Frame (AI detections)
    #  ref_vectors: (N, 3) array of unit vectors in ECI Frame (Catalog)
    # weights: (N,) array of weights for each star (optional)
    # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # - # -
    
    if weights is None:
        weights = np.ones(len(obs_vectors))
        
    # 1. Compute the Correlation Matrix H
    # H = sum(w_i * (obs_i * ref_i^T))
    B = np.zeros((3, 3))
    for i in range(len(obs_vectors)):
        B += weights[i] * np.outer(obs_vectors[i], ref_vectors[i])

    # 2. Perform Singular Value Decomposition
    U, S, Vt = np.linalg.svd(B)

    # 3. Handle the reflection case to ensure a proper rotation matrix (det = +1)
    # The determinant of the rotation matrix must be positive.
    d = np.linalg.det(U) * np.linalg.det(Vt)
    M = np.diag([1, 1, d])
    
    # 4. Calculate the optimal Rotation Matrix
    A_estimated = U @ M @ Vt
    
    return A_estimated

class USpaceNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = nn.Conv2d(1, 16, 3, padding = 1)
        self.enc2 = nn.Conv2d(16, 32, 3, padding = 1)

        # Middle
        self.bottle = nn.Conv2d(32, 64, 3, padding = 1)

        # Decoder
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride = 2)
        self.dec1 = nn.Conv2d(64, 32, 3, padding = 1)

        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride = 2)
        self.dec2 = nn.Conv2d(32, 16, 3, padding = 1)

        self.final = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        # --- Encoder ---
        skip1 = F.relu(self.enc1(x))      # 256x256
        e1 = F.max_pool2d(skip1, 2)       # 128x128

        skip2 = F.relu(self.enc2(e1))     # 128x128
        e2 = F.max_pool2d(skip2, 2)       # 64x64

        # --- Middle ---
        middle = F.relu(self.bottle(e2))  # 64x64

        # --- Decoder Block 1 ---
        d1 = self.up1(middle)             # Upsample to 128x128
        d1 = torch.cat([d1, skip2], dim=1)# Match 128 with 128
        d1 = F.relu(self.dec1(d1))

        # --- Decoder Block 2 ---
        d2 = self.up2(d1)                 # Upsample to 256x256
        d2 = torch.cat([d2, skip1], dim=1)# FIX: Match 256 with 256
        d2 = F.relu(self.dec2(d2))

        # Final pixel-wise probability map
        return torch.sigmoid(self.final(d2))


def plot_attitude(A_true, A_est):
    """
    Visualizes the True and Estimated Body Axes in two 3D subplots.
    """
    # Initialize subplots: 1 row, 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("True Attitude", "Estimated Attitude")
    )

    colors = ['red', 'green', 'blue'] # X, Y, Z
    true_names = ['X_True', 'Y_True', 'Z_True']
    est_names = ['X_Est', 'Y_Est', 'Z_Est']

    for i in range(3):
        # --- SUBPLOT 1: TRUE ATTITUDE ---
        # Line
        fig.add_trace(go.Scatter3d(
            x=[0, A_true[i, 0]], y=[0, A_true[i, 1]], z=[0, A_true[i, 2]],
            mode='lines', line=dict(color=colors[i], width=6),
            name=true_names[i], legendgroup='true'
        ), row=1, col=1)
        # Cone
        fig.add_trace(go.Cone(
            x=[A_true[i, 0]], y=[A_true[i, 1]], z=[A_true[i, 2]],
            u=[A_true[i, 0]], v=[A_true[i, 1]], w=[A_true[i, 2]],
            sizemode="absolute", sizeref=0.1, showscale=False,
            colorscale=[[0, colors[i]], [1, colors[i]]], anchor="tail"
        ), row=1, col=1)

        # --- SUBPLOT 2: ESTIMATED ATTITUDE ---
        # Line
        fig.add_trace(go.Scatter3d(
            x=[0, A_est[i, 0]], y=[0, A_est[i, 1]], z=[0, A_est[i, 2]],
            mode='lines', line=dict(color=colors[i], width=6, dash='dash'),
            name=est_names[i], legendgroup='est'
        ), row=1, col=2)
        # Cone
        fig.add_trace(go.Cone(
            x=[A_est[i, 0]], y=[A_est[i, 1]], z=[A_est[i, 2]],
            u=[A_est[i, 0]], v=[A_est[i, 1]], w=[A_est[i, 2]],
            sizemode="absolute", sizeref=0.1, showscale=False,
            colorscale=[[0, colors[i]], [1, colors[i]]], anchor="tail"
        ), row=1, col=2)

    # Common axis settings
    scene_settings = dict(
        xaxis=dict(range=[-1.2, 1.2], title="X"),
        yaxis=dict(range=[-1.2, 1.2], title="Y"),
        zaxis=dict(range=[-1.2, 1.2], title="Z"),
        aspectmode='cube'
    )

    fig.update_layout(
        title_text="Attitude Determination: True vs Estimated",
        height=600,
        scene=scene_settings,   # Applies to first subplot
        scene2=scene_settings,  # Applies to second subplot
        showlegend=True
    )

    fig.show()