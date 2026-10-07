"""Hourly base profiles shared by the three cases (index 0 = 00:00-01:00)."""
import numpy as np


def get_load_profile():
    """Normalized load multiplier (fraction of peak load) for each hour of the day."""
    profile = np.array([
        0.45, 0.40, 0.38, 0.36, 0.36, 0.38,   # 0-5: late night / early morning
        0.50, 0.70, 0.85, 0.90, 0.90, 0.88,   # 6-11: morning ramp
        0.87, 0.87, 0.85, 0.85, 0.88, 0.95,   # 12-17: afternoon
        1.00, 0.97, 0.90, 0.80, 0.65, 0.52,   # 18-23: evening peak then decline
    ])
    return profile  # shape (24,)


def get_pv_profile():
    """Normalized PV availability: sin bell between 06:00 and 19:00, zero at night."""
    profile = np.zeros(24)
    for h in range(6, 20):
        angle = np.pi * (h - 6.0) / (19.0 - 6.0)
        profile[h] = np.sin(angle)  # 0 at 6h, peak at ~12.5h, 0 at 19h
    return profile  # shape (24,)


def get_carbon_intensity_profile():
    """Carbon intensity of grid imports (kg CO2 per kWh) for each hour of the day."""
    profile = np.array([
        0.90, 0.90, 0.85, 0.85, 0.85, 0.90,  # 0-5:  coal baseload
        0.80, 0.60, 0.35, 0.15, 0.08, 0.05,  # 6-11: solar kicks in
        0.05, 0.05, 0.08, 0.20, 0.45, 0.65,  # 12-17: solar fading
        0.80, 0.90, 0.92, 0.88, 0.85, 0.90   # 18-23: evening coal+gas peak
    ])
    return profile
