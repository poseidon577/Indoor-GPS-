import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataclasses
from paper_utils import save_fig, save_data, get_standard_env

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(base_dir, "backend"))

from VLCL_AI.physics.physics_engine import PhysicsEngine
from VLCL_AI.communication.ber import BERCalculator
from VLCL_AI.communication.led_frequency_response import LEDFrequencyResponse

def main():
    print("Running Step 4: Communication Experiments")
    engine = PhysicsEngine()
    
    # 1. Channel Gain vs Distance
    distances = np.linspace(1, 5, 20)
    gains = []
    snrs_db = []
    snrs_linear = []
    for d in distances:
        env = get_standard_env()
        # Single LED at [2.5, 2.5, 3], receiver moving down in Z
        env = dataclasses.replace(env, 
            led_positions={1: [2.5, 2.5, 3]},
            led_powers={1: 10},
            receiver_position=[2.5, 2.5, 3 - d],
            distances={1: d}
        )
        state = engine.compute(env)
        gains.append(state.total_gains.get(1, 0))
        snr_db = state.snrs.get(1, 0)
        snrs_db.append(snr_db)
        snrs_linear.append(10**(snr_db/10.0))
        
    df_dist = pd.DataFrame({"Distance_m": distances, "Channel_Gain": gains, "SNR_dB": snrs_db})
    save_data(df_dist, "channel_gain_vs_distance")
    
    fig, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(distances, gains, 'b-o', label='Channel Gain')
    ax1.set_xlabel("Distance (m)")
    ax1.set_ylabel("Channel Gain", color='b')
    ax1.tick_params('y', colors='b')
    
    ax2 = ax1.twinx()
    ax2.plot(distances, snrs_db, 'r-s', label='SNR (dB)')
    ax2.set_ylabel("SNR (dB)", color='r')
    ax2.tick_params('y', colors='r')
    plt.title("Channel Gain and SNR vs Distance")
    save_fig(fig, "gain_snr_vs_distance")
    plt.close(fig)
    
    # 2. BER vs Distance and SNR
    df_ber = pd.DataFrame({"Distance_m": distances, "SNR_dB": snrs_db})
    for M in [2, 4, 16, 64]:
        df_ber[f"BER_{M}QAM"] = [BERCalculator.compute_analytical_qam(snr_lin, M) for snr_lin in snrs_linear]
        
    save_data(df_ber, "ber_vs_distance")
    fig, ax = plt.subplots(figsize=(6,4))
    for M in [2, 4, 16, 64]:
        ax.semilogy(distances, df_ber[f"BER_{M}QAM"], marker='o', label=f"{M}-QAM")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Bit Error Rate")
    ax.set_ylim(1e-6, 1)
    ax.legend()
    ax.grid(True, which="both", ls="--")
    plt.title("BER vs Distance")
    save_fig(fig, "ber_vs_distance")
    plt.close(fig)

    # 3. LED Frequency Response
    led = LEDFrequencyResponse(model_type='first_order', cutoff_frequency_hz=10e6)
    freqs = np.linspace(0, 50e6, 500)
    mag_db = [20 * np.log10(led.magnitude(f) + 1e-12) for f in freqs]
    df_led = pd.DataFrame({'Frequency_Hz': freqs, 'Magnitude_dB': mag_db})
    save_data(df_led, "led_frequency_response")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(freqs / 1e6, mag_db, 'b-')
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("LED Frequency Response ($f_c = 10$ MHz)")
    ax.grid(True, linestyle='--')
    save_fig(fig, "led_response")
    plt.close(fig)

    print("Step 4 complete.")

if __name__ == "__main__":
    main()
