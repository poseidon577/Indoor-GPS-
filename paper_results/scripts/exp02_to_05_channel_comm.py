import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os, dataclasses
from paper_utils import get_standard_env, save_fig, save_data

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
from VLCL_AI.physics.physics_engine import PhysicsEngine
from VLCL_AI.communication.ber import BERCalculator

def main():
    print("Running Part 2-5: Channel and Comm Analysis")
    base_env = get_standard_env()
    physics_engine = PhysicsEngine()
    
    # --- Part 3: BER Curves ---
    snr_db_range = np.linspace(0, 40, 41)
    M_values = [2, 4, 16, 64, 256]
    
    ber_results = []
    for snr_db in snr_db_range:
        snr_linear = 10 ** (snr_db / 10)
        row = {'snr_db': snr_db}
        for M in M_values:
            ber = BERCalculator.compute_analytical_qam(snr_linear, M)
            row[f'M_{M}'] = float(ber)
        ber_results.append(row)
        
    df_ber = pd.DataFrame(ber_results)
    save_data(df_ber, "ber_vs_snr")
    
    fig, ax = plt.subplots(figsize=(5, 4))
    for M in M_values:
        ax.semilogy(df_ber['snr_db'], df_ber[f'M_{M}'], label=f'{M}-QAM')
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title("Theoretical BER vs SNR")
    ax.legend()
    ax.grid(True, which='both', linestyle='--')
    save_fig(fig, "ber_curves")
    plt.close(fig)
    
    # --- Part 5: Channel Analysis (LED Response) ---
    from VLCL_AI.communication.led_frequency_response import LEDFrequencyResponse
    led = LEDFrequencyResponse(model_type='first_order', cutoff_frequency_hz=10e6)
    
    freqs = np.linspace(0, 50e6, 500)
    mag_db = [20 * np.log10(led.magnitude(f) + 1e-12) for f in freqs]
    
    df_led = pd.DataFrame({'Frequency_Hz': freqs, 'Magnitude_dB': mag_db})
    save_data(df_led, "led_frequency_response")
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(freqs / 1e6, mag_db, 'b-')
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("LED Frequency Response ($f_c = 10$ MHz)")
    ax.grid(True, linestyle='--')
    save_fig(fig, "led_response")
    plt.close(fig)
    
    print("Completed Part 2-5.")

if __name__ == "__main__":
    main()
