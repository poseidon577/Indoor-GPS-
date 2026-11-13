# receiver.py
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from VLCL_AI.communication.adc import ADCModel
from VLCL_AI.communication.synchronization import Synchronizer
from VLCL_AI.communication.ofdm import OFDMDemodulator
from VLCL_AI.communication.channel_equalizer import ChannelEqualizer
from VLCL_AI.communication.qam import QAMModem
from VLCL_AI.communication.frame import CommunicationFrame

class VLCReceiver:
    """
    Manages the complete Visible Light Communication receiver chain.
    Transforms received analog/electrical signals back into estimated information bits.
    """
    
    def __init__(
        self,
        adc: ADCModel,
        synchronizer: Synchronizer,
        demodulator: OFDMDemodulator,
        equalizer: ChannelEqualizer,
        modem: QAMModem
    ):
        self.adc = adc
        self.synchronizer = synchronizer
        self.demodulator = demodulator
        self.equalizer = equalizer
        self.modem = modem

    def receive(
        self,
        rx_waveform: np.ndarray,
        tx_frame: CommunicationFrame,
        channel_response: np.ndarray,
        noise_variance: float = 1e-12
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Processes received electrical waveform to recover information bits.
        
        Args:
            rx_waveform (np.ndarray): Waveform received from the physical channel.
            tx_frame (CommunicationFrame): Original transmitted frame (for sync/demod parameters).
            channel_response (np.ndarray): Estimated frequency response of channel H_n.
            noise_variance (float): Noise variance for MMSE equalization.
            
        Returns:
            recovered_bits (np.ndarray): Decoded bit stream.
            equalized_symbols (np.ndarray): Complex symbol coordinates after equalization.
            metadata (dict): Receiver performance KPIs.
        """
        # 1. Analog to Digital Converter (ADC) quantization and clipping
        digital_waveform = self.adc.process(rx_waveform)
        
        # 2. Perfect Frame Synchronization
        sync_waveform = self.synchronizer.synchronize(digital_waveform, tx_frame.metadata)
        
        # 3. AC Coupling / DC Bias Removal
        # We remove the known DC bias applied by DCO-OFDM (stored in frame metadata)
        dc_bias = tx_frame.metadata.get("dc_bias", 0.0)
        ac_waveform = sync_waveform - dc_bias
        drive_gain = tx_frame.metadata.get("drive_gain", 1.0)
        
        # 4. Demodulate OFDM signal back to frequency domain complex symbols
        rx_symbols, freq_grid = self.demodulator.demodulate(ac_waveform)
        
        # 5. Extract Communication/Pilot symbols and apply Channel Equalization
        # Keep only communication/pilot length of rx_symbols matching tx_frame.qam_symbols
        rx_symbols_trimmed = rx_symbols[:len(tx_frame.qam_symbols)]
        
        # We need the channel response H_n mapped to the active carrier subcarriers.
        # Ensure channel_response is matched in shape
        if len(channel_response) != len(rx_symbols_trimmed):
            # If a flat or single vector is given, duplicate/interpolate as needed
            h_eq = np.resize(channel_response, len(rx_symbols_trimmed))
        else:
            h_eq = channel_response
        h_eq = h_eq * drive_gain
            
        equalized_symbols = self.equalizer.equalize(
            rx_symbols=rx_symbols_trimmed,
            h_channel=h_eq,
            noise_variance=noise_variance,
            subcarrier_powers=np.ones_like(h_eq)  # Default equal power for MMSE
        )
        
        # 6. QAM Constellation Slicing & Demodulation
        recovered_bits = self.modem.demodulate(equalized_symbols, tx_frame.modulation_order)
        
        # Crop or pad recovered bits to exactly match original payload length
        if len(recovered_bits) > len(tx_frame.payload_bits):
            recovered_bits = recovered_bits[:len(tx_frame.payload_bits)]
            
        metrics = {
            "num_recovered_symbols": len(equalized_symbols),
            "dc_offset_removed": dc_bias
        }
        
        return recovered_bits, equalized_symbols, metrics
