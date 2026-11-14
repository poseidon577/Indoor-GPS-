# spectrum_partitioner.py
import numpy as np
from typing import Dict, List, Set, Tuple
from VLCL_AI.communication.subcarrier import SubcarrierPurpose
from VLCL_AI.communication.subcarrier_grid import SubcarrierGrid
from VLCL_AI.localization.frequency_plan import LocalizationFrequencyPlan

class SpectrumPartitioner:
    """
    Partitions the OFDM subcarrier grid into:
    1. A Localization Signal Group (SG_loc) corresponding to A-DPDOA tones.
    2. K Communication Signal Groups (SGs) allocated to different LEDs/users.
    
    Generates the static allocation matrix rho[k, n] where rho[k, n] = 1 if 
    subcarrier n belongs to communication group k.
    """
    
    def __init__(
        self,
        grid: SubcarrierGrid,
        frequency_plan: LocalizationFrequencyPlan,
        num_comm_groups: int = 4,
        guard_width: int = 1
    ):
        self.grid = grid
        self.plan = frequency_plan
        self.num_comm_groups = num_comm_groups
        self.guard_width = guard_width
        
        # SGs and allocation matrices
        self.loc_indices: Set[int] = set()
        self.comm_groups: Dict[int, List[int]] = {g: [] for g in range(1, num_comm_groups + 1)}
        self.rho = np.zeros((num_comm_groups + 1, self.grid.fft_size), dtype=int) # shape: (K+1, N)
        
        self.partition_spectrum()

    def partition_spectrum(self):
        """
        Determines the subcarriers associated with localization tones and
        partitions the remaining communication subcarriers.
        """
        # 1. Identify localization subcarrier indices
        subcarrier_spacing = self.grid.sample_rate / self.grid.fft_size
        
        # Find exact FFT bins closest to localization frequencies
        raw_loc_indices = []
        for freq in self.plan.frequencies:
            # Check positive frequency bins (0 to Fs/2)
            idx = int(round(freq / subcarrier_spacing))
            if 0 < idx < self.grid.fft_size // 2:
                raw_loc_indices.append(idx)
                # Map negative symmetric conjugate index for hermitian symmetry
                sym_idx = self.grid.fft_size - idx
                raw_loc_indices.append(sym_idx)
        
        # Apply guard bands around the localization tones to prevent leakage
        for idx in raw_loc_indices:
            for offset in range(-self.guard_width, self.guard_width + 1):
                target_idx = (idx + offset) % self.grid.fft_size
                # Avoid overwriting DC, Guard band edges, or Nyquist subcarriers
                if (target_idx != 0 and 
                    target_idx != self.grid.fft_size // 2 and 
                    target_idx >= self.grid.guard_low and 
                    target_idx < self.grid.fft_size - self.grid.guard_high):
                    self.loc_indices.add(target_idx)
                    
        # Update the SubcarrierGrid to reflect these reservations
        for k in range(self.grid.fft_size):
            sc = self.grid.subcarriers[k]
            if k in self.loc_indices:
                sc.purpose = SubcarrierPurpose.LOCALIZATION_RESERVED
                sc.active = False
                sc.reserved = True
            elif sc.purpose == SubcarrierPurpose.LOCALIZATION_RESERVED:
                # If it was previously reserved but not in our active list, free it
                sc.purpose = SubcarrierPurpose.COMMUNICATION
                sc.active = True
                sc.reserved = False

        # 2. Extract remaining active communication subcarriers
        # Only subcarriers in the lower half (1 to N/2-1) are independent
        # Hermitian symmetry means k and N-k are conjugate paired
        active_independent = []
        for k in range(1, self.grid.fft_size // 2):
            sc = self.grid.subcarriers[k]
            if sc.active and sc.purpose == SubcarrierPurpose.COMMUNICATION:
                active_independent.append(k)

        # 3. Partition independent communication subcarriers into K groups
        if len(active_independent) > 0:
            # Contiguous partitioning
            chunks = np.array_split(active_independent, self.num_comm_groups)
            for g_idx, chunk in enumerate(chunks):
                group_id = g_idx + 1
                for k in chunk:
                    self.comm_groups[group_id].append(int(k))
                    # Add symmetric index
                    sym_idx = self.grid.fft_size - k
                    self.comm_groups[group_id].append(int(sym_idx))

        # 4. Fill allocation matrix rho
        # rho[k, n] = 1 for communication group k (1-indexed, 1 to K)
        for g_id, indices in self.comm_groups.items():
            for idx in indices:
                self.rho[g_id, idx] = 1
                
        # Group 0 is reserved for localization
        for idx in self.loc_indices:
            self.rho[0, idx] = 1

    def assign_comm_groups(self, rho: np.ndarray, device_ids) -> None:
        """
        Makes the communication groups follow a Module 6 allocation: device k's group
        becomes exactly the subcarriers allocated to it (plus their Hermitian mirrors).
        Without this the transmitter and receiver kept the fixed contiguous split and an
        allocation only changed which bins carried bits inside it.
        """
        N = self.grid.fft_size
        groups = {g: [] for g in range(1, self.num_comm_groups + 1)}
        for k_idx, dev_id in enumerate(device_ids):
            if dev_id not in groups:
                continue
            for n in range(1, N // 2):
                if rho[k_idx, n] == 1 and n not in self.loc_indices:
                    groups[dev_id].extend([int(n), int(N - n)])
        self.comm_groups = groups
        self.rho[1:, :] = 0
        for g_id, indices in self.comm_groups.items():
            for idx in indices:
                self.rho[g_id, idx] = 1

    def get_group_for_subcarrier(self, subcarrier_index: int) -> int:
        """Returns the group ID (1 to K) for a subcarrier index, or 0 if localization/guard/DC."""
        if subcarrier_index in self.loc_indices:
            return 0
        for g_id, indices in self.comm_groups.items():
            if subcarrier_index in indices:
                return g_id
        return -1 # DC, Guard, or Pilot
