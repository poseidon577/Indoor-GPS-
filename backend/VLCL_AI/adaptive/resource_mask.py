# resource_mask.py
from enum import Enum
from typing import Dict, List, Set, Tuple, Optional
from VLCL_AI.communication.subcarrier_grid import SubcarrierGrid
from VLCL_AI.communication.subcarrier import SubcarrierPurpose

class SubcarrierLockType(Enum):
    AVAILABLE_COMM = "AVAILABLE_COMM"
    ALLOCATED_COMM = "ALLOCATED_COMM"
    LOCALIZATION_LOCKED = "LOCALIZATION_LOCKED"
    GUARD_LOCKED = "GUARD_LOCKED"
    DC_LOCKED = "DC_LOCKED"
    NYQUIST_LOCKED = "NYQUIST_LOCKED"
    PILOT_LOCKED = "PILOT_LOCKED"
    NULL = "NULL"

class ResourceMask:
    """
    Manages subcarrier reservation masks and locks to protect localization (SG_{K+1}),
    guard bands, DC carriers, pilots, and Nyquist frequencies.
    """

    def __init__(self, grid: SubcarrierGrid, localization_indices: Optional[List[int]] = None):
        self.fft_size = grid.fft_size
        self.masks: Dict[int, SubcarrierLockType] = {}
        self._build_mask(grid, localization_indices or [])

    def _build_mask(self, grid: SubcarrierGrid, localization_indices: List[int]):
        """Categorizes every subcarrier in the grid."""
        loc_set = set(localization_indices)
        
        for k, sc in grid.subcarriers.items():
            if k == 0:
                self.masks[k] = SubcarrierLockType.DC_LOCKED
            elif k == grid.fft_size // 2:
                self.masks[k] = SubcarrierLockType.NYQUIST_LOCKED
            elif sc.purpose == SubcarrierPurpose.GUARD or k < grid.guard_low or k >= grid.fft_size - grid.guard_high:
                self.masks[k] = SubcarrierLockType.GUARD_LOCKED
            elif k in loc_set or sc.purpose == SubcarrierPurpose.LOCALIZATION_RESERVED or sc.reserved:
                self.masks[k] = SubcarrierLockType.LOCALIZATION_LOCKED
            elif sc.purpose == SubcarrierPurpose.PILOT or sc.pilot:
                self.masks[k] = SubcarrierLockType.PILOT_LOCKED
            elif k > grid.fft_size // 2:
                # DCO-OFDM: bin N-k is the conjugate of bin k and carries no extra data
                self.masks[k] = SubcarrierLockType.NULL
            elif sc.active and sc.purpose == SubcarrierPurpose.COMMUNICATION:
                self.masks[k] = SubcarrierLockType.AVAILABLE_COMM
            else:
                self.masks[k] = SubcarrierLockType.NULL

    def get_available_comm_indices(self) -> List[int]:
        """Returns sorted list of communication subcarrier indices available for allocation."""
        return [k for k, lock_type in self.masks.items() if lock_type == SubcarrierLockType.AVAILABLE_COMM]

    def is_allocatable(self, index: int) -> bool:
        """Returns True if subcarrier index is an available communication subcarrier."""
        return self.masks.get(index) == SubcarrierLockType.AVAILABLE_COMM

    def is_localization_locked(self, index: int) -> bool:
        """Returns True if subcarrier is locked for localization."""
        return self.masks.get(index) == SubcarrierLockType.LOCALIZATION_LOCKED

    def get_lock_type(self, index: int) -> SubcarrierLockType:
        return self.masks.get(index, SubcarrierLockType.NULL)
