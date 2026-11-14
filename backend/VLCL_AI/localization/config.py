# config.py
import os
import yaml
from typing import List, Dict, Any, Optional
from VLCL_AI.localization.exceptions import ConfigurationError

class LocalizationConfig:
    """Manages parsing, defaults, and validation of localization parameters."""
    
    def __init__(self, config_data: Optional[Dict[str, Any]] = None):
        self.raw_config = config_data or {}
        
        # Extracted config matching default schema
        loc_data = self.raw_config.get("localization", {})
        
        self.enabled = loc_data.get("enabled", True)
        self.algorithm = loc_data.get("algorithm", "A_DPDOA")
        self.signal_mode = loc_data.get("signal_mode", "phase_equivalent") # "phase_equivalent" or "full_waveform"
        
        # Frequency Plan
        fp = loc_data.get("frequency_plan", {})
        self.fp_count = fp.get("count", loc_data.get("tone_count", 5))
        self.fp_start_freq = fp.get("start_frequency_hz", loc_data.get("start_frequency_hz", 1.0e6))
        self.fp_spacing = fp.get("spacing_hz", loc_data.get("spacing_hz", 1.0e5))
        self.enforce_arithmetic_progression = fp.get("enforce_arithmetic_progression", True)
        
        # Sampling
        samp = loc_data.get("sampling", {})
        self.sample_rate = samp.get("sample_rate_hz", loc_data.get("sample_rate_hz", 10.0e6))
        self.duration = samp.get("duration_s", loc_data.get("duration_s", 0.01))
        
        self.initial_phase = loc_data.get("initial_phase_rad", 0.0)
        
        # Power
        pwr = loc_data.get("power", {})
        self.default_tone_power = pwr.get("default_tone_power_w", 0.1)
        self.per_tone_power = pwr.get("per_tone_power_w", [self.default_tone_power] * self.fp_count)
        
        # Channel
        chan = loc_data.get("channel", {})
        self.channel_mode = chan.get("mode", "los_only") # "los_only" or "multipath"
        self.use_module2_noise = chan.get("use_module2_noise", True)
        
        # Filters
        filts = loc_data.get("filters", {})
        bp = filts.get("bandpass", {})
        self.bp_type = bp.get("type", "butterworth")
        self.bp_order = bp.get("order", 4)
        self.bp_bandwidth = bp.get("bandwidth_hz", 20000.0)
        
        lp = filts.get("lowpass", {})
        self.lp_type = lp.get("type", "butterworth")
        self.lp_order = lp.get("order", 4)
        self.lp_cutoff = lp.get("cutoff_hz", 10000.0)
        
        # Phase
        ph = loc_data.get("phase", {})
        self.unwrap_phase = ph.get("unwrap", True)
        self.ambiguity_resolution = ph.get("ambiguity_resolution", "physical_constraints")
        self.offline_zero_phase = ph.get("offline_zero_phase", True)
        
        # Solver
        sol = loc_data.get("solver", {})
        self.solver_method = sol.get("method", "trust_region") # "trust_region", "levenberg-marquardt"
        self.solver_dimensions = sol.get("dimensions", "2D_fixed_height") # "2D_fixed_height" or "3D"
        self.solver_initialization = sol.get("initialization", "previous_or_room_center")
        self.solver_robust_loss = sol.get("robust_loss", "soft_l1")
        self.solver_max_iterations = sol.get("max_iterations", 200)
        self.solver_tolerance = sol.get("tolerance", 1.0e-9)
        self.fixed_height = sol.get("fixed_height_m", 0.85)
        # "hybrid": A-DPDOA distance differences + RSS from the same tones, solved in 3D
        # (handles any phone height). "adpdoa": the paper's distance-difference-only solve.
        self.solver_mode = sol.get("mode", "hybrid")
        self.hybrid_sigma_dd_m = sol.get("hybrid_sigma_dd_m", 0.005)
        self.hybrid_sigma_log_rss = sol.get("hybrid_sigma_log_rss", 0.02)
        # Tilt reported by the phone's IMU: true tilt plus this much error (degrees, 1 sigma)
        self.imu_tilt_noise_deg = sol.get("imu_tilt_noise_deg", 1.0)
        self.imu_seed = sol.get("imu_seed", 7)
        
        # Tone to LED mapping (Configurable mapping)
        # Default maps Tones 1,2,3,4 to LEDs 1,2,3,4 and Tone 5 to LED 1
        self.tone_to_led_map = loc_data.get("tone_to_led_map", {
            1: [1],
            2: [2],
            3: [3],
            4: [4],
            5: [1]
        })
        
        # Calibration
        cal = loc_data.get("calibration", {})
        self.cal_enabled = cal.get("enabled", True)
        self.cal_phase_bias = cal.get("phase_bias_rad", [])
        self.cal_delay_bias = cal.get("delay_bias_s", [])
        
        # Confidence
        conf = loc_data.get("confidence", {})
        self.min_snr_db = conf.get("minimum_snr_db", 5.0)
        self.max_condition_number = conf.get("maximum_condition_number", 1.0e8)
        
        # Debug
        dbg = loc_data.get("debug", {})
        self.save_intermediate_signals = dbg.get("save_intermediate_signals", False)
        
        # Automatically validate configuration
        self.validate()

    def validate(self):
        """Runs sanity checks on the configuration and raises ConfigurationError if invalid."""
        if self.fp_count != 5:
            raise ConfigurationError(f"A-DPDOA algorithm requires exactly 5 frequencies, got {self.fp_count}.")
        
        if self.solver_mode not in ("hybrid", "adpdoa"):
            raise ConfigurationError(f"solver.mode must be 'hybrid' or 'adpdoa', got {self.solver_mode!r}.")

        if self.fp_start_freq <= 0:
            raise ConfigurationError(f"Start frequency must be positive, got {self.fp_start_freq} Hz.")
            
        if self.fp_spacing <= 0:
            raise ConfigurationError(f"Frequency spacing must be positive, got {self.fp_spacing} Hz.")
            
        if self.sample_rate <= 0:
            raise ConfigurationError(f"Sample rate must be positive, got {self.sample_rate} Hz.")
            
        if self.duration <= 0:
            raise ConfigurationError(f"Signal duration must be positive, got {self.duration} seconds.")
            
        # Verify Nyquist Shannon limit for full waveform mode
        if self.signal_mode == "full_waveform":
            max_freq = self.fp_start_freq + (self.fp_count - 1) * self.fp_spacing
            if self.sample_rate <= 2 * max_freq:
                raise ConfigurationError(
                    f"Sample rate {self.sample_rate} Hz does not satisfy Nyquist limit "
                    f"for max frequency {max_freq} Hz. Must be > {2 * max_freq} Hz."
                )
                
        # Validate tone powers
        if len(self.per_tone_power) != self.fp_count:
            # Pad or truncate to match fp_count
            if len(self.per_tone_power) < self.fp_count:
                self.per_tone_power.extend([self.default_tone_power] * (self.fp_count - len(self.per_tone_power)))
            else:
                self.per_tone_power = self.per_tone_power[:self.fp_count]
                
        for i, p in enumerate(self.per_tone_power):
            if p < 0:
                raise ConfigurationError(f"Power for tone {i+1} cannot be negative: {p} W.")

        # Ensure mapping is clean
        # Keys of tone_to_led_map must be ints
        clean_map = {}
        for k, v in self.tone_to_led_map.items():
            try:
                tone_id = int(k)
                if not isinstance(v, list):
                    clean_map[tone_id] = [int(v)]
                else:
                    clean_map[tone_id] = [int(x) for x in v]
            except (ValueError, TypeError):
                raise ConfigurationError(f"Invalid tone_to_led_map keys or values: {k} -> {v}")
        self.tone_to_led_map = clean_map

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "LocalizationConfig":
        """Loads configuration from YAML file."""
        if not os.path.exists(yaml_path):
            raise ConfigurationError(f"Configuration file not found: {yaml_path}")
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f) or {}
                return cls(data)
        except Exception as e:
            raise ConfigurationError(f"Failed to parse YAML configuration: {e}")
