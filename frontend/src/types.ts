export interface RoomParams {
  width: number;
  length: number;
  height: number;
  wallReflectivity: number;
  floorReflectivity: number;
  ceilingReflectivity: number;
}

export interface LEDParams {
  id: number;
  position: [number, number, number];
  orientation: [number, number, number];
  power: number;
  biasCurrent: number;
  frequency: number;
  lambertianOrder: number;
  beamAngle: number;
  fov: number;
  communicationEnabled: boolean;
  localizationEnabled: boolean;
}

export interface ReceiverParams {
  position: [number, number, number];
  orientation: [number, number, number];
  velocity: [number, number, number];
  acceleration: [number, number, number];
  fov: number;
  apdSize: number;
  noise: number;
  gain: number;
  roll: number;
  pitch: number;
  yaw: number;
}

export interface ObstacleParams {
  id: string;
  type: "box" | "cylinder" | "sphere";
  position: [number, number, number];
  rotation: [number, number, number]; // roll, pitch, yaw
  scale: [number, number, number]; // box: [dx,dy,dz], cylinder: [r,r,h], sphere: [r,r,r]
  reflectivity: number;
  material: string;
}

export interface MobilityParams {
  type: "static" | "linear" | "circular" | "random_walk" | "waypoint";
  speed: number;
  radius: number;
  center: [number, number, number];
  waypoints: [number, number, number][];
}

// Full physics metrics returned by the PhysicsEngine
export interface PhysicsMetrics {
  snrs: Record<string, number>;               // dB per LED
  received_powers: Record<string, number>;    // Watts per LED
  los_gains: Record<string, number>;          // LOS channel gain per LED
  nlos_gains: Record<string, number>;         // NLOS channel gain per LED
  electrical_currents: Record<string, number>; // Amps per LED
  voltages: Record<string, number>;           // Volts per LED
  metrics: {
    average_channel_gain: number;
    average_snr: number;                      // dB
    visible_leds: number;
    blocked_leds: number;
    propagation_delay: number;                // seconds
    total_optical_power: number;              // Watts
  };
}

// Communication KPIs returned by CommunicationEngine (Module 3)
export interface CommunicationMetrics {
  simulation_time: number;

  // Data Rate
  sum_rate_mbps: number;                      // Raw aggregate throughput in Mbps
  effective_throughput_mbps: number;          // BER-adjusted effective Mbps
  spectral_efficiency: number;                // bps/Hz

  // Waveform Quality
  papr_db: number;                            // Peak-to-Average Power Ratio (dB)
  clipping_ratio_pct: number;                 // % of samples clipped by DAC

  // Error Metrics (per user)
  ber_per_user: Record<string, number>;       // Bit Error Rate per user
  evm_per_user_pct: Record<string, number>;   // Error Vector Magnitude % per user
  rate_per_user_mbps: Record<string, number>; // Data rate per user in Mbps

  // Diagnostic metadata
  metadata: {
    active_led_id: number;                    // LED selected as primary transmitter
    bit_errors: number;                       // Raw count of errored bits in frame
    average_analytical_ber: number;           // Theoretical BER from Shannon bounds
    clipping_distortion: number;              // Signal distortion from clipping
    electrical_power: number;                 // Electrical drive power (W)
  };
}

// Localization metrics returned by LocalizationEngine (Module 4 — A-DPDOA)
export interface LocalizationMetrics {
  simulation_time: number;
  frame_id: number;

  // 3D coordinates
  estimated_position: [number, number, number];
  true_position: [number, number, number];

  // Error metrics
  errors: {
    instantaneous_m: number;
    horizontal_m: number;
    vertical_m: number;
    rmse_m: number;
  };

  // Signal info
  signals: {
    frequencies_hz: number[];
    tone_powers: number[];
    received_powers: Record<string, number>;
    localization_snr: Record<string, number>;
  };

  // Phase measurements
  measurements: {
    I: number[];
    Q: number[];
    wrapped_phases: number[];
    unwrapped_phases: number[];
  };

  // Geometry
  geometry: {
    distance_differences: Record<string, number>;
    used_led_ids: number[];
    rejected_measurements: string[];
  };

  // Solver
  solver: {
    method: string;
    iterations: number;
    cost: number;
    residual: number[];
  };

  // Quality
  quality: {
    confidence: number;
    status: string;   // "VALID" | "LOW_CONFIDENCE" | "SOLVER_FAILED" | "INSUFFICIENT_GEOMETRY"
    calibration_applied: boolean;
  };

  // Metadata (includes running_stats dict from LocalizationMetrics.get_metrics())
  metadata: Record<string, unknown>;
}

// Adaptive Transmission metrics returned by AdaptiveTransmissionEngine (Module 6)
export interface AdaptiveMetrics {
  sum_rate_bps: number;
  achievable_rates_bps: Record<string, number>;  // device_id -> rate (bps)
  qos_status: string;                            // "FEASIBLE" | "PARTIALLY_FEASIBLE" | "INFEASIBLE_QOS"
  qos_satisfied: Record<string, boolean>;        // device_id -> satisfied?
  qos_deficits_bps: Record<string, number>;      // device_id -> deficit (bps)
  unused_subcarriers_count: number;
  diagnostics: {
    sum_rate_bps: number;
    spectral_efficiency_bps_hz: number;
    jains_fairness_index: number;
    subcarrier_utilization_ratio: number;
    allocated_subcarrier_count: number;
    total_comm_subcarrier_count: number;
    average_bits_per_symbol: number;
  };
}

// Power & Pre-Equalization metrics returned by PowerPreEqualizationEngine (Module 7)
export interface PowerMetrics {
  power_allocation: {
    mode: string;                                       // "EQUAL_POWER" | "WATER_FILLING"
    total_power_budget_w: number;
    per_led_max_power_w: Record<string, number>;        // LED id -> max power (W)
    localization_reserved_power_w: Record<string, number>;
    communication_available_power_w: Record<string, number>;
    per_device_power_w: Record<string, number>;
  };
  pre_eq: {
    mode: string;                                       // "REGULARIZED" | "ZERO_FORCING" | "NONE"
    max_gain_db: number;
    papr_before_db: Record<string, number>;             // LED id -> PAPR before pre-eq (dB)
    papr_after_db: Record<string, number>;              // LED id -> PAPR after pre-eq (dB)
    clipping_ratio: Record<string, number>;
  };
  predicted_ber: Record<string, number>;               // device_id -> predicted BER
  modulation_feasible: Record<string, boolean>;        // device_id -> BER <= BER_max?
  nominal_sum_rate_bps: number;
  feasible_sum_rate_bps: number;
  warnings: string[];
}

export interface ConstraintStatus {
  localization_satisfied: boolean;
  qos_satisfied: boolean;
  ber_satisfied: boolean;
  power_satisfied: boolean;
  spectrum_satisfied: boolean;
  overall_feasible: boolean;
  localization_error_m: number;
  localization_target_m: number;
  rate_deficits_bps: Record<string, number>;
  ber_excesses: Record<string, number>;
  power_excess_w: number;
}

export interface JointMetrics {
  iteration_count: number;
  converged: boolean;
  convergence_reason: string;
  total_power_w: number;
  loc_power_w: number;
  comm_power_w: number;
  per_device_rates_bps: Record<string, number>;
  sum_rate_bps: number;
  per_device_ber: Record<string, number>;
  localization_error_m: number;
  constraint_status: ConstraintStatus;
  active_subcarriers_count: number;
  history_summary: Array<{
    iteration: number;
    sum_rate_bps: number;
    localization_error_m: number;
    loc_power_w: number;
    feasible: boolean;
  }>;
}

// Integrated metrics returned by IntegratedVLCLEngine (Module 5)
export interface IntegratedMetrics {
  simulation_time: number;
  communications: Record<string, {
    num_transmitted_bits: number;
    bit_errors: number;
    empirical_ber: number;
    num_recovered_symbols: number;
  }>;
  localization: {
    estimated_position: [number, number, number];
    error_3d_m: number;
    success: boolean;
  };
  transmitter: {
    papr_db: Record<string, number>;
    clipping_ratio_pct: Record<string, number>;
    dc_bias_volts: Record<string, number>;
  };
  metadata: Record<string, unknown>;
}

// Rooms with more than four LEDs (Python backend — integrated_vlcl/multicell.py)
export interface MulticellFix {
  success: boolean;
  status: string | null;
  estimated_position: [number, number, number] | null;
  error_xy_m: number | null;
  error_3d_m: number | null;
  cell?: Record<string, number>;
}
export interface MulticellMetrics {
  num_leds: number;
  colours: Record<string, number>;          // LED id -> tone set 1..4
  true_position: [number, number, number];
  interference_aware: MulticellFix;          // models every LED that shares a tone
  nearest_cell: MulticellFix;                // assumes each tone comes from the nearest LED
}

export interface SimulationState {
  currentTime: number;
  frameIndex: number;
  fps: number;
  isPlaying: boolean;
  speedFactor: number;
  room: RoomParams;
  leds: LEDParams[];
  receiver: ReceiverParams;
  obstacles: ObstacleParams[];
  mobility: MobilityParams;
  
  // Real-time calculated telemetry (JS)
  distances: Record<number, number>;
  incidentAngles: Record<number, number>;
  irradianceAngles: Record<number, number>;
  dcGains: Record<number, number>;
  losMatrix: Record<number, boolean>;
  visibilityMatrix: Record<number, boolean>;
  blockingObstacles: Record<number, string>;
  trajectoryPoints: [number, number, number][];

  // Full physics engine metrics (Python backend — Module 2)
  physicsMetrics: PhysicsMetrics | null;
  physicsLoading: boolean;

  // Communication engine metrics (Python backend — Module 3)
  commMetrics: CommunicationMetrics | null;
  commLoading: boolean;

  // Localization engine metrics (Python backend — Module 4)
  localizationMetrics: LocalizationMetrics | null;
  localizationLoading: boolean;

  // Many-light localization (Python backend — multicell), used when there are more than 4 LEDs
  multicellMetrics: MulticellMetrics | null;
  multicellLoading: boolean;

  // Integrated engine metrics (Python backend — Module 5)
  integratedMetrics: IntegratedMetrics | null;
  integratedLoading: boolean;

  // Adaptive Transmission engine metrics (Python backend — Module 6)
  adaptiveMetrics: AdaptiveMetrics | null;
  adaptiveLoading: boolean;

  // Power & Pre-Equalization engine metrics (Python backend — Module 7)
  powerMetrics: PowerMetrics | null;
  powerLoading: boolean;

  // Joint Optimization engine metrics (Python backend — Module 8)
  jointMetrics: JointMetrics | null;
  jointLoading: boolean;
}
