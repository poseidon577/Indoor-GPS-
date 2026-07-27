# demo_receiver_mobility.py
import numpy as np
from loguru import logger
from VLCL_AI.environment.room import Room
from VLCL_AI.environment.led import LED
from VLCL_AI.environment.receiver import Receiver
from VLCL_AI.environment.scene import Scene
from VLCL_AI.environment.simulator import VLCLSimulator, MobilityEngine
from VLCL_AI.physics.physics_engine import PhysicsEngine
from VLCL_AI.communication.engine import CommunicationEngine

def main():
    logger.info("Initializing Receiver Mobility VLC Tracking Demo...")
    
    # 1. Setup Scene (Module 1)
    room = Room(5.0, 5.0, 3.0)
    receiver = Receiver([2.5, 2.5, 1.0], [0.0, 0.0, 1.0])
    led = LED(1, [2.5, 2.5, 3.0], [0.0, 0.0, -1.0], power=20.0)
    scene = Scene(room, receiver, [led])
    
    # Simple linear motion from center [2.5, 2.5] to corner [0.5, 0.5] over 5 steps
    positions = [
        np.array([2.5, 2.5, 1.0]),
        np.array([2.0, 2.0, 1.0]),
        np.array([1.5, 1.5, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.5, 0.5, 1.0])
    ]
    
    physics = PhysicsEngine()
    comm_engine = CommunicationEngine()
    
    print("\n" + "="*80)
    print("                VLC TRACKING ACROSS RECEIVER TRAJECTORY")
    print("="*80)
    print(f"{'Position (m)':<22}{'Distance (m)':<15}{'Avg SNR (dB)':<15}{'Empirical BER':<15}{'Rate (Mbps)'}")
    print("-"*80)
    
    for pos in positions:
        receiver.position = pos
        # Run environment simulation frame
        metrics = scene.get_geometric_metrics()
        env_state = scene.render()
        
        # Build dynamic state representation
        from VLCL_AI.environment.state import EnvironmentState
        env_snapshot = EnvironmentState(
            current_time=1.0,
            frame_index=1,
            fps=60.0,
            receiver_position=pos.tolist(),
            receiver_orientation=receiver.orientation.tolist(),
            receiver_velocity=[0.0, 0.0, 0.0],
            receiver_acceleration=[0.0, 0.0, 0.0],
            receiver_angles={"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
            led_positions={1: led.position.tolist()},
            led_powers={1: led.power},
            led_active={1: led.active},
            distances=metrics["distances"],
            incident_angles_rad=metrics["incident_angles_rad"],
            irradiance_angles_rad=metrics["irradiance_angles_rad"],
            led_orientations=metrics["led_orientations"],
            led_beam_angles=metrics["led_beam_angles"],
            visibility_matrix=metrics["visibility_matrix"],
            los_matrix=metrics["los_matrix"],
            blocking_obstacles=metrics["blocking_obstacles"],
            room_dims=[5.0, 5.0, 3.0],
            obstacles=[]
        )
        
        # Get optical parameters
        phys_state = physics.step(env_snapshot)
        
        # Run OFDM communication link
        comm_state = comm_engine.step(env_snapshot, phys_state)
        
        pos_str = f"[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]"
        dist = metrics["distances"][1]
        snr_avg = np.mean(comm_state.snr_per_subcarrier)
        ber = comm_state.ber_per_user[1]
        rate_mbps = comm_state.sum_rate / 1e6
        
        print(f"{pos_str:<22}{dist:<15.2f}{snr_avg:<15.2f}{ber:<15.2e}{rate_mbps:<15.2f}")
        
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
