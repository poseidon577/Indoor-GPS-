import sys
import os

# Add the backend directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from VLCL_AI.pipeline import VLCLPipeline
from VLCL_AI.system_state import SystemState

@pytest.mark.xfail(strict=True, reason="VLCLPipeline builds VLCLSimulator() without a scene or mobility engine; orchestrator unfinished")
def test_pipeline():
    print("Initializing pipeline...")
    pipeline = VLCLPipeline()
    
    print("Running baseline step...")
    state = pipeline.step()
    print("Baseline state complete:", state)
    print("Metrics:", state.metrics.to_dict() if state.metrics else None)
    
    print("Running adaptive step...")
    state = pipeline.step(adaptive_mode=True)
    print("Adaptive state complete:", state)
    print("Metrics:", state.metrics.to_dict() if state.metrics else None)
    
    print("Running optimization step...")
    state = pipeline.step(optimize=True)
    print("Optimization state complete:", state)
    print("Metrics:", state.metrics.to_dict() if state.metrics else None)

if __name__ == "__main__":
    test_pipeline()
    print("Pipeline integration test passed!")
