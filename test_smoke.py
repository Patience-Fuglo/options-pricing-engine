"""
Minimal smoke test for options_pricing package.
Run with: python test_smoke.py
"""
from options_pricing import pricing_engine

def test_pricing_engine_runs():
    try:
        pricing_engine.main()
        print("[PASS] pricing_engine.main() ran without error.")
    except Exception as e:
        print(f"[FAIL] pricing_engine.main() raised: {e}")

if __name__ == "__main__":
    test_pricing_engine_runs()
