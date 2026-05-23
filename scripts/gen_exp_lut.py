"""
Generate exp_lut.mem — 256-entry Q1.15 LUT for exp(x) over [-8, 0].
Used by softmax_engine.v for online softmax score computation.
"""

import math
from pathlib import Path


ENTRIES = 256
X_MIN = -8.0
X_MAX = 0.0
Q_SCALE = 32768  # 2^15 for Q1.15
Q_MAX = 32767    # 0x7FFF


def x_for_index(i: int) -> float:
    return X_MIN + i * (X_MAX - X_MIN) / (ENTRIES - 1)


def to_q1_15(val: float) -> int:
    return max(0, min(Q_MAX, round(val * Q_SCALE)))


def main() -> None:
    scripts_dir = Path(__file__).parent
    rtl_dir = scripts_dir.parent / "rtl"
    rtl_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for i in range(ENTRIES):
        x = x_for_index(i)
        q = to_q1_15(math.exp(x))
        entries.append(q)

    out_path = rtl_dir / "exp_lut.mem"
    with out_path.open("w") as f:
        for q in entries:
            f.write(f"{q:04X}\n")

    print(f"Wrote {ENTRIES} entries to {out_path}")
    print()

    header = f"{'Index':>6}  {'x':>10}  {'exp(x)':>12}  {'Q1.15 dec':>10}  {'Hex':>6}"
    print(header)
    print("-" * len(header))

    spot_indices = [0, 32, 64, 96, 128, 160, 192, 224, 255]
    for i in spot_indices:
        x = x_for_index(i)
        ex = math.exp(x)
        q = entries[i]
        print(f"{i:>6}  {x:>10.6f}  {ex:>12.8f}  {q:>10}  {q:>04X}")

    assert entries[255] == Q_MAX, (
        f"Entry 255 should be 0x{Q_MAX:04X} but got 0x{entries[255]:04X}"
    )
    assert entries[0] > 0, (
        f"Entry 0 (exp(-8)) should be nonzero but got {entries[0]}"
    )
    print()
    print("Assertions passed: entry[255] == 0x7FFF, entry[0] != 0")


if __name__ == "__main__":
    main()
