"""CLI for the packaged offline PiSA-SR runtime; never publishes character maps."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'create_metahuman'))
from texture_sr_prior import run, verify

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/albedo-superresolution/pisa-sr')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scale', type=int, choices=(2, 4), default=2)
    parser.add_argument('--pixel', type=float, default=1)
    parser.add_argument('--semantic', type=float, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tile', type=int, default=64)
    parser.add_argument('--prompt', default='')
    run(parser.parse_args())
