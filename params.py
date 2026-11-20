#!/usr/bin/env python3
"""Registered parameter counts and tensor payload sizes (no latency benchmark)."""
import json
from models import LiSalNet
from profiling import count_parameters

if __name__ == '__main__':
    print(json.dumps(count_parameters(LiSalNet(pretrained=False)), indent=2))
