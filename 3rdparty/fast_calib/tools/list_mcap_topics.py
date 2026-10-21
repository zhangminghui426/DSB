#!/usr/bin/env python3
"""列出 mcap 全部 channel 话题（不依赖解码器，内置解析器兜底）。

用法:
  cd 3rdparty/fast_calib
  python tools/list_mcap_topics.py data/xxx.mcap
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from utils import scan_mcap_channels


def main():
    ap = argparse.ArgumentParser(description="列出 mcap 全部 channel 话题")
    ap.add_argument("mcap", help="mcap 文件路径")
    args = ap.parse_args()

    chans = scan_mcap_channels(args.mcap)
    if not chans:
        print("mcap 内无消息或无法解析")
        return
    for t, s, n in chans:
        print(f"{t}  ({s}, {n} msgs)")


if __name__ == "__main__":
    main()
