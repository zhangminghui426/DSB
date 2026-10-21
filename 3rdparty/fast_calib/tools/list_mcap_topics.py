#!/usr/bin/env python3
"""列出 mcap 全部 channel 话题（不依赖解码器，优先用 summary）。

用法:
  cd 3rdparty/fast_calib
  python tools/list_mcap_topics.py data/xxx.mcap
"""
import argparse
from collections import Counter

from mcap.reader import make_reader


def main():
    ap = argparse.ArgumentParser(description="列出 mcap 全部 channel 话题")
    ap.add_argument("mcap", help="mcap 文件路径")
    args = ap.parse_args()

    cnt = Counter()
    names = {}
    with open(args.mcap, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        if summary is not None and summary.channels:
            for cid, ch in summary.channels.items():
                cnt[ch.topic] = summary.statistics.channel_message_counts.get(cid, 0)
                names[ch.topic] = (summary.schemas[ch.schema_id].name
                                   if ch.schema_id in summary.schemas else "?")
        else:
            for schema, channel, _ in reader.iter_messages():
                cnt[channel.topic] += 1
                names[channel.topic] = schema.name

    if not cnt:
        print("mcap 内无消息")
        return
    for t in sorted(cnt):
        print(f"{t}  ({names[t]}, {cnt[t]} msgs)")


if __name__ == "__main__":
    main()
