import logging
import os
import struct
import zlib
from collections import Counter

import yaml
import numpy as np
import open3d as o3d
from typing import Optional, Tuple

from mcap.reader import make_reader

# 通用 opcode（两版规范一致）
_OP_SCHEMA = 0x03
_OP_CHANNEL = 0x04
_OP_MSG_INDEX = 0x07
# mcap 文件头/尾 magic：老规范与 mcap>=1.x 两种写法都兼容
_MCAP_MAGICS = (b"\x89MCAP\x01\x13\x37\x0a", b"\x89MCAP0\r\n")
# 版本相关 opcode：老规范(MCAP0 0.2.x) 与新规范(0.3.x/1.x) 不同
_OP_TABLES = {
    b"\x89MCAP\x01\x13\x37\x0a": {"message": 0x06, "chunk": 0x05,
                                  "stats": 0x0A, "data_end": 0x0D,
                                  "new": False},
    b"\x89MCAP0\r\n": {"message": 0x05, "chunk": 0x06,
                       "stats": 0x0B, "data_end": 0x0F,
                       "new": True},
}


def _read_str(buf, off):
    n = struct.unpack_from("<I", buf, off)[0]
    return buf[off + 4:off + 4 + n].decode("utf-8", "replace"), off + 4 + n


def _iter_records(buf, start=0):
    """顺序迭代 record：(opcode, data_offset, length)，非法/越界即停止。"""
    off = start
    n = len(buf)
    while off + 9 <= n:
        opcode = buf[off]
        length = struct.unpack_from("<Q", buf, off + 1)[0]
        if length > n - off - 9:
            break
        yield opcode, off + 9, length
        off += 9 + length


def _parse_schema(buf, off):
    sid = struct.unpack_from("<H", buf, off)[0]
    off += 2
    name, off = _read_str(buf, off)
    _, off = _read_str(buf, off)  # encoding
    return sid, name


def _parse_channel(buf, off, new_spec):
    cid = struct.unpack_from("<H", buf, off)[0]
    schema_id = struct.unpack_from("<H", buf, off + 2)[0]
    off += 4
    topic, off = _read_str(buf, off)
    _, off = _read_str(buf, off)  # message_encoding
    n = struct.unpack_from("<I", buf, off)[0]
    off += 4
    if new_spec:
        # metadata 为字节长度前缀 + (prefixed_string, prefixed_string)*
        end = off + n
        while off < end:
            _, off = _read_str(buf, off)
            _, off = _read_str(buf, off)
    else:
        # 老规范 metadata 为条目数
        for _ in range(n):
            _, off = _read_str(buf, off)
            _, off = _read_str(buf, off)
    return cid, topic, schema_id


def _parse_stats_counts(buf, off, new_spec):
    """Statistics.channel_message_counts。
    老规范: message_count(u64) schema_count(u16) channel_count(u32)
            message_start(u64) message_end(u64)
            channel_message_counts(u32 条数 + (u16,u64)*n)
    新规范: 中间多 attachment/metadata/chunk_count 三个 u32，
            channel_message_counts 前缀是字节长度。"""
    o = off + 8 + 2 + 4
    if new_spec:
        o += 4 + 4 + 4  # attachment_count, metadata_count, chunk_count
    o += 8 + 8  # message_start_time, message_end_time
    n = struct.unpack_from("<I", buf, o)[0]
    o += 4
    n_entries = n // 10 if new_spec else n
    counts = {}
    for _ in range(n_entries):
        cid = struct.unpack_from("<H", buf, o)[0]
        cnt = struct.unpack_from("<Q", buf, o + 2)[0]
        counts[cid] = cnt
        o += 10
    return counts


def _decompress(records, compression):
    c = (compression or "").lower()
    if c in ("", "none"):
        return records
    try:
        if c == "zstd":
            import zstandard
            # 部分写入器(rosbag2/mcap-cpp)的 zstd 帧不带 content size，
            # 用流式解压兼容两种
            dctx = zstandard.ZstdDecompressor()
            return dctx.decompressobj().decompress(records)
        if c == "lz4":
            import lz4.frame
            return lz4.frame.decompress(records)
        if c in ("zlib", "deflate"):
            return zlib.decompress(records)
    except Exception as e:
        logging.warning(f"chunk 解压失败 ({compression}: {e})")
    return None


def _scan_records(buf, start, end, ops, sid2name, cid2topic, cid2sid, cnt):
    for opcode, off, length in _iter_records(buf, start):
        if off + length > end:
            break
        try:
            if opcode == _OP_SCHEMA:
                sid, name = _parse_schema(buf, off)
                sid2name[sid] = name
            elif opcode == _OP_CHANNEL:
                cid, topic, schema_id = _parse_channel(buf, off, ops["new"])
                cid2topic[cid] = topic
                cid2sid[cid] = schema_id
            elif opcode == ops["chunk"]:
                # 头部: start(u64) end(u64) uncompressed_size(u64)
                #       [uncompressed_crc(u32) 仅新规范] compression(str) records(bytes u64)
                o = off + 8 + 8 + 8
                if ops["new"]:
                    o += 4  # uncompressed_crc
                compression, o = _read_str(buf, o)
                rlen = struct.unpack_from("<Q", buf, o)[0]
                o += 8
                dec = _decompress(buf[o:o + rlen], compression)
                if dec is not None:
                    _scan_records(dec, 0, len(dec), ops, sid2name, cid2topic, cid2sid, cnt)
            elif opcode == ops["message"]:
                cid = struct.unpack_from("<H", buf, off)[0]
                topic = cid2topic.get(cid)
                if topic is not None:
                    cnt[topic] += 1
            elif opcode == _OP_MSG_INDEX:
                pass  # 与 Message 记录同批消息，避免重复计数
            elif opcode == ops["data_end"]:
                break
        except Exception:
            continue  # 单条记录解析失败时跳过，不拖垮整体扫描


def _manual_scan(mcap_path):
    """不依赖 mcap 库版本的内置解析：summary 区收 Channel/Schema，
    Statistics 或数据区 Message/MessageIndex 补计数。"""
    with open(mcap_path, "rb") as f:
        data = f.read()
    # 仅要求头部 magic 匹配（尾部可能因未封口/截断缺失）
    magic = next((m for m in _MCAP_MAGICS if data.startswith(m)), None)
    if magic is None:
        return []
    ops = _OP_TABLES[magic]

    sid2name, cid2topic, cid2sid = {}, {}, {}
    counts = {}

    # 1) summary 区（不压缩）：Schema/Channel/Statistics
    try:
        summary_start = None
        # footer record 布局: [opcode(1B)][length(8B)][data][magic(8B)]
        # data 老规范 24B（message/chunk count），新规范 20B（summary_crc）
        for flen in (24, 20):
            op_pos = len(data) - 8 - flen - 9
            if op_pos >= 0 and data[op_pos] == 0x02:
                l = struct.unpack_from("<Q", data, op_pos + 1)[0]
                if l == flen:
                    summary_start = struct.unpack_from("<Q", data, op_pos + 9)[0]
                    break
        if summary_start is not None and summary_start < len(data):
            for opcode, off, length in _iter_records(data, summary_start):
                if opcode == _OP_SCHEMA:
                    sid, name = _parse_schema(data, off)
                    sid2name[sid] = name
                elif opcode == _OP_CHANNEL:
                    cid, topic, schema_id = _parse_channel(data, off, ops["new"])
                    cid2topic[cid] = topic
                    cid2sid[cid] = schema_id
                elif opcode == ops["stats"]:
                    counts = _parse_stats_counts(data, off, ops["new"])
                elif opcode == 0x02:  # Footer
                    break
    except Exception as e:
        logging.warning(f"内置 summary 解析失败 ({e})")

    # 2) 数据区：Message/MessageIndex 计数（直写或 chunk 内层均可）
    cnt = Counter()
    try:
        sid2name_d, cid2topic_d, cid2sid_d = {}, {}, {}
        _scan_records(data, len(magic), len(data), ops,
                      sid2name_d, cid2topic_d, cid2sid_d, cnt)
        # 数据区内层出现的 Channel/Schema 优先（chunk 场景）
        if cid2topic_d:
            sid2name.update(sid2name_d)
            cid2topic.update(cid2topic_d)
            cid2sid.update(cid2sid_d)
    except Exception as e:
        logging.warning(f"内置 data section 解析失败 ({e})")

    if not cid2topic:
        return []

    # 计数：summary Statistics（key=channel_id）优先，否则数据区计数（key=topic）
    cnt_by_topic = dict(cnt)
    result = []
    for cid, t in cid2topic.items():
        c = counts.get(cid, 0) if counts else cnt_by_topic.get(t, 0)
        result.append((t, sid2name.get(cid2sid.get(cid, -1), "?"), c))
    return result


def _parse_schema_full(buf, rec_off, length):
    """完整解析 Schema record（含 data 段）。data 为长度前缀 + 定义文本：
    libmcap 0.8 / mcap python 库用 u32 前缀，老规范为 u64；
    用记录边界严格相等校验选择，异常时退化为整体尾部。"""
    class _Schema:
        __slots__ = ("id", "name", "encoding", "data")
    sid = struct.unpack_from("<H", buf, rec_off)[0]
    off = rec_off + 2
    name, off = _read_str(buf, off)
    encoding, off = _read_str(buf, off)
    end = rec_off + length
    n32 = struct.unpack_from("<I", buf, off)[0]
    if off + 4 + n32 == end:
        data = buf[off + 4:off + 4 + n32]
    else:
        n64 = struct.unpack_from("<Q", buf, off)[0]
        if off + 8 + n64 == end:
            data = buf[off + 8:off + 8 + n64]
        else:
            data = buf[off:end]
    s = _Schema()
    s.id, s.name, s.encoding, s.data = sid, name, encoding, data
    return s


def _parse_channel_full(buf, off, new_spec):
    class _Channel:
        __slots__ = ("id", "topic", "message_encoding", "metadata", "schema_id")
    cid = struct.unpack_from("<H", buf, off)[0]
    schema_id = struct.unpack_from("<H", buf, off + 2)[0]
    off += 4
    topic, off = _read_str(buf, off)
    msg_enc, off = _read_str(buf, off)
    n = struct.unpack_from("<I", buf, off)[0]
    off += 4
    metadata = {}
    if new_spec:
        # metadata 为字节长度前缀 + (prefixed_string, prefixed_string)*
        end = off + n
        while off < end:
            k, off = _read_str(buf, off)
            v, off = _read_str(buf, off)
            metadata[k] = v
    else:
        # 老规范 metadata 为条目数
        for _ in range(n):
            k, off = _read_str(buf, off)
            v, off = _read_str(buf, off)
            metadata[k] = v
    ch = _Channel()
    ch.id, ch.topic, ch.message_encoding = cid, topic, msg_enc
    ch.metadata, ch.schema_id = metadata, schema_id
    return ch


def iter_mcap_messages(mcap_path, topics=None):
    """不依赖 mcap 库 summary 的消息迭代：产出 (schema, channel, message)。

    适用于尾部未封口/截断等 summary 不可用的 rosbag2 mcap（此时
    iter_decoded_messages 会在 get_summary 崩溃）。产出对象用
    mcap.records 构造，可直接交给 mcap.decoder.Decoder 解码。
    topics 为 None 时产出全部话题。"""
    class _Message:
        __slots__ = ("channel_id", "log_time", "publish_time", "data")
    with open(mcap_path, "rb") as f:
        data = f.read()
    magic = next((m for m in _MCAP_MAGICS if data.startswith(m)), None)
    if magic is None:
        return
    ops = _OP_TABLES[magic]
    sid2sch, cid2chan = {}, {}

    def handle(buf, start, end):
        for opcode, off, length in _iter_records(buf, start):
            if off + length > end:
                break
            try:
                if opcode == _OP_SCHEMA:
                    sch = _parse_schema_full(buf, off, length)
                    # 首次出现为准：尾部损坏 chunk 的错位 record 不得污染定义
                    sid2sch.setdefault(sch.id, sch)
                elif opcode == _OP_CHANNEL:
                    chan = _parse_channel_full(buf, off, ops["new"])
                    cid2chan.setdefault(chan.id, chan)
                elif opcode == ops["message"]:
                    chan = cid2chan.get(struct.unpack_from("<H", buf, off)[0])
                    if chan is not None and (topics is None or chan.topic in topics):
                        lt = struct.unpack_from("<Q", buf, off + 2)[0]
                        pt = struct.unpack_from("<Q", buf, off + 10)[0]
                        m = _Message()
                        m.channel_id, m.log_time, m.publish_time = chan.id, lt, pt
                        m.data = buf[off + 18:off + length]
                        yield (sid2sch.get(chan.schema_id), chan, m)
                elif opcode == ops["chunk"]:
                    o = off + 24
                    if ops["new"]:
                        o += 4  # uncompressed_crc
                    compression, o = _read_str(buf, o)
                    rlen = struct.unpack_from("<Q", buf, o)[0]
                    o += 8
                    dec = _decompress(buf[o:o + rlen], compression)
                    if dec is not None:
                        yield from handle(dec, 0, len(dec))
                elif opcode == ops["data_end"]:
                    break
            except Exception:
                continue  # 坏记录跳过，不拖垮整体

    yield from handle(data, len(magic), len(data))


def iter_decoded_messages_robust(mcap_path, topics=None, decoder_factories=None):
    """解码迭代：优先走官方 mcap 库（与 iter_decoded_messages 一致，
    正常文件零回归）；get_summary 在截断/未封口文件上崩溃时，
    回退到内置数据区迭代（iter_mcap_messages）+ 同一批解码器。
    产出 (schema, channel, message, ros_msg) 四元组。"""
    factories = decoder_factories or []
    try:
        with open(mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=factories)
            for schema, channel, message, ros_msg in reader.iter_decoded_messages(topics=topics):
                yield schema, channel, message, ros_msg
        return
    except Exception as e:
        logging.warning(f"mcap 库迭代失败 ({e}), 回退内置数据区迭代")
    for schema, chan, msg in iter_mcap_messages(mcap_path, topics=topics):
        decoded = None
        for factory in factories:
            decoder = factory.decoder_for(chan.message_encoding, schema)
            if decoder is not None:
                decoded = decoder(msg.data)
                break
        yield schema, chan, msg, decoded


def scan_mcap_channels(mcap_path) -> list:
    """扫描 mcap 全部 channel：(topic, schema_name, count)。不依赖解码器。
    优先用 mcap 库 summary（秒级）；失败/为空时用内置解析器
    （summary 区 + data section 兜底），不再受 mcap 库版本影响。"""
    # 1) 快速路径：mcap 库 summary
    try:
        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
        if summary is not None and summary.channels:
            cnt = Counter()
            names = {}
            for cid, ch in summary.channels.items():
                t = ch.topic
                cnt[t] = summary.statistics.channel_message_counts.get(cid, 0)
                names[t] = (summary.schemas[ch.schema_id].name
                            if ch.schema_id in summary.schemas else "?")
            return [(t, names[t], cnt[t]) for t in cnt]
    except Exception as e:
        logging.warning(f"mcap 库 summary 解析失败 ({e}), 使用内置解析器")
    # 2) 内置解析
    return _manual_scan(mcap_path)


def setup_logger(log_path: str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_yaml_config(yaml_path: str = "config.yaml") -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pointcloud2_to_xyzi(msg, point_field_datatype_to_np) -> np.ndarray:
    dt = np.dtype({
        "names": [f.name for f in msg.fields],
        "formats": [point_field_datatype_to_np[f.datatype] for f in msg.fields],
        "offsets": [f.offset for f in msg.fields],
        "itemsize": msg.point_step,
    })
    arr = np.frombuffer(bytes(msg.data), dtype=dt, count=msg.width * msg.height)
    inten = arr["intensity"] if "intensity" in arr.dtype.names else np.zeros(len(arr))
    ring = arr["ring"] if "ring" in arr.dtype.names else np.zeros(len(arr))
    xyzi = np.stack([arr["x"], arr["y"], arr["z"], inten, ring], axis=-1).astype(np.float32)
    return xyzi[np.isfinite(xyzi[:, :3]).all(axis=1)]


def msg_stamp_ns(ros_msg, msg) -> int:
    try:
        return ros_msg.header.stamp.sec * 10 ** 9 + ros_msg.header.stamp.nanosec
    except AttributeError:
        pass
    try:
        return ros_msg.timestamp.sec * 10 ** 9 + ros_msg.timestamp.nanosec
    except AttributeError:
        return msg.log_time


def save_T_yaml(path, T, extra=None):
    r = [f"{v:.9f}" for v in np.asarray(T[:3, :3], dtype=np.float64).ravel()]
    t = [f"{v:.9f}" for v in np.asarray(T[:3, 3], dtype=np.float64)]
    with open(path, "w", encoding="utf-8") as f:
        f.write("Rcl: [ " + ", ".join(r[0:3]) + ",\n"
                + "       " + ", ".join(r[3:6]) + ",\n"
                + "       " + ", ".join(r[6:9]) + " ]\n")
        f.write("Pcl: [ " + ", ".join(t) + " ]\n")
        if extra:
            for k, v in extra.items():
                f.write(f"{k}: {v}\n")


def save_point_cloud(pc_np: np.ndarray, out_path: str):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_np[:, 0:3])
    intensity = pc_np[:, 3]
    rgb = np.zeros((len(pc_np), 3), dtype=np.float32)
    rgb[:, 0] = intensity
    rgb[:, 1] = intensity
    rgb[:, 2] = intensity
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(out_path, pcd)


def colorize_by_holes(pc_xyz, holes3d, hole_radius, hole_palette):
    cols = np.full((len(pc_xyz), 3), 0.75)
    if holes3d is not None:
        for k, h in enumerate(holes3d[:4]):
            cols[np.linalg.norm(pc_xyz - h, axis=1) <= 1.3 * hole_radius] = hole_palette[k % 4]
    return cols


def circle_fit_2d(xy):
    A = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    b = (xy ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    return np.array([cx, cy]), float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0)))


def residual_3d(params, lidar_pts, cam_pts):
    from scipy.spatial.transform import Rotation
    R = Rotation.from_rotvec(params[:3]).as_matrix()
    return ((R @ lidar_pts.T).T + params[3:6] - cam_pts).ravel()
