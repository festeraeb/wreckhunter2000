#!/usr/bin/env python3
# core_shared.py — shared helpers (varstruct, CRC, magic scan, progress)

import struct

MAGIC_REC_HDR = 0xB7E9DA86  # header magic (little-endian value)
MAGIC_REC_TRL = 0xD9264B7C  # trailer magic (little-endian value)

_progress_hook = None
def set_progress_hook(fn):  # fn(percent_float, message)
    global _progress_hook; _progress_hook = fn

def _emit(pct, msg):
    if _progress_hook:
        try: _progress_hook(float(pct), str(msg))
        except Exception: pass

def _crc32_custom(data: bytes) -> int:
    poly=0x04C11DB7; crc=0
    for b in data:
        crc ^= (int(b)<<24) & 0xFFFFFFFF
        for _ in range(8):
            if crc & 0x80000000: crc=((crc<<1)^poly)&0xFFFFFFFF
            else: crc=(crc<<1)&0xFFFFFFFF
    # bit-reverse
    rev=0; tmp=crc
    for _ in range(32):
        rev=(rev<<1)|(tmp&1); tmp>>=1
    return (rev ^ 0xFFFFFFFF) & 0xFFFFFFFF

def _read_varuint_from(mm,pos,limit):
    res=0; shift=0
    while pos<limit:
        b=mm[pos]; pos+=1; res|=(b&0x7F)<<shift
        if not(b&0x80): return res,pos
        shift+=7
        if shift>35: break
    raise ValueError('VarUInt overflow')

def _read_varint_from(buf,pos,limit):
    res=0; shift=0; i=pos
    while i<limit:
        b=buf[i-pos]; i+=1
        res|=(b&0x7F)<<shift
        if not(b&0x80):
            u=res
            v=(u>>1)^(-(u&1))  # zigzag
            return v,i
        shift+=7
        if shift>35: break
    raise ValueError('VarInt overflow')

def _parse_varstruct(mm,pos,limit,crc_mode='warn'):
    start = pos
    n, pos = _read_varuint_from(mm, pos, limit)
    if n < 0 or n > 10000:
        raise ValueError(f'Unreasonable field count: {n}')
    fields = {}
    for _ in range(n):
        key, pos = _read_varuint_from(mm, pos, limit)
        fn = key >> 3
        lc = key & 7
        if lc == 7:
            vlen, pos = _read_varuint_from(mm, pos, limit)
            if vlen < 0 or vlen > (limit - pos): raise ValueError('Varstruct value exceeds file size')
        else:
            vlen = lc
        endv = pos + vlen
        if endv > limit: raise ValueError('Varstruct value exceeds file size')
        fields[fn] = bytes(mm[pos:endv])
        pos = endv
    if pos + 4 > limit: raise ValueError('Truncated before CRC')
    crc_read = struct.unpack('>I', mm[pos:pos+4])[0]; pos += 4
    data = bytes(mm[start:pos-4]); crc_calc = _crc32_custom(data)
    if crc_mode == 'strict' and crc_calc != crc_read:
        raise ValueError(f'CRC mismatch: calc=0x{crc_calc:08X} read=0x{crc_read:08X}')
    elif crc_mode == 'warn' and crc_calc != crc_read:
        import logging; logging.warning('CRC mismatch at 0x%X', start)
    return fields, pos

def _mapunit_to_deg(x:int)->float: return x*(360.0/float(1<<32))

def find_magic(mm, magic_bytes, start, end, chunk=32*1024*1024):
    """Chunked .find with progress updates."""
    start = max(0, start); end = min(len(mm), end)
    if end <= start: return -1
    if end - start <= chunk:
        return mm.find(magic_bytes, start, end)
    s = start
    total = end - start
    while s < end:
        e = min(end, s + chunk)
        idx = mm.find(magic_bytes, s, e)
        done = e - start
        _emit((done/total)*100.0, f"Scanning… {done//1024//1024} / {total//1024//1024} MB")
        if idx != -1: return idx
        s = e
    return -1
