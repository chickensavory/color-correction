from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union
from datetime import date as _date

SOI = b"\xff\xd8"
SOS = b"\xff\xda"

XMP_ID = b"http://ns.adobe.com/xap/1.0/\x00"

DEFAULT_PROCESS_TOOL = "color-correction"

_slug_rx = re.compile(r"[^a-z0-9]+", re.IGNORECASE)


def slugify(stem: str) -> str:
    s = (stem or "").strip().lower()
    s = _slug_rx.sub("_", s).strip("_")
    return s or "image"


def _ensure_ns(prefix: str, uri: str):
    try:
        ET.register_namespace(prefix, uri)
    except Exception:
        pass


_NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
}
_ensure_ns("x", _NS["x"])
_ensure_ns("rdf", _NS["rdf"])
_ensure_ns("dc", _NS["dc"])


def _minimal_xmp_packet_root() -> ET.Element:
    xmpmeta = ET.Element(f"{{{_NS['x']}}}xmpmeta")
    rdf = ET.SubElement(xmpmeta, f"{{{_NS['rdf']}}}RDF")
    ET.SubElement(rdf, f"{{{_NS['rdf']}}}Description")
    return xmpmeta


def _get_or_create_rdf_description(xmpmeta_root: ET.Element) -> ET.Element:
    rdf = None
    for el in xmpmeta_root.iter():
        if el.tag == f"{{{_NS['rdf']}}}RDF":
            rdf = el
            break
    if rdf is None:
        rdf = ET.SubElement(xmpmeta_root, f"{{{_NS['rdf']}}}RDF")

    for el in list(rdf):
        if el.tag == f"{{{_NS['rdf']}}}Description":
            return el

    return ET.SubElement(rdf, f"{{{_NS['rdf']}}}Description")


def _find_child(parent: ET.Element, ns: str, name: str) -> Optional[ET.Element]:
    tag = f"{{{ns}}}{name}"
    for ch in list(parent):
        if ch.tag == tag:
            return ch
    return None


def _ensure_dc_subject_keyword(desc: ET.Element, keyword: str) -> bool:
    changed = False

    dc_subject = _find_child(desc, _NS["dc"], "subject")
    if dc_subject is None:
        dc_subject = ET.SubElement(desc, f"{{{_NS['dc']}}}subject")
        changed = True

    bag = None
    for ch in list(dc_subject):
        if ch.tag == f"{{{_NS['rdf']}}}Bag":
            bag = ch
            break
    if bag is None:
        bag = ET.SubElement(dc_subject, f"{{{_NS['rdf']}}}Bag")
        changed = True

    for li in list(bag):
        if li.tag == f"{{{_NS['rdf']}}}li" and (li.text or "").strip() == keyword:
            return changed

    li = ET.SubElement(bag, f"{{{_NS['rdf']}}}li")
    li.text = keyword
    return True


def _ensure_dc_description_xdefault(desc: ET.Element, text: str) -> bool:
    changed = False

    dc_desc = _find_child(desc, _NS["dc"], "description")
    if dc_desc is None:
        dc_desc = ET.SubElement(desc, f"{{{_NS['dc']}}}description")
        changed = True

    alt = None
    for ch in list(dc_desc):
        if ch.tag == f"{{{_NS['rdf']}}}Alt":
            alt = ch
            break
    if alt is None:
        alt = ET.SubElement(dc_desc, f"{{{_NS['rdf']}}}Alt")
        changed = True

    xml_lang_key = "{http://www.w3.org/XML/1998/namespace}lang"
    for li in list(alt):
        if li.tag != f"{{{_NS['rdf']}}}li":
            continue
        lang = (li.attrib.get(xml_lang_key, "") or "").strip().lower()
        if lang == "x-default":
            if (li.text or "") == text:
                return changed
            li.text = text
            return True

    li = ET.SubElement(alt, f"{{{_NS['rdf']}}}li")
    li.set(xml_lang_key, "x-default")
    li.text = text
    return True


def _decode_xml_bytes(xmp_xml_bytes: bytes) -> Optional[str]:
    if not xmp_xml_bytes:
        return None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return xmp_xml_bytes.decode(enc, errors="replace")
        except Exception:
            continue
    return None


def _parse_or_create_xmpmeta_root(xmp_xml_bytes: Optional[bytes]) -> ET.Element:
    if xmp_xml_bytes:
        txt = _decode_xml_bytes(xmp_xml_bytes)
        if txt:
            try:
                root = ET.fromstring(txt)
                if root.tag.endswith("xmpmeta"):
                    return root
                for el in root.iter():
                    if el.tag.endswith("xmpmeta"):
                        return el
            except Exception:
                pass
    return _minimal_xmp_packet_root()


def _serialize_xmpmeta(root: ET.Element) -> bytes:
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _sidecar_path_for_image(image_path: Path) -> Path:
    raw_like = image_path.suffix.lower() in (
        ".nef",
        ".arw",
        ".cr2",
        ".cr3",
        ".dng",
    )
    if raw_like:
        return image_path.with_suffix(".xmp")
    return image_path.with_name(image_path.name + ".xmp")


def _extract_xmp_from_jpeg_bytes(data: bytes) -> Optional[bytes]:
    if not data.startswith(SOI):
        return None

    i = 2
    n = len(data)

    while i < n:
        if data[i] != 0xFF:
            i += 1
            continue

        j = i
        while j < n and data[j] == 0xFF:
            j += 1
        if j >= n:
            break

        marker = data[j]
        i = j + 1

        if marker in (0xDA, 0xD9):
            break

        if i + 2 > n:
            break

        seglen = struct.unpack(">H", data[i : i + 2])[0]
        seg_end = i + seglen
        payload_start = i + 2

        if marker == 0xE1:
            payload = data[payload_start:seg_end]
            if payload.startswith(XMP_ID):
                return payload[len(XMP_ID) :]

        i = seg_end

    return None


def _iter_jpeg_segments_with_bounds(data: bytes):
    if not data.startswith(SOI):
        return

    i = 2
    n = len(data)

    while i < n:
        if data[i] != 0xFF:
            i += 1
            continue

        seg_start = i
        j = i
        while j < n and data[j] == 0xFF:
            j += 1
        if j >= n:
            break

        marker = data[j]
        i = j + 1

        if marker in (0xD9, 0xDA):
            yield (marker, seg_start, seg_start + 2, seg_start + 2)
            break

        if i + 2 > n:
            break

        seglen = struct.unpack(">H", data[i : i + 2])[0]
        payload_start = i + 2
        seg_end = i + seglen

        yield (marker, seg_start, seg_end, payload_start)
        i = seg_end


def _build_app1_xmp_segment(xmp_packet: bytes) -> bytes:
    payload = XMP_ID + xmp_packet
    return b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload


def _make_updated_xmp_packet(
    existing_xmp_packet: Optional[bytes], *, tool: str, processed_date: str
) -> bytes:
    keyword = f"ProcessedWith:{tool}"
    desc_text = f"Processed by {tool} on {processed_date}"

    root = _parse_or_create_xmpmeta_root(existing_xmp_packet)
    rdf_desc = _get_or_create_rdf_description(root)

    _ensure_dc_subject_keyword(rdf_desc, keyword)
    _ensure_dc_description_xdefault(rdf_desc, desc_text)

    return _serialize_xmpmeta(root)


def write_processed_xmp_sidecar(
    image_path: Union[str, Path],
    *,
    tool: str = DEFAULT_PROCESS_TOOL,
    processed_date: Optional[str] = None,
) -> bool:
    p = Path(image_path)
    processed_date = processed_date or _date.today().isoformat()

    sidecar = _sidecar_path_for_image(p)

    existing = None
    if sidecar.exists():
        try:
            existing = sidecar.read_bytes()
        except Exception:
            existing = None

    root = _parse_or_create_xmpmeta_root(existing)
    rdf_desc = _get_or_create_rdf_description(root)

    changed = False
    changed = _ensure_dc_subject_keyword(rdf_desc, f"ProcessedWith:{tool}") or changed
    changed = (
        _ensure_dc_description_xdefault(
            rdf_desc, f"Processed by {tool} on {processed_date}"
        )
        or changed
    )

    if not changed and sidecar.exists():
        return True

    try:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_bytes(_serialize_xmpmeta(root))
        return True
    except Exception:
        return False


def write_processed_xmp_embed_jpeg(
    jpeg_path: Union[str, Path],
    *,
    tool: str = DEFAULT_PROCESS_TOOL,
    processed_date: Optional[str] = None,
) -> bool:
    p = Path(jpeg_path)
    if p.suffix.lower() not in (".jpg", ".jpeg"):
        return False

    processed_date = processed_date or _date.today().isoformat()

    try:
        data = p.read_bytes()
    except Exception:
        return False

    if not data.startswith(SOI):
        return False

    existing_xmp = _extract_xmp_from_jpeg_bytes(data)
    new_xmp_packet = _make_updated_xmp_packet(
        existing_xmp, tool=tool, processed_date=processed_date
    )
    new_seg = _build_app1_xmp_segment(new_xmp_packet)

    replace_start = None
    replace_end = None
    insert_at = 2

    for marker, seg_start, seg_end, payload_start in _iter_jpeg_segments_with_bounds(
        data
    ):
        if marker == 0xDA:
            break

        if marker == 0xE1:
            payload = data[payload_start:seg_end]
            if payload.startswith(XMP_ID):
                replace_start = seg_start
                replace_end = seg_end
                break

        if seg_start == insert_at and marker == 0xE0:
            insert_at = seg_end

    if replace_start is not None and replace_end is not None:
        new_data = data[:replace_start] + new_seg + data[replace_end:]
    else:
        new_data = data[:insert_at] + new_seg + data[insert_at:]

    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        tmp.write_bytes(new_data)
        tmp.replace(p)
        return True
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False


def write_processed_tags(
    image_path: Union[str, Path],
    *,
    tool: str = DEFAULT_PROCESS_TOOL,
    processed_date: Optional[str] = None,
) -> bool:
    p = Path(image_path)
    processed_date = processed_date or _date.today().isoformat()

    if p.suffix.lower() in (".jpg", ".jpeg"):
        return write_processed_xmp_embed_jpeg(
            p, tool=tool, processed_date=processed_date
        )

    return write_processed_xmp_sidecar(p, tool=tool, processed_date=processed_date)
