"""Tests for signa/marker_pdf.py (print-ready ArUco marker sheets).

Needs real opencv-contrib (like test_integration_opencv.py); skipped when it
is not installed, CI installs it. marker_pdf.py is loaded standalone — the
plugin package __init__ imports WebODM, which is absent in CI.
"""

import importlib.util
import os
import zlib

import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("cv2.aruco")

HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.join(HERE, "..", "signa", "marker_pdf.py")


def _load():
    spec = importlib.util.spec_from_file_location("signa_marker_pdf", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mp = _load()


def _detect_ids(page_bgr, adict):
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(adict, cv2.aruco.DetectorParameters())
    _corners, ids, _ = detector.detectMarkers(gray)
    return [] if ids is None else [int(i) for i in ids.ravel()]


@pytest.mark.parametrize("aid", ["none", "cross", "cross_halo", "dot_ring"])
@pytest.mark.parametrize("gray", [False, True])
def test_every_aid_and_variant_stays_detectable(aid, gray):
    """The aiming aid / gray variant must never break detection — even on the
    smallest offered page, where the aid is largest relative to the modules."""
    adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    page = mp._render_page(cv2, cv2.aruco, adict, 7, "a6", gray, aid, 1)
    assert _detect_ids(page, adict) == [7]


def test_marker_physical_size_on_page():
    """The rendered marker must keep a one-module quiet zone: side(px) at DPI
    equals page-width * modules/(modules+2), rounded to whole modules."""
    adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    page = mp._render_page(cv2, cv2.aruco, adict, 0, "a4", False, "none", 1)
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    dark = (gray < 128).any(axis=1).nonzero()[0]  # rows containing marker/label
    page_w = mp._px(210)
    expected = page_w * 6 // 8
    expected -= expected % 6
    # column extent of the black border row at the marker's vertical center
    mid = (dark[0] + dark[0] + expected) // 2
    cols = (gray[mid] < 128).nonzero()[0]
    assert cols[-1] - cols[0] + 1 == expected


def test_build_pdf_structure_and_page_count():
    pdf, err = mp.build_marker_pdf(1, 0, 2, page="a5", gray=False, aid="cross")
    assert err is None
    assert pdf.startswith(b"%PDF-1.4")
    assert pdf.rstrip().endswith(b"%%EOF")
    assert pdf.count(b"/Type /Page ") == 3       # 3 pages...
    assert pdf.count(b"/Type /Pages ") == 1      # ...one page tree
    assert b"/Count 3" in pdf
    # A5 MediaBox in points (148x210 mm)
    assert b"/MediaBox [0 0 419.53 595.28]" in pdf


def test_pdf_image_roundtrip_detects_marker():
    """Decode the first embedded Flate image back to pixels and detect."""
    import numpy as np
    pdf, err = mp.build_marker_pdf(1, 5, 5, page="a6", gray=True, aid="dot_ring")
    assert err is None
    head_at = pdf.index(b"/Subtype /Image")
    stream_at = pdf.index(b"stream\n", head_at) + len(b"stream\n")
    end_at = pdf.index(b"\nendstream", stream_at)
    raw = zlib.decompress(pdf[stream_at:end_at])
    w, h = mp._px(105), mp._px(148)
    rgb = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
    adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    assert _detect_ids(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), adict) == [5]


def test_capacity_is_rechecked_against_the_real_dictionary():
    pdf, err = mp.build_marker_pdf(0, 48, 52, page="a4")  # DICT_4X4_50
    assert pdf is None
    assert err == mp._ERR_CAPACITY


def test_custom_3x3_dictionary_builds():
    pdf, err = mp.build_marker_pdf(99, 0, 1, page="a5")
    assert err is None
    assert pdf.startswith(b"%PDF-1.4")


def test_error_strings_are_catalog_msgids():
    """The module's error strings are translated via _(error) in plugin.py, so
    they must stay in sync with the de catalog (mirrors test_translations)."""
    import importlib.util as iu
    spec = iu.spec_from_file_location(
        "cm", os.path.join(HERE, "..", "scripts", "compile_messages.py"))
    cm = iu.module_from_spec(spec)
    spec.loader.exec_module(cm)
    po = cm.parse_po(os.path.join(HERE, "..", "signa", "locale", "de",
                                  "LC_MESSAGES", "django.po"))
    for s in (mp._ERR_NO_CV2, mp._ERR_CAPACITY, mp._ERR_SELF_CHECK):
        assert s in po, "marker_pdf error not in de catalog: {!r}".format(s)


@pytest.mark.parametrize('gray', [False, True])
@pytest.mark.parametrize('marker_id', range(6))
@pytest.mark.parametrize('aid', ['none', 'cross', 'cross_halo', 'dot_ring'])
def test_six_marker_dictionary_print_and_worker_agree(tmp_path, marker_id, aid, gray):
    import inspect
    from signa_core import make_dictionary
    adict = make_dictionary(100)
    assert adict.markerSize == 3 and adict.bytesList.shape[0] == 6
    page = mp._render_page(cv2, cv2.aruco, adict, marker_id, 'a6', gray, aid, 100)
    for turns in range(4):
        import numpy as np
        assert _detect_ids(np.rot90(page, turns).copy(), adict) == [marker_id]
    path = tmp_path / 'marker.png'
    cv2.imwrite(str(path), page)
    spec = importlib.util.spec_from_file_location('worker_detector', os.path.join(HERE, '..', 'signa', 'gcp_detect.py'))
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    namespace = {}
    exec(inspect.getsource(worker.detect_gcps), namespace)
    result = namespace['detect_gcps']([str(path)], '{} 1 2 3'.format(marker_id), 28191)
    assert 'error' not in result, result
    assert result['output']['gcp_list'].count('marker.png') == 1


def _pdf_images(pdf):
    import re
    import numpy as np
    for match in re.finditer(rb'/Subtype /Image /Width (\d+) /Height (\d+)[^\n]*\nstream\n', pdf):
        end = pdf.index(b'\nendstream', match.end())
        raw = zlib.decompress(pdf[match.end():end])
        rgb = np.frombuffer(raw, np.uint8).reshape(int(match[2]), int(match[1]), 3)
        yield cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def test_a3_tiled_pdf_reassembles_without_scaling_or_missing_pixels():
    import numpy as np
    from signa_core import make_dictionary, markers
    adict = make_dictionary(100)
    expected = mp._render_page(cv2, cv2.aruco, adict, 5, 'a3', False, 'cross', 100)
    pdf, error = mp.build_marker_pdf(100, 5, 5, page='a3', tile_a4=True)
    assert error is None
    assert b'/MediaBox [0 0 595.28 841.89]' in pdf
    images = list(_pdf_images(pdf))
    cols, rows = markers.tile_layout('a3')
    assert len(images) == cols * rows == 4
    margin = mp._px(markers.TILE_MARGIN_MM)
    overlap = mp._px(markers.TILE_OVERLAP_MM)
    cw, ch = mp._px(210) - 2 * margin, mp._px(297) - 2 * margin
    assembled = np.zeros_like(expected)
    for i, tile in enumerate(images):
        x, y = (i % cols) * (cw - overlap), (i // cols) * (ch - overlap)
        h, w = min(ch, expected.shape[0] - y), min(cw, expected.shape[1] - x)
        assembled[y:y+h, x:x+w] = tile[margin:margin+h, margin:margin+w]
    assert np.array_equal(assembled, expected)
    assert _detect_ids(assembled, adict) == [5]


@pytest.mark.parametrize('page,box', [('a1', b'1683.78 2383.94'), ('a0', b'2383.94 3370.39')])
def test_large_format_pdf_dimensions_and_detection(page, box):
    from signa_core import make_dictionary
    pdf, error = mp.build_marker_pdf(100, 0, 0, page=page)
    assert error is None
    assert b'/MediaBox [0 0 ' + box + b']' in pdf
    rendered = next(_pdf_images(pdf))
    assert _detect_ids(rendered, make_dictionary(100)) == [0]
