"""Decode all orbit movies and write the Finnish video index.

Run .venv/Scripts/python.exe tools/validation/verify_model_orbits.py [--watch] while rendering.
"""

# Allow both direct script execution and python -m from the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pathlib import Path
import argparse
import hashlib
import json
import time

import cv2
import numpy as np
from pipeline.paths import PROJECT_ROOT
from pipeline.source_fingerprints import source_hash_matches

ROOT = PROJECT_ROOT
OUT = ROOT / 'Animaatiot_4K'
MODELS = json.loads((ROOT / 'LaserScans/comparison_results/blender/data/manifest.json').read_text(encoding='utf-8'))['models']


def verify(path, model, kind):
    report = json.loads(path.read_text(encoding='utf-8'))
    assert report['render_complete'] and report['model'] == model['model'] and report['kind'] == kind
    assert report['framing_verification'] == 'passed'
    assert report['all_geometry_inside_frame'] and report['source_geometry_unchanged']
    assert report['camera_always_faces_origin'] and report['elevation_degrees'] == 40
    assert report['fingerprint']['source'] == hashlib.sha256((ROOT / report['source']).read_bytes()).hexdigest()
    current_script = hashlib.sha256((ROOT / 'tools/visualization/create_model_orbits.py').read_bytes()).hexdigest()
    saved_script = report['fingerprint']['script']
    legacy_script = '55222b237fe1ad325c8d9f3644c65e5f7cc4ecfb16ee75fb8df2aae62b1c54d3'
    legacy = () if kind == 'error' and model['object'] == 'Sini' else (legacy_script,)
    assert source_hash_matches(saved_script, current_script,
                               'tools/visualization/create_model_orbits.py', legacy_hashes=legacy)
    if kind == 'error':
        assert report['statistics'] == model['forward_photo_to_scan']
        assert report['reference_scan'] == model['scan_frame']
    movie = OUT / report['video']
    assert (OUT / report['blend']).is_file() and path.with_suffix('.png').is_file()
    capture = cv2.VideoCapture(str(movie))
    assert capture.isOpened(), movie
    fps = capture.get(cv2.CAP_PROP_FPS)
    codec_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    codec = ''.join(chr((codec_value >> (8 * i)) & 255) for i in range(4))
    count = 0
    samples = {}
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        assert frame.shape[:2] == (2160, 3840), (movie, count, frame.shape)
        if count in (0, 90, 180, 270, 359):
            samples[count] = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        count += 1
    capture.release()
    assert count == 360 and fps == 30 and codec == 'h264', (movie, count, fps, codec)
    motion = max(float(np.mean(np.abs(samples[0].astype(float) - samples[k]))) for k in (90, 180, 270))
    seam = float(np.mean(np.abs(samples[0].astype(float) - samples[359])))
    assert motion > .2, (movie, motion)
    assert seam < motion, (movie, seam, motion)
    proof = OUT / 'tarkistuskuvat'
    proof.mkdir(exist_ok=True)
    cv2.imwrite(str(proof / (path.stem + '.jpg')),
                np.vstack([np.hstack([samples[0], samples[90]]), np.hstack([samples[180], samples[270]])]))
    result = {'model': model['model'], 'kind': kind, 'status': 'passed', 'video': report['video'],
              'decoded_frames': count, 'fps': fps, 'resolution': [3840, 2160], 'codec': codec,
              'size_bytes': movie.stat().st_size, 'sha256': hashlib.sha256(movie.read_bytes()).hexdigest(),
              'quarter_turn_pixel_difference': motion, 'loop_seam_pixel_difference': seam,
              'minimum_frame_edge_margin_fraction': report['minimum_frame_edge_margin_fraction']}
    print('VERIFIED', model['model'], kind, flush=True)
    return result


def index(results):
    lines = [
        '# Kaikkien mallien 4K-animaatiot', '',
        '22 videota: 11 alkuperäistä dense-pintamallia ja 11 laserreferenssiin verrattua pistepilveä. '
        'Kaikki videot ovat 3840 × 2160 pikseliä, 30 fps ja 12 sekuntia. '
        'Kamera kiertää 360 astetta ja katsoo origoon 40 astetta alaspäin.', '',
        'Kunkin mallin kameraetäisyys ja rajaus pysyvät samoina koko kierroksen ajan. '
        'Rajaus on sovitettu erikseen jokaiselle mallille: eri videoiden näennäinen koko ei ole yhteinen mittakaava. '
        'Kaikki näkyvän mallin pisteet ja irralliset osat ovat mukana. '
        'Alkuperäisiä malleja, kohdistuksia tai virhearvoja ei muutettu.', '',
        '| Malli | Pintamalli (MP4) | Poikkeamapistepilvi (MP4) |',
        '| --- | --- | --- |',
    ]
    for model in MODELS:
        name = model['model']
        lines.append(f'| {name} | [Avaa](../videot/{name}_pinta_4K.mp4) | [Avaa](../videot/{name}_poikkeamat_4K.mp4) |')
    lines += ['',
        'Piiristä on kaksi erillistä single-versiota (`single_frame_1` ja `single_frame_2`). '
        'Harmun poikkeamavertailu käyttää valokuvattua puolta vastaavaa skannausta 0003.', '',
        'Poikkeamavideoiden yhteinen väriasteikko näyttää etäisyyden lähimpään laserpisteeseen millimetreinä. '
        'Mediaani, RMSE ja P95 ovat kyseisen mallin omat luvut. Kyseessä eivät ole kalibroidut ±mittausepävarmuusrajat.', '',
        'MP4-videot ovat projektin `videot`-kansiossa. Tämän kansion `pinnat`- ja `poikkeamat`-alikansioissa ovat samanniminen animoitu `.blend`, 4K-esikatselukuva `.png` ja tekninen '
        'tarkistusraportti `.json`. Blenderissä animaation voi käynnistää välilyönnillä. '
        'Alkuperäinen pintamalli näkyy pinnat-kansion tiedostoissa; poikkeamat-kansion tiedostoissa näkyy väritetty pistepilvi.', '',
        '[Videotarkistusten yhteenveto](verification.json). Tarkistus purkaa jokaisen videon kaikki 360 kuvaa, '
        'varmistaa resoluution, kuvataajuuden ja kierron sekä tarkistaa lähdetiedostojen muuttumattomuuden. '
        'Renderöintivaihe tarkistaa lisäksi koko kameraradan rajauksen ja kameran katselusuunnan.', '',
        'Sinin poikkeamavideot on korvattu 6.11.2026: pitkä ja lyhyt syvennys vastaavat nyt laserreferenssiä. Ennen jäykkää kohdistusta käytetään dokumentoitua koordinaatiston kätisyyden korjausta. Alkuperäisen akselisuuntaeron syntypaikka on edelleen selvittämättä.', '',
        'Uudelleenajo projektin juuressa:', '', '```powershell',
        r'& "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background --threads 4 --python-exit-code 1 --python tools/visualization/create_model_orbits.py -- --kind surface',
        r'& "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background --threads 4 --python-exit-code 1 --python tools/visualization/create_model_orbits.py -- --kind error',
        '.venv/Scripts/python.exe tools/validation/verify_model_orbits.py', '```', '',
    ]
    (OUT / 'LUE_MINUT.md').write_text('\n'.join(lines), encoding='utf-8')
    for folder, suffix in [('pinnat', 'pinta'), ('poikkeamat', 'poikkeamat')]:
        thumbs = []
        for model in MODELS:
            pic = cv2.imread(str(OUT / folder / (model['model'] + '_' + suffix + '_4K.png')))
            assert pic is not None
            thumbs.append(cv2.resize(pic, (640, 360), interpolation=cv2.INTER_AREA))
        while len(thumbs) % 3:
            thumbs.append(np.zeros_like(thumbs[0]))
        sheet = np.vstack([np.hstack(thumbs[i:i + 3]) for i in range(0, len(thumbs), 3)])
        cv2.imwrite(str(OUT / (folder + '_yhteiskuva.jpg')), sheet)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    expected = [(model, kind, OUT / folder / (model['model'] + '_' + suffix + '_4K.json'))
                for model in MODELS for kind, folder, suffix in
                [('surface', 'pinnat', 'pinta'), ('error', 'poikkeamat', 'poikkeamat')]]
    results = {}
    while len(results) < len(expected):
        for model, kind, path in expected:
            key = (model['model'], kind)
            if key in results or not path.exists():
                continue
            if not json.loads(path.read_text(encoding='utf-8')).get('render_complete'):
                continue
            results[key] = verify(path, model, kind)
            report = {'status': 'complete_pass' if len(results) == len(expected) else 'in_progress',
                      'expected_videos': len(expected), 'verified_videos': len(results),
                      'results': list(results.values())}
            (OUT / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if len(results) == len(expected):
            break
        if not args.watch:
            raise RuntimeError(f'Only {len(results)}/{len(expected)} movies complete')
        time.sleep(10)
    index(results)
    print('ALL_22_VIDEOS_VERIFIED', flush=True)


if __name__ == "__main__":
    main()
