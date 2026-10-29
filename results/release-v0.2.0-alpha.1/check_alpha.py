"""Verify source-alpha records, including the known failed hardware run."""
import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def read(path):
    return json.loads(path.read_text())


def signature(path, complete_only=False):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if complete_only:
        rows = [row for row in rows if row['response']['tokens'] is not None]
    for row in rows:
        assert row['request_sha256'] == hashlib.sha256(json.dumps(row['request'], sort_keys=True).encode()).hexdigest()
    return collections.Counter((row['request_sha256'], row['response']['text'], row['response']['tokens']) for row in rows)


def main():
    status = read(HERE / 'validation-status.json')
    assert status['state'] == 'known_issue' and status['full_regression_passed'] is False
    for name, digest in read(HERE / 'alpha-files.sha256.json').items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest, name
    build = read(HERE / 'build.json')
    assert build['image_id'] == build['worker_image_id'] == status['image_id']
    assert build['no_application_build_cache']
    for name, digest in build['source_files'].items():
        assert hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() == digest, name
    installed = read(HERE / 'installed-image.json')
    modules = PROJECT / 'src/spark1017'
    expected = {'spark1017/spark1017/' + p.relative_to(modules).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in modules.rglob('*.py')}
    actual = {k: v for k, v in installed['source_hashes'].items() if k.startswith('spark1017/spark1017/')}
    assert expected == actual
    gpu = read(HERE / 'engram-cuda.json')
    assert gpu['image_id'] == build['image_id'] and set(gpu['nodes']) == {'head', 'worker'}
    assert all(v['passed'] and v['exact_cases'] == 9 and v['peak_saved_bytes'] >= 450 * 2**20 for v in gpu['nodes'].values())
    diagnostic = HERE / 'tiled-workspace-probe'
    equality = read(diagnostic / 'runtime-equivalence.json')
    assert equality['runtime_source_hashes_equal'] and equality['installed_versions_equal']
    assert equality['clean_image'] == build['image_id']
    baseline = signature(diagnostic / 'bench/tiled-baseline-speed/requests.jsonl')
    assert sum(baseline.values()) == 58
    for profile in ('edit', 'prefill-experimental'):
        assert baseline == signature(diagnostic / 'bench' / ('tiled-' + profile + '-speed') / 'requests.jsonl')
    full = read(diagnostic / 'bench/tiled-edit-fullfile/summary.json')
    assert len(full) == 6 and sum(r['grade']['passed'] for r in full) == 836
    assert all(r['grade']['passed'] == r['grade']['total'] for r in full)
    failure = read(HERE / 'failed-cold-tiled/failure.json')
    assert failure['status'] == 'failed' and failure['completed_requests'] == 55
    failed_records = HERE / 'failed-cold-tiled/bench/baseline-speed/requests.jsonl'
    assert len(failed_records.read_text().splitlines()) == 56
    complete = signature(failed_records, complete_only=True)
    assert sum(complete.values()) == 55 and not (complete - baseline)
    restore = read(HERE / 'restoration.json')
    assert restore['passed'] and restore['original_service_api']['passed'] == 5
    assert restore['original_service_api']['web_status'] == 200
    assert sum(n['original_files_checked'] for n in restore['nodes'].values()) == 50
    assert all(n['original_files_unchanged'] and n['release_containers_absent'] for n in restore['nodes'].values())
    for node in restore['nodes'].values():
        assert all(all(v.values()) for v in node['containers'].values())
    print(json.dumps({'source_alpha_records': 'consistent', 'hardware_qualification': 'known_issue',
                      'full_clean_image_regression_passed': False, 'original_deployment_restored': True,
                      'diagnostic_profiles': 3, 'diagnostic_requests_per_profile': 58,
                      'diagnostic_edit_assertions': 836, 'cold_run_completed_requests': 55}, indent=2))


if __name__ == '__main__':
    main()
