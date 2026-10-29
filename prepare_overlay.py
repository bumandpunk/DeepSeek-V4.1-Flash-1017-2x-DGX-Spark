"""Create a separate overlay; never edit the input checkout or a serving mount.

Accepts the validated TensorFold source snapshot plus optional deployment's
engine.py/multi.py safety overlay. Exact input/output hashes are recorded.
"""
import argparse
import hashlib
import json
import pathlib
import shutil

def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('unsupported source revision: hook does not match exactly once')
    return text.replace(old, new, 1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cuda-source', type=pathlib.Path, required=True)
    p.add_argument('--safety-overlay', type=pathlib.Path)
    p.add_argument('--output', type=pathlib.Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    native = pathlib.Path(__file__).resolve().parent / 'native'
    originals = {}
    for name in ['model.py', 'engine.py', 'multi.py']:
        source = a.cuda_source / name
        if a.safety_overlay and name != 'model.py' and (a.safety_overlay / name).exists():
            source = a.safety_overlay / name
        originals[name] = source.read_text()
    supported = {
        'model.py': 'a5a2d643970856e662a04eabe665994696084d80bbfcd30b9b253ebe18abf843',
        'engine.py': '62c1409215d76d190bde59ffa21f27c9b647416d45051f4752b62195a3dc2ad8',
        'multi.py': '1e045c266a1b6b1f8672f60c10176a8caefe6abece26fa4230ccfe61498673bc',
    }
    for name, value in originals.items():
        if hashlib.sha256(value.encode()).hexdigest() != supported[name]:
            raise ValueError('unsupported input ' + name + '; use the pinned source and compatibility patch')
    model = originals['model.py']
    model = replace_once(model, 'name="tf_ds_engram_io_v7"', 'name="spark1017_engram_io_v1"')
    (a.output / 'model.py').write_text(model)
    engine = originals['engine.py']
    engine = replace_once(engine, '        self._memory_ceiling()\n        if rank == 0:',
        '        self._memory_ceiling()\n'
        '        from spark1017.storage import initialize as spark1017_initialize\n'
        '        spark1017_initialize(self)\n'
        '        if rank == 0:')
    (a.output / 'engine.py').write_text(engine)
    multi = originals['multi.py']
    multi = replace_once(multi, '                for s, r, cf in zip(drafting, rows, conf_rows):',
        '                if os.environ.get("SPARK1017_ADAPTIVE_DEPTH", "0") == "1":\n'
        '                    from spark1017.integration import choose\n'
        '                    even = choose(self, live, drafting, conf_rows, [want[s.sid] for s in drafting], ROUND_MS, DRAFT_MS)\n'
        '                for s, r, cf in zip(drafting, rows, conf_rows):')
    multi = replace_once(multi, '                        if CONF_LOG and drafts:',
        '                        if hasattr(self, "_spark1017_depth"):\n'
        '                            from spark1017.integration import observe\n'
        '                            observe(self, confs.get(s.sid, []), len(drafts), a)\n'
        '                        if CONF_LOG and drafts:')
    multi = replace_once(multi, '            self.round_log.append((len(live), r0, time.perf_counter() - t0, tokens))',
        '            round_elapsed = time.perf_counter() - t0\n'
        '            if os.environ.get("SPARK1017_PROFILE"):\n'
        '                from spark1017.integration import record_round\n'
        '                record_round(self, r0, len(live), round_elapsed)\n'
        '            self.round_log.append((len(live), r0, round_elapsed, tokens))')
    multi = replace_once(multi, '        e = self.e\n        t0 = time.perf_counter()\n        self._step(True)',
        '        e = self.e\n'
        '        if os.environ.get("SPARK1017_KERNEL_TRACE"):\n'
        '            from spark1017.profiling import begin\n'
        '            begin(self)\n'
        '        t0 = time.perf_counter()\n        self._step(True)')
    multi = replace_once(multi, '            self.rounds += 1',
        '            self.rounds += 1\n'
        '            if os.environ.get("SPARK1017_KERNEL_TRACE"):\n'
        '                from spark1017.profiling import end\n'
        '                end(self)')
    multi = replace_once(multi, '            drafting = [s for s in live if want[s.sid] > 0]',
        '            drafting = [s for s in live if want[s.sid] > 0]\n'
        '            lookup = {}\n'
        '            if os.environ.get("SPARK1017_LOOKUP") == "1":\n'
        '                from spark1017.lookup import propose as lookup_propose\n'
        '                lookup = lookup_propose(self, live, ROUND_MS)\n'
        '                drafting = [s for s in drafting if s.sid not in lookup]')
    multi = replace_once(multi, '            proposed, confs = {}, {}',
        '            proposed, confs = dict(lookup), {}')
    multi = replace_once(multi, '                        if hasattr(self, "_spark1017_depth"):',
        '                        if os.environ.get("SPARK1017_LOOKUP") == "1":\n'
        '                            from spark1017.lookup import observe as lookup_observe\n'
        '                            lookup_observe(self, s, len(drafts), a, ROUND_MS, DRAFT_MS)\n'
        '                        if hasattr(self, "_spark1017_depth") and s.sid not in lookup:')
    multi = replace_once(multi, '                        if CONF_LOG and drafts:',
        '                        if CONF_LOG and drafts and s.sid not in lookup:')
    (a.output / 'multi.py').write_text(multi)
    for name in ['engram_io.cpp', 'tiered_store.hpp']:
        shutil.copy2(native / name, a.output / name)
    digest = lambda b: hashlib.sha256(b).hexdigest()
    manifest = {'base_sha256': {n: digest(v.encode()) for n, v in originals.items()},
                'overlay_sha256': {f.name: digest(f.read_bytes()) for f in a.output.iterdir() if f.is_file()}}
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__': main()
