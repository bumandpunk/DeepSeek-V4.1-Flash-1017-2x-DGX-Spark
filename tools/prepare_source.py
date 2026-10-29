"""Apply the measured compatibility patch and generate the exact engine overlay.

Input must be a clean pinned TensorFold checkout. This command modifies that
checkout after validating the original hashes; use a dedicated build directory.
"""
import argparse, hashlib, json, pathlib, shutil, subprocess
p=argparse.ArgumentParser();p.add_argument("checkout",type=pathlib.Path);a=p.parse_args()
r=pathlib.Path(__file__).resolve().parents[1]; c=a.checkout.resolve()
m=json.loads((r/"compat/manifest.json").read_text())
sha=subprocess.check_output(["git","-C",str(c),"rev-parse","HEAD"],text=True).strip()
if sha!=m["upstream_commit"]:raise SystemExit("Unsupported upstream commit: "+sha)
for rel,v in m["files"].items():
 if hashlib.sha256((c/rel).read_bytes()).hexdigest()!=v["before"]:raise SystemExit("Modified input: "+rel)
subprocess.run(["git","-C",str(c),"apply","--check",str(r/"compat/deployment-safety.patch")],check=True)
subprocess.run(["git","-C",str(c),"apply",str(r/"compat/deployment-safety.patch")],check=True)
out=c/"spark1017-overlay"; cuda=c/"src/tensorfold/families/deepseek_v41/cuda"
subprocess.run(["python3",str(r/"prepare_overlay.py"),"--cuda-source",str(cuda),"--output",str(out)],check=True)
round2=c/"spark1017-round2-overlay"
subprocess.run(["python3",str(r/"prepare_round2_overlay.py"),"--v1-overlay",str(out),
                "--cuda-source",str(cuda),"--output",str(round2)],check=True)
out=round2
for f in out.iterdir():
 if f.suffix in (".py",".cpp",".hpp"):shutil.copy2(f,cuda/f.name)
subprocess.run(["python3",str(r/"tools/prepare_memory_budget.py"),str(cuda)],check=True)
subprocess.run(["python3",str(r/"tools/prepare_engram.py"),str(cuda)],check=True)
exl3=c/"src/tensorfold/cuda/exl3"
experts=c/"spark1017-experts-overlay"
subprocess.run(["python3",str(r/"prepare_experts.py"),"--source",str(exl3/"experts.py"),"--output",str(experts/"experts.py")],check=True)
shutil.copy2(experts/"experts.py",exl3/"experts.py")
subprocess.run(["python3",str(r/"prepare_transport.py"),"--source",str(c/"src/tensorfold/cuda")],check=True)
print("Source prepared. Compile extensions while inference is stopped; mount immutable model files separately.")
