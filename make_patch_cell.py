"""Emit a single paste-into-Colab cell that adds the ensemble to a LIVE session.

Re-uploading the notebook would wipe the trained checkpoint, so this ships only
the files the ensemble needs, writes them next to the ones already there, and
runs it. Everything else in the session is left alone.

  python make_patch_cell.py      ->  patch_cell.txt
"""
import pathlib

SRC = ["probe.py", "model.py", "plots.py", "07_ensemble.py"]
OUT = "patch_cell.txt"

HEADER = '''# --- ensemble patch -------------------------------------------------------
# Adds the frozen probe + the ensemble step to this running session.
# Your trained jevlite.pt is untouched - nothing here retrains anything.
import pathlib, subprocess, os

FILES = {}
'''

FOOTER = '''
for name, text in FILES.items():
    pathlib.Path(name).write_text(text.lstrip("\\n") + "\\n", encoding="utf-8")
print("patched:", ", ".join(sorted(FILES)))


def run(cmd):                      # redefined in case this cell runs standalone
    print("$", cmd, flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    p = subprocess.Popen(cmd, shell=True, env=env, text=True, bufsize=1,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait():
        raise SystemExit(f"*** FAILED (exit {p.returncode}): {cmd}")
    print("ok", flush=True)


run("python 07_ensemble.py")
run("python plots.py --outdir plots")

from IPython.display import Image, display
display(Image(filename="plots/6_ensemble.png"))
'''


if __name__ == "__main__":
    here = pathlib.Path(__file__).parent
    parts = [HEADER]
    for name in SRC:
        text = (here / name).read_text(encoding="utf-8")
        if "'''" in text:
            raise SystemExit(f"{name} contains ''' and cannot be embedded raw")
        parts += [f"FILES[{name!r}] = r'''", text.rstrip("\n"), "'''", ""]
    parts.append(FOOTER)
    body = "\n".join(parts)
    (here / OUT).write_text(body, encoding="utf-8")
    print(f"wrote {OUT}: {len(body)/1024:.0f} KB, {len(SRC)} files embedded")
