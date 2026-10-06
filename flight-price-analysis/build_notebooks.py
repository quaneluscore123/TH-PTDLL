"""Chuyển các file notebooks/*.py (định dạng '# %%') thành .ipynb và chạy để lưu kết quả vào notebook.

    python build_notebooks.py 01_bronze_ingest          # chuyển + chạy một notebook
    python build_notebooks.py 03 --no-run               # chỉ chuyển, không chạy
"""
import glob, os, re, sys, time
import nbformat
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(HERE, "notebooks")


def py_to_nb(py_path):
    text = open(py_path, encoding="utf-8").read()
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    for block in re.split(r"^# %%", text, flags=re.M)[1:]:
        header, _, body = block.partition("\n")
        if header.strip().startswith("[markdown]"):
            md = "\n".join(re.sub(r"^# ?", "", l) for l in body.strip("\n").splitlines())
            nb.cells.append(nbformat.v4.new_markdown_cell(md))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(body.strip("\n")))
    return nb


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for prefix in args:
        for py_path in sorted(glob.glob(os.path.join(NB_DIR, f"{prefix}*.py"))):
            nb = py_to_nb(py_path)
            out = py_path[:-3] + ".ipynb"
            if "--no-run" not in sys.argv:
                t0 = time.time()
                print("Chạy", os.path.basename(out), flush=True)
                try:
                    NotebookClient(nb, timeout=None, kernel_name="python3",
                                   resources={"metadata": {"path": NB_DIR}}).execute()
                except Exception:
                    nbformat.write(nb, out)      # vẫn lưu để xem ô nào lỗi
                    raise
                print(f"  xong sau {(time.time() - t0) / 60:.1f} phút", flush=True)
            nbformat.write(nb, out)
            print("Đã ghi", out, flush=True)


if __name__ == "__main__":
    main()
