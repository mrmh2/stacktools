import os
import sys
import shutil
import pathlib

diter = pathlib.Path(sys.argv[1]).iterdir()
dst_dirpath = pathlib.Path("local-data")

for dirpath in diter:
    src_fpath = dirpath/"Root_segments.tif.csv"
    root_name = os.path.basename(str(dirpath))
    dst_fpath = root_name + ".csv"
    shutil.copy(str(src_fpath), dst_fpath)
