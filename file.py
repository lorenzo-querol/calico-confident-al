import os
import shutil


def copy_files(src_dir, dst_dir):
    for dirpath, dirnames, filenames in os.walk(src_dir):
        dst_subdir = dirpath.replace(src_dir, dst_dir)
        if not os.path.exists(dst_subdir):
            os.makedirs(dst_subdir)

        for filename in filenames:
            src_file = os.path.join(dirpath, filename)
            dst_file = os.path.join(dst_subdir, filename)

            if os.path.exists(dst_file):
                base, extension = os.path.splitext(dst_file)
                i = 1
                while os.path.exists(dst_file):
                    dst_file = f"{base}_{i}{extension}"
                    i += 1
            shutil.copy2(src_file, dst_file)


# Usage
src_dir = "/path/to/source/directory"
dst_dir = "/path/to/destination/directory"
copy_files(src_dir, dst_dir)
