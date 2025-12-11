import os
import os.path as osp
import shutil
import sys
import warnings
from setuptools import find_packages, setup


def readme():
    """Load README.md."""
    with open("README.md", encoding="utf-8") as f:
        return f.read()


def parse_requirements(fname="requirements.txt", with_version=True):
    """Parse a requirements.txt file."""
    import re
    from os.path import exists

    require_fpath = fname

    def parse_line(line):
        if line.startswith("-r "):
            target = line.split(" ")[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {"line": line}
            if line.startswith("-e "):
                info["package"] = line.split("#egg=")[1]
            elif "@git+" in line:
                info["package"] = line
            else:
                pat = "(" + "|".join([">=", "==", ">"]) + ")"
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info["package"] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ";" in rest:
                        version, platform_deps = map(str.strip, rest.split(";"))
                        info["platform_deps"] = platform_deps
                    else:
                        version = rest
                    info["version"] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info["package"]]
                if with_version and "version" in info:
                    parts.extend(info["version"])
                platform_deps = info.get("platform_deps")
                if platform_deps is not None:
                    parts.append(";" + platform_deps)
                yield "".join(parts)

    return list(gen_packages_items())


def add_mim_extension():
    """Add extra files required by MIM by either symlink or copy."""
    if "develop" in sys.argv:
        mode = "symlink"
    elif "sdist" in sys.argv or "bdist_wheel" in sys.argv:
        mode = "copy"
    else:
        return

    filenames = ["tools", "configs", "demo", "model-index.yml"]
    repo_path = osp.dirname(__file__)
    mim_path = osp.join(repo_path, "mmrotate", ".mim")
    os.makedirs(mim_path, exist_ok=True)

    for filename in filenames:
        if osp.exists(filename):
            src_path = osp.join(repo_path, filename)
            tar_path = osp.join(mim_path, filename)

            if osp.isfile(tar_path) or osp.islink(tar_path):
                os.remove(tar_path)
            elif osp.isdir(tar_path):
                shutil.rmtree(tar_path)

            if mode == "symlink":
                src_relpath = osp.relpath(src_path, osp.dirname(tar_path))
                try:
                    os.symlink(src_relpath, tar_path)
                except OSError:
                    mode = "copy"
                    warnings.warn(
                        f"Failed to create symlink for {src_relpath}; switching to copy."
                    )
                else:
                    continue

            if mode == "copy":
                if osp.isfile(src_path):
                    shutil.copyfile(src_path, tar_path)
                elif osp.isdir(src_path):
                    shutil.copytree(src_path, tar_path)
                else:
                    warnings.warn(f"Cannot copy file: {src_path}")
            else:
                raise ValueError(f"Invalid mode: {mode}")


if __name__ == "__main__":
    add_mim_extension()
    setup(
        name="abbspo",
        version="0.1.0",
        description=(
            "ABBSPO: Adaptive Bounding Box Scaling and Symmetric Prior Based "
            "Orientation Prediction for Detecting Aerial Image Objects"
        ),
        long_description=readme(),
        long_description_content_type="text/markdown",
        author="ABBSPO Authors",
        author_email="",
        keywords="computer vision, oriented object detection, weak supervision",
        url="https://github.com/KAIST-VICLab/ABBSPO",
        packages=find_packages(exclude=("configs", "tools", "demo", "docs")),
        include_package_data=True,
        classifiers=[
            "Development Status :: 4 - Beta",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
        ],
        license="Apache License 2.0",
        install_requires=parse_requirements("requirements.txt"),
        extras_require={
            "all": parse_requirements("requirements.txt"),
        },
        zip_safe=False,
    )
