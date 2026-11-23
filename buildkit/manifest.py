"""
buildkit.manifest: load the project manifest (project.toml) and resolve
the language, source files, and target settings.
"""
from dataclasses import dataclass, field
from pathlib import Path

from .toml import load as load_toml
from .language import Language


@dataclass
class Manifest:
    root: Path
    name: str
    language: Language
    sources: list
    cc: str = "gcc"
    cflags: str = "-O2 -Wall"
    output: str = ""
    runtime: str = ""
    build_dir: str = ".light"

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        root = path.parent
        data = load_toml(path)

        proj = data.get("project", {})
        name = proj.get("name", root.name)

        lang_ref = proj.get("language")
        if not lang_ref:
            raise ValueError("project.language is required (path to lang.toml)")
        language = Language.load(root / lang_ref)

        src = data.get("sources", {})
        ext = src.get("ext", ".light")
        if "files" in src:
            sources = [root / f for f in src["files"]]
        else:
            src_root = root / src.get("root", "src")
            sources = sorted(src_root.rglob(f"*{ext}"))
        if not sources:
            raise ValueError(f"no {ext} sources found")

        target = data.get("target", {})
        return cls(
            root=root,
            name=name,
            language=language,
            sources=sources,
            cc=target.get("cc", "gcc"),
            cflags=target.get("cflags", "-O2 -Wall"),
            output=target.get("output", name),
            runtime=target.get("runtime", ""),
            build_dir=data.get("build", {}).get("dir", ".light"),
        )
