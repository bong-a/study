#!/usr/bin/env python3
"""Regenerate index.html as a table-of-contents page linking every HTML file in the repo, grouped by folder."""
import re
from pathlib import Path

EXCLUDE_DIR_NAMES = {".git", ".claude", ".github", "node_modules"}

TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    color-scheme: light dark;
    --bg: #ffffff;
    --fg: #1a1a1a;
    --muted: #666666;
    --border: #e5e5e5;
    --accent: #2563eb;
    --card-bg: #f7f7f8;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #14151a;
      --fg: #e8e8ea;
      --muted: #9a9aa2;
      --border: #2a2b32;
      --accent: #6ea8fe;
      --card-bg: #1c1d24;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 2.5rem 1.25rem 4rem;
    background: var(--bg);
    color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.5;
  }}
  main {{
    max-width: 720px;
    margin: 0 auto;
  }}
  h1 {{
    font-size: 1.6rem;
    margin: 0 0 0.4rem;
  }}
  p.subtitle {{
    color: var(--muted);
    margin: 0 0 2.5rem;
    font-size: 0.95rem;
  }}
  h2 {{
    font-size: 1.05rem;
    margin: 2.2rem 0 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }}
  ul {{
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: 0.5rem;
  }}
  li a {{
    display: block;
    padding: 0.7rem 0.9rem;
    border-radius: 8px;
    background: var(--card-bg);
    color: var(--fg);
    text-decoration: none;
    font-size: 0.92rem;
    border: 1px solid var(--border);
    transition: border-color 0.15s ease;
  }}
  li a:hover {{
    border-color: var(--accent);
    color: var(--accent);
  }}
</style>
</head>
<body>
<main>
  <h1>{title}</h1>
  <p class="subtitle">{subtitle}</p>

{body}
</main>
</body>
</html>
"""


def find_repo_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / ".git").exists():
            return candidate
    return start.resolve()


def natural_key(s: str):
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", s)]


def humanize(name: str) -> str:
    name = re.sub(r"^\d+(-\d+)*-", "", name)
    name = name.replace("-", " ").replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()
    if name and name[0].islower():
        name = name[0].upper() + name[1:]
    return name


def breadcrumb(rel_dir: Path) -> str:
    return " / ".join(humanize(part) for part in rel_dir.parts)


def collect_html_files(root: Path):
    groups = {}
    for path in sorted(root.rglob("*.html")):
        rel = path.relative_to(root)
        if rel.parent == Path(".") and rel.name.lower() == "index.html":
            continue
        if any(part in EXCLUDE_DIR_NAMES for part in rel.parts):
            continue
        groups.setdefault(rel.parent, []).append(rel)
    for files in groups.values():
        files.sort(key=lambda p: natural_key(p.name))
    return dict(sorted(groups.items(), key=lambda kv: natural_key(str(kv[0]))))


def render(groups, title: str, subtitle: str) -> str:
    sections = []
    for rel_dir, files in groups.items():
        heading = breadcrumb(rel_dir) if str(rel_dir) != "." else "Root"
        items = "\n".join(
            f'    <li><a href="{str(f).replace(chr(92), "/")}">{humanize(f.stem)}</a></li>'
            for f in files
        )
        sections.append(f"  <h2>{heading}</h2>\n  <ul>\n{items}\n  </ul>")
    body = "\n\n".join(sections)
    return TEMPLATE.format(title=title, subtitle=subtitle, body=body)


def main():
    root = find_repo_root(Path.cwd())
    groups = collect_html_files(root)
    html = render(groups, title="Study Notes", subtitle="저장소 내 HTML 문서 목차")
    out_path = root / "index.html"
    out_path.write_text(html, encoding="utf-8")
    total = sum(len(files) for files in groups.values())
    print(f"Wrote {out_path} — {total} files across {len(groups)} folders")


if __name__ == "__main__":
    main()
