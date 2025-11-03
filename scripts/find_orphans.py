import os
from pathlib import Path

root = Path('.').resolve()
src_dir = root / 'src'

# Gather files to search
search_files = []
for p in root.rglob('*'):
    if p.is_file():
        # ignore venv and .git
        if 'venv' in p.parts or '.venv' in p.parts or p.parts[0].startswith('.git'):
            continue
        # include code and templates
        if p.suffix in ('.py', '.html', '.js', '.json', '.md'):
            search_files.append(p)

# Candidate modules under src
candidates = []
for py in src_dir.rglob('*.py'):
    # skip package __init__ and dunder
    if py.name == '__init__.py':
        continue
    # skip obvious entrypoints
    if py.name in ('main.py', 'streamlit_app.py'):
        continue
    candidates.append(py)

results = []
for c in candidates:
    rel = c.relative_to(root)
    module_path = '.'.join(rel.with_suffix('').parts)  # e.g., src.llm.client
    basename = c.stem

    found_elsewhere = False
    for f in search_files:
        # skip the file itself
        if f.resolve() == c.resolve():
            continue
        try:
            text = f.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        if module_path in text or f"from {module_path}" in text or f"import {module_path}" in text:
            found_elsewhere = True
            break
        # also check basename occurrences but avoid trivial hits for common words
        if basename and basename not in ('utils','config','client','manager','schema','routes','views','main','actions','parser'):
            if basename in text:
                # heuristic: check for import statements or attribute access
                if f"from {basename} " in text or f"import {basename}" in text or f".{basename}" in text:
                    found_elsewhere = True
                    break
    if not found_elsewhere:
        results.append(str(rel))

print('Likely orphaned python files (candidates):')
for r in sorted(results):
    print(r)

print('\nNote: This is a heuristic scan. Review before deleting. It may false-positive for files used via dynamic imports or templates.')
