import os


def _strip_inline_comment(line):
    in_single = False
    in_double = False
    out = []
    for ch in line:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        if ch == "#" and not in_single and not in_double:
            break
        out.append(ch)
    return "".join(out)


def _parse_scalar(value):
    if not value:
        return ""
    if (value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'"):
        inner = value[1:-1]
        if value[0] == '"':
            inner = inner.replace("\\\\", "\\")
        return inner
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.isdigit():
        return int(value)
    return value


def _parse_simple_yaml(text):
    root = {}
    stack = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        line = _strip_inline_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key_part, sep, value_part = line.lstrip().partition(":")
        if not sep:
            continue
        key = key_part.strip()
        if key.startswith("\ufeff"):
            key = key.lstrip("\ufeff")
        value = value_part.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1] if stack else root
        if value == "":
            node = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return _parse_simple_yaml(handle.read())


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    defaults_path = os.path.join(root, "configs", "defaults.yaml")
    paths_path = os.path.join(root, "configs", "paths.yaml")

    _ = _load_yaml(defaults_path)
    paths = _load_yaml(paths_path)

    datasets = paths.get("datasets", {})
    for key in sorted(datasets.keys()):
        print(key)

    missing = []
    for section in ("datasets", "embeddings"):
        items = paths.get(section, {})
        for key, value in items.items():
            if not os.path.exists(value):
                missing.append(f"{section}.{key}")

    if missing:
        print("Missing paths:")
        for item in missing:
            print(item)
        raise SystemExit(1)

    print("All paths exist.")


if __name__ == "__main__":
    main()
