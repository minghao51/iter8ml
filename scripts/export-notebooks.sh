#!/usr/bin/env bash
set -euo pipefail

if command -v md5sum &>/dev/null; then
  md5hash() { md5sum "$1" | cut -d' ' -f1; }
else
  md5hash() { md5 -q "$1"; }
fi

export_dir="docs/notebooks/exports"
cache_file="$export_dir/.export-cache"
mkdir -p "$export_dir"

# Initialize cache file
touch "$cache_file"

for f in notebooks/*.py; do
  if ! head -5 "$f" | grep -q 'marimo.App('; then
    echo "Skipping $f (not a marimo notebook)"
    continue
  fi

  name=$(basename "$f" .py)
  hash=$(md5hash "$f")
  cached=$(grep "^$name " "$cache_file" 2>/dev/null || true)

  if [ "$cached" = "$name $hash" ] && [ -f "$export_dir/$name.html" ]; then
    echo "Cached $name (unchanged)"
    continue
  fi

  echo "Exporting $name..."
  uv run marimo export html "$f" -o "$export_dir/$name.html"

  # Update cache
  if grep -q "^$name " "$cache_file" 2>/dev/null; then
    sed -i.bak "s/^$name .*/$name $hash/" "$cache_file" && rm -f "$cache_file.bak"
  else
    echo "$name $hash" >> "$cache_file"
  fi
done

echo "Done. Exports in $export_dir/"
