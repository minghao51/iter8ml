#!/usr/bin/env bash
set -euo pipefail

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
  hash=$(md5 -q "$f")
  cached=$(grep "^$name " "$cache_file" 2>/dev/null || true)

  if [ "$cached" = "$name $hash" ] && [ -f "$export_dir/$name.html" ]; then
    echo "Cached $name (unchanged)"
    continue
  fi

  echo "Exporting $name..."
  uv run marimo export html "$f" -o "$export_dir/$name.html"

  # Update cache
  if grep -q "^$name " "$cache_file" 2>/dev/null; then
    sed -i '' "s/^$name .*/$name $hash/" "$cache_file"
  else
    echo "$name $hash" >> "$cache_file"
  fi
done

echo "Done. Exports in $export_dir/"
