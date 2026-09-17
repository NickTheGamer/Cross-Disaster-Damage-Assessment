#!/usr/bin/env bash
# Fetch, verify and extract the xBD challenge-format dataset.
#
# The download links from xview2.org are signed and short-lived: if a download
# stalls, reload https://xview2.org/download and replace the URL in the URL file.
# SHA1s below are from that page and do NOT change, so a resumed or re-fetched
# file is still checked against the real thing.
#
# Usage:
#   1. cp scripts/xbd_urls.example.txt xbd_urls.txt
#   2. paste your signed URLs into xbd_urls.txt
#   3. ./scripts/fetch_xbd.sh              # all keys
#      ./scripts/fetch_xbd.sh train tier3  # only these
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL_FILE="${XBD_URL_FILE:-$ROOT/xbd_urls.txt}"
DEST="${XBD_DEST:-$ROOT/raw_dataset}"

# key -> expected SHA1, straight from the xview2.org download page.
declare -A SHA1=(
  [train]=b37a4ef4ee9c909e2b19d046e49d42ee3965714b
  [tier3]=5bf6aaf8a71980b633fb4661776a99a200891de5
  [test]=86ed3dba2f8d16ceceb75d451005054fefa9616f
  [hold]=fe7f162f0895bfaff134cab3abc23872f38d17da
  [geotransforms]=0b3bda08084ac102d8b540261ebbba0094203a2f
)

[[ -f "$URL_FILE" ]] || { echo "No URL file at $URL_FILE" >&2
  echo "Copy scripts/xbd_urls.example.txt there and paste your signed URLs." >&2; exit 1; }

keys=("$@"); [[ ${#keys[@]} -gt 0 ]] || keys=(train tier3 test hold geotransforms)
mkdir -p "$DEST"

# Warn early rather than dying 25GB in.
avail=$(df -BG --output=avail "$DEST" | tail -1 | tr -dc '0-9')
(( avail > 80 )) || echo "WARNING: only ${avail}GB free; ~65GB needed for archives + extraction." >&2

for key in "${keys[@]}"; do
  [[ -v SHA1[$key] ]] || { echo "Unknown key: $key" >&2; exit 1; }
  url=$(awk -v k="$key" '$1==k {print $2; exit}' "$URL_FILE")
  [[ -n "$url" ]] || { echo "SKIP $key — no URL in $URL_FILE"; continue; }

  archive="$DEST/$key.tar.gz"
  stamp="$DEST/.$key.verified"

  if [[ -f "$stamp" ]]; then
    echo "== $key: already downloaded, verified and extracted; skipping."
    continue
  fi

  echo "== $key: downloading (resumable)"
  # -C - resumes a partial file; a stale link fails here, not silently.
  curl -fL -C - --retry 5 --retry-delay 5 -o "$archive" "$url" || {
    echo "FAILED $key. If the link went stale, reload the download page and" >&2
    echo "replace the '$key' URL in $URL_FILE, then re-run — the partial file resumes." >&2
    exit 1; }

  echo "== $key: verifying SHA1"
  actual=$(sha1sum "$archive" | cut -d' ' -f1)
  if [[ "$actual" != "${SHA1[$key]}" ]]; then
    echo "SHA1 MISMATCH for $key" >&2
    echo "  expected ${SHA1[$key]}" >&2
    echo "  actual   $actual" >&2
    echo "Download is corrupt or incomplete. Delete $archive and re-run." >&2
    exit 1
  fi
  echo "   ok: $actual"

  echo "== $key: extracting"
  tar -xzf "$archive" -C "$DEST"
  touch "$stamp"
  echo "== $key: done. Archive kept at $archive (delete once you're happy)."
done

echo
echo "All requested keys complete. Contents of $DEST:"
ls -la "$DEST"
