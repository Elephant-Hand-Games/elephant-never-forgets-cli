#!/usr/bin/env sh
set -eu

REPO="${ENF_REPO:-Elephant-Hand-Games/elephant-never-forgets-cli}"
VERSION="${ENF_VERSION:-latest}"
INSTALL_DIR="${ENF_INSTALL_DIR:-$HOME/.local/bin}"

detect_target() {
  os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  arch="$(uname -m)"

  case "$os" in
    darwin)
      case "$arch" in
        arm64|aarch64) echo "aarch64-apple-darwin" ;;
        x86_64|amd64) echo "x86_64-apple-darwin" ;;
        *) echo "unsupported architecture: $arch" >&2; exit 1 ;;
      esac
      ;;
    linux)
      case "$arch" in
        x86_64|amd64) echo "x86_64-unknown-linux-gnu" ;;
        aarch64|arm64) echo "aarch64-unknown-linux-gnu" ;;
        *) echo "unsupported architecture: $arch" >&2; exit 1 ;;
      esac
      ;;
    *)
      echo "unsupported OS: $os" >&2
      exit 1
      ;;
  esac
}

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

download() {
  url="$1"
  dest="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$dest"
  elif command -v wget >/dev/null 2>&1; then
    wget -q "$url" -O "$dest"
  else
    echo "missing curl or wget" >&2
    exit 1
  fi
}

install_from_cargo() {
  need cargo
  if [ -f Cargo.toml ] && grep -q '^name = "elephant-never-forgets"' Cargo.toml; then
    cargo install --path . --locked
  else
    cargo install --git "https://github.com/$REPO.git" --locked
  fi
}

target="$(detect_target)"
archive="enf-$target.tar.gz"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

if [ "$VERSION" = "latest" ]; then
  url="https://github.com/$REPO/releases/latest/download/$archive"
else
  url="https://github.com/$REPO/releases/download/$VERSION/$archive"
fi

if download "$url" "$tmpdir/$archive"; then
  mkdir -p "$INSTALL_DIR"
  tar -xzf "$tmpdir/$archive" -C "$tmpdir"
  install "$tmpdir/enf" "$INSTALL_DIR/enf"
  echo "installed enf to $INSTALL_DIR/enf"
else
  echo "release archive unavailable for $target; falling back to cargo install" >&2
  install_from_cargo
fi

