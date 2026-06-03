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
        *) echo "unsupported architecture: $arch" >&2; exit 1 ;;
      esac
      ;;
    linux)
      case "$arch" in
        x86_64|amd64) echo "x86_64-unknown-linux-gnu" ;;
        *) echo "unsupported architecture: $arch" >&2; exit 1 ;;
      esac
      ;;
    mingw*|msys*|cygwin*)
      case "$arch" in
        x86_64|amd64) echo "x86_64-pc-windows-msvc" ;;
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

shell_profile() {
  if [ -n "${ENF_PROFILE:-}" ]; then
    echo "$ENF_PROFILE"
    return
  fi

  shell_name="$(basename "${SHELL:-sh}")"
  case "$shell_name" in
    zsh) echo "$HOME/.zshrc" ;;
    bash)
      if [ "$(uname -s)" = "Darwin" ]; then
        echo "$HOME/.bash_profile"
      else
        echo "$HOME/.bashrc"
      fi
      ;;
    *) echo "$HOME/.profile" ;;
  esac
}

ensure_on_path() {
  case ":$PATH:" in
    *":$INSTALL_DIR:"*) return ;;
  esac

  profile="$(shell_profile)"
  line="export PATH=\"$INSTALL_DIR:\$PATH\""
  if [ ! -f "$profile" ] || ! grep -F "$line" "$profile" >/dev/null 2>&1; then
    {
      printf '\n# Elephant Never Forgets CLI\n'
      printf '%s\n' "$line"
    } >> "$profile"
    echo "added $INSTALL_DIR to PATH in $profile"
  fi
  echo "restart your shell or run: export PATH=\"$INSTALL_DIR:\$PATH\""
}

target="$(detect_target)"
archive="enf-$target.tar.gz"
binary="enf"
case "$target" in
  *windows*) binary="enf.exe" ;;
esac
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
  cp "$tmpdir/$binary" "$INSTALL_DIR/$binary"
  chmod 755 "$INSTALL_DIR/$binary"
  echo "installed $binary to $INSTALL_DIR/$binary"
  ensure_on_path
else
  echo "release archive unavailable for $target; falling back to cargo install" >&2
  install_from_cargo
fi
