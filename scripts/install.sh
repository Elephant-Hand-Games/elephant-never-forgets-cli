#!/usr/bin/env sh
set -eu

REPO="${ENF_REPO:-Elephant-Hand-Games/elephant-never-forgets-cli}"
VERSION="${ENF_VERSION:-latest}"
INSTALL_DIR="${ENF_INSTALL_DIR:-$HOME/.enf/bin}"
INSTALL_METHOD="${ENF_INSTALL_METHOD:-binary}"

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
  cargo_root="$tmpdir/cargo-root"
  cargo_home="$tmpdir/cargo-home"
  if [ -f Cargo.toml ] && grep -q '^name = "elephant-never-forgets"' Cargo.toml; then
    CARGO_HOME="$cargo_home" cargo install --path . --locked --root "$cargo_root"
  else
    if [ "$VERSION" = "latest" ]; then
      CARGO_HOME="$cargo_home" cargo install --git "https://github.com/$REPO.git" --locked --root "$cargo_root"
    else
      CARGO_HOME="$cargo_home" cargo install --git "https://github.com/$REPO.git" --tag "$VERSION" --locked --root "$cargo_root"
    fi
  fi
  mkdir -p "$INSTALL_DIR"
  cp "$cargo_root/bin/$binary" "$INSTALL_DIR/$binary"
  chmod 755 "$INSTALL_DIR/$binary"
  echo "installed $binary to $INSTALL_DIR/$binary from source"
  ensure_on_path
}

verify_checksum() {
  archive_path="$1"
  checksum_path="$2"
  expected="$(awk '{print $1}' "$checksum_path")"
  if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$archive_path" | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    actual="$(shasum -a 256 "$archive_path" | awk '{print $1}')"
  else
    echo "missing sha256sum or shasum for checksum verification" >&2
    exit 1
  fi
  if [ "$actual" != "$expected" ]; then
    echo "checksum mismatch for $(basename "$archive_path")" >&2
    echo "expected: $expected" >&2
    echo "actual:   $actual" >&2
    exit 1
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

case "$INSTALL_METHOD" in
  binary) ;;
  cargo|source)
    install_from_cargo
    exit 0
    ;;
  *)
    echo "unsupported ENF_INSTALL_METHOD: $INSTALL_METHOD (use binary or cargo)" >&2
    exit 1
    ;;
esac

if [ "$VERSION" = "latest" ]; then
  url="https://github.com/$REPO/releases/latest/download/$archive"
else
  url="https://github.com/$REPO/releases/download/$VERSION/$archive"
fi

checksum_url="$url.sha256"

if download "$url" "$tmpdir/$archive"; then
  if download "$checksum_url" "$tmpdir/$archive.sha256"; then
    verify_checksum "$tmpdir/$archive" "$tmpdir/$archive.sha256"
  else
    echo "checksum unavailable for $archive; falling back to cargo install" >&2
    install_from_cargo
    exit 0
  fi
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
