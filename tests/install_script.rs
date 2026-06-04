#[test]
fn installer_stages_binary_before_replacing_existing_executable() {
    let script = include_str!("../scripts/install.sh");

    assert!(script.contains(r#"mv -f "$staged_path" "$INSTALL_DIR/$binary""#));
    assert!(!script.contains(r#"cp "$tmpdir/$binary" "$INSTALL_DIR/$binary""#));
    assert!(!script.contains(r#"cp "$cargo_root/bin/$binary" "$INSTALL_DIR/$binary""#));
}
