use std::io::Write;

use elephant_never_forgets::extract;
use zip::{write::SimpleFileOptions, ZipWriter};

#[test]
fn extracts_docx_text_with_line_breaks_and_entities() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("sample.docx");
    let file = std::fs::File::create(&path).unwrap();
    let mut zip = ZipWriter::new(file);
    zip.start_file("word/document.xml", SimpleFileOptions::default())
        .unwrap();
    zip.write_all(
        br#"
        <w:document>
          <w:body>
            <w:p><w:r><w:t>Agent &amp; workflow</w:t></w:r></w:p>
            <w:p><w:r><w:t>Second</w:t><w:tab/><w:t>line</w:t></w:r></w:p>
          </w:body>
        </w:document>
        "#,
    )
    .unwrap();
    zip.finish().unwrap();

    let extracted = extract::extract_text(&path).unwrap();

    assert!(extracted.text.contains("Agent & workflow"));
    assert!(extracted.text.contains("Second"));
    assert!(extracted.text.contains("line"));
    assert_eq!(extracted.line_count, 2);
}
