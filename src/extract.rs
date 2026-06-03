use std::{fs::File, io::Read, path::Path};

use anyhow::{Context, Result};
use zip::ZipArchive;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedDocument {
    pub text: String,
    pub line_count: usize,
}

pub fn extract_text(path: &Path) -> Result<ExtractedDocument> {
    if is_docx(path) {
        return extract_docx(path);
    }

    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(ExtractedDocument {
        line_count: line_count(&text),
        text,
    })
}

fn extract_docx(path: &Path) -> Result<ExtractedDocument> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut archive =
        ZipArchive::new(file).with_context(|| format!("reading {}", path.display()))?;
    let mut entry = archive
        .by_name("word/document.xml")
        .with_context(|| format!("reading word/document.xml from {}", path.display()))?;
    let mut xml = String::new();
    entry
        .read_to_string(&mut xml)
        .with_context(|| format!("extracting text from {}", path.display()))?;

    let text = extract_docx_xml_text(&xml);
    Ok(ExtractedDocument {
        line_count: line_count(&text),
        text,
    })
}

fn is_docx(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("docx"))
}

fn line_count(text: &str) -> usize {
    if text.is_empty() {
        0
    } else {
        text.lines().count()
    }
}

fn extract_docx_xml_text(xml: &str) -> String {
    let mut text = String::new();
    let mut cursor = 0usize;

    while cursor < xml.len() {
        let remainder = &xml[cursor..];
        if remainder.starts_with("<w:t") {
            if let Some(tag_end) = remainder.find('>') {
                let content_start = cursor + tag_end + 1;
                if let Some(content_end) = xml[content_start..].find("</w:t>") {
                    let raw = &xml[content_start..content_start + content_end];
                    push_decoded_xml_text(&mut text, raw);
                    cursor = content_start + content_end + "</w:t>".len();
                    continue;
                }
            }
        } else if remainder.starts_with("<w:tab") {
            text.push('\t');
            cursor += tag_length(remainder);
            continue;
        } else if remainder.starts_with("<w:br") || remainder.starts_with("<w:cr") {
            push_newline(&mut text);
            cursor += tag_length(remainder);
            continue;
        } else if remainder.starts_with("</w:p>") {
            push_newline(&mut text);
            cursor += "</w:p>".len();
            continue;
        }

        cursor += 1;
    }

    text.trim_end_matches(['\n', '\r']).to_string()
}

fn tag_length(fragment: &str) -> usize {
    fragment
        .find('>')
        .map(|idx| idx + 1)
        .unwrap_or(fragment.len())
}

fn push_newline(text: &mut String) {
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
}

fn push_decoded_xml_text(text: &mut String, raw: &str) {
    text.push_str(&decode_xml_entities(raw));
}

fn decode_xml_entities(raw: &str) -> String {
    let mut decoded = raw.replace("&lt;", "<");
    decoded = decoded.replace("&gt;", ">");
    decoded = decoded.replace("&amp;", "&");
    decoded = decoded.replace("&apos;", "'");
    decoded.replace("&quot;", "\"")
}
