use std::io::{Cursor, Read};

use serde_json::Value;
use zip::ZipArchive;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub file_path: String,
    pub language: String,
    pub chunk_type: String,
    pub title: Option<String>,
    pub breadcrumb: Vec<String>,
    pub start_line: usize,
    pub end_line: usize,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct ChunkOptions {
    pub target_min_tokens: usize,
    pub target_max_tokens: usize,
    pub hard_max_tokens: usize,
    pub overlap_tokens: usize,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        Self {
            target_min_tokens: 300,
            target_max_tokens: 900,
            hard_max_tokens: 1_500,
            overlap_tokens: 100,
        }
    }
}

pub fn estimate_tokens(text: &str) -> usize {
    text.chars().count() / 4
}

pub fn line_range_text(lines: &[&str], start: usize, end: usize) -> String {
    lines[start..end].join("\n")
}

#[allow(clippy::too_many_arguments)]
pub fn make_chunk(
    file_path: &str,
    language: &str,
    chunk_type: &str,
    title: Option<String>,
    breadcrumb: Vec<String>,
    start_line: usize,
    end_line: usize,
    body: String,
) -> Chunk {
    let mut header = String::new();
    header.push_str(&format!("File: {file_path}\n"));
    header.push_str(&format!("Language: {language}\n"));
    header.push_str(&format!("Chunk type: {chunk_type}\n"));
    if !breadcrumb.is_empty() {
        header.push_str(&format!("Section: {}\n", breadcrumb.join(" > ")));
    }
    if let Some(title) = &title {
        header.push_str(&format!("Symbol: {title}\n"));
    }
    header.push('\n');
    header.push_str(&body);
    Chunk {
        file_path: file_path.to_string(),
        language: language.to_string(),
        chunk_type: chunk_type.to_string(),
        title,
        breadcrumb,
        start_line,
        end_line,
        text: header,
    }
}

pub fn process_full_file(file_path: &str, language: &str, source: &str) -> Vec<Chunk> {
    let text = source.trim_end_matches(['\n', '\r']).to_string();
    if text.is_empty() {
        return Vec::new();
    }
    let line_count = source.lines().count().max(1);
    vec![Chunk {
        file_path: file_path.to_string(),
        language: language.to_string(),
        chunk_type: "full_file".into(),
        title: None,
        breadcrumb: Vec::new(),
        start_line: 1,
        end_line: line_count,
        text,
    }]
}

pub fn process_line_window(
    file_path: &str,
    language: &str,
    source: &str,
    opts: &ChunkOptions,
) -> Vec<Chunk> {
    let lines: Vec<&str> = source.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < lines.len() {
        let (end, _token_count) = chunk_window(&lines, start, opts);
        let chunk_text = lines[start..end].join("\n");
        chunks.push(Chunk {
            file_path: file_path.to_string(),
            language: language.to_string(),
            chunk_type: "line_window".into(),
            title: None,
            breadcrumb: Vec::new(),
            start_line: start + 1,
            end_line: end,
            text: chunk_text,
        });

        if end >= lines.len() {
            break;
        }
        start = overlap_start(&lines, start, end, opts.overlap_tokens);
    }

    chunks
}

pub fn process_file_for_rag(file_path: &str, bytes: &[u8]) -> Vec<Chunk> {
    let ext = file_path
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext == "docx" {
        return process_docx(file_path, bytes);
    }

    let source = String::from_utf8_lossy(bytes);
    match ext.as_str() {
        "md" | "markdown" => process_markdown(file_path, &source),
        "txt" => process_txt(file_path, &source),
        "js" | "ts" | "jsx" | "tsx" => process_js_ts_family(file_path, &source),
        "gd" => process_gdscript(file_path, &source),
        "rs" => process_rust(file_path, &source),
        "html" | "htm" => process_html(file_path, &source),
        "css" => process_css(file_path, &source),
        "json" => process_json(file_path, &source),
        "toml" | "tmol" => process_toml_like(file_path, &source),
        "yml" | "yaml" => process_yaml(file_path, &source),
        "py" => process_python(file_path, &source),
        "go" => process_go(file_path, &source),
        "zig" => process_zig(file_path, &source),
        _ => {
            if looks_like_source_code(&source) {
                process_general_code(file_path, infer_code_language(file_path), &source)
            } else {
                process_txt(file_path, &source)
            }
        }
    }
}

pub fn process_markdown(file_path: &str, source: &str) -> Vec<Chunk> {
    let opts = ChunkOptions {
        target_min_tokens: 300,
        target_max_tokens: 900,
        hard_max_tokens: 1_500,
        overlap_tokens: 100,
    };
    let lines: Vec<&str> = source.lines().collect();
    let mut chunks = Vec::new();
    let mut heading_stack: Vec<(usize, String)> = Vec::new();
    let mut section_start = 0usize;
    let mut current_breadcrumb: Vec<String> = Vec::new();
    let mut in_fence = false;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        if let Some((level, title)) = parse_markdown_heading(trimmed) {
            if i > section_start {
                let section = line_range_text(&lines, section_start, i);
                chunks.extend(chunk_markdown_section(
                    file_path,
                    &section,
                    section_start + 1,
                    i,
                    current_breadcrumb.clone(),
                    &opts,
                ));
            }
            while heading_stack
                .last()
                .map(|(existing_level, _)| *existing_level >= level)
                .unwrap_or(false)
            {
                heading_stack.pop();
            }
            heading_stack.push((level, title));
            current_breadcrumb = heading_stack.iter().map(|(_, t)| t.clone()).collect();
            section_start = i;
        }
    }

    if section_start < lines.len() {
        let section = line_range_text(&lines, section_start, lines.len());
        chunks.extend(chunk_markdown_section(
            file_path,
            &section,
            section_start + 1,
            lines.len(),
            current_breadcrumb,
            &opts,
        ));
    }
    chunks
}

pub fn process_general_code(file_path: &str, language: &str, source: &str) -> Vec<Chunk> {
    let opts = ChunkOptions {
        target_min_tokens: 150,
        target_max_tokens: 500,
        hard_max_tokens: 1_000,
        overlap_tokens: 0,
    };
    let lines: Vec<&str> = source.lines().collect();
    let mut chunks = Vec::new();
    let mut units = Vec::new();
    let mut depth = 0isize;
    let mut unit_start = 0usize;
    let mut pending_comment_start: Option<usize> = None;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if depth == 0 && is_doc_comment(trimmed) && pending_comment_start.is_none() {
            pending_comment_start = Some(i);
        }
        if depth == 0 && looks_like_code_declaration(trimmed) {
            if i > unit_start {
                units.push((unit_start, i));
            }
            unit_start = pending_comment_start.take().unwrap_or(i);
        }
        depth += brace_delta_ignoring_simple_strings(line);
        if depth < 0 {
            depth = 0;
        }
    }

    if unit_start < lines.len() {
        units.push((unit_start, lines.len()));
    }

    for (start, end) in units {
        let body = line_range_text(&lines, start, end);
        let title = infer_code_title(&body);
        if estimate_tokens(&body) <= opts.target_max_tokens {
            chunks.push(make_chunk(
                file_path,
                language,
                "code_symbol",
                title,
                vec![],
                start + 1,
                end,
                body,
            ));
        } else {
            chunks.extend(split_large_code_unit(
                file_path,
                language,
                title,
                &body,
                start + 1,
                &opts,
            ));
        }
    }
    chunks
}

pub fn process_js_ts_family(file_path: &str, source: &str) -> Vec<Chunk> {
    let language = match file_path.rsplit('.').next().unwrap_or("") {
        "ts" => "typescript",
        "tsx" => "tsx",
        "jsx" => "jsx",
        _ => "javascript",
    };
    let opts = ChunkOptions {
        target_min_tokens: 150,
        target_max_tokens: 550,
        hard_max_tokens: 1_100,
        overlap_tokens: 0,
    };
    let lines: Vec<&str> = source.lines().collect();
    let mut chunks = Vec::new();
    let mut units = Vec::new();
    let mut start = 0usize;
    let mut pending_doc_start: Option<usize> = None;

    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if t.starts_with("/**") || t.starts_with("/*") || t.starts_with("//") {
            pending_doc_start.get_or_insert(i);
        }
        if is_js_ts_boundary(t) {
            if i > start {
                units.push((start, i));
            }
            start = pending_doc_start.take().unwrap_or(i);
        }
    }
    if start < lines.len() {
        units.push((start, lines.len()));
    }

    for (start, end) in units {
        let body = line_range_text(&lines, start, end);
        let title = infer_js_ts_title(&body);
        if estimate_tokens(&body) <= opts.target_max_tokens {
            chunks.push(make_chunk(
                file_path,
                language,
                "code_symbol",
                title,
                vec![],
                start + 1,
                end,
                body,
            ));
        } else {
            chunks.extend(split_large_js_ts_unit(
                file_path,
                language,
                title,
                &body,
                start + 1,
                &opts,
            ));
        }
    }
    chunks
}

pub fn process_gdscript(file_path: &str, source: &str) -> Vec<Chunk> {
    let opts = ChunkOptions {
        target_min_tokens: 150,
        target_max_tokens: 500,
        hard_max_tokens: 1_000,
        overlap_tokens: 0,
    };
    let lines: Vec<&str> = source.lines().collect();
    let mut file_context = Vec::new();
    for line in &lines {
        let t = line.trim();
        if t.starts_with("extends ")
            || t.starts_with("class_name ")
            || t.starts_with("signal ")
            || t.starts_with("@export")
            || t.starts_with("export ")
            || t.starts_with("@onready")
        {
            file_context.push(t.to_string());
        }
    }

    let mut units = Vec::new();
    let mut start = 0usize;
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim_start();
        if t.starts_with("func ") || t.starts_with("static func ") {
            if i > start {
                units.push((start, i));
            }
            start = attach_leading_comments(&lines, i);
        }
    }
    if start < lines.len() {
        units.push((start, lines.len()));
    }

    let mut chunks = Vec::new();
    for (start, end) in units {
        let mut body = String::new();
        if !file_context.is_empty() {
            body.push_str("# File context\n");
            body.push_str(&file_context.join("\n"));
            body.push_str("\n\n");
        }
        body.push_str(&line_range_text(&lines, start, end));
        let title = infer_gdscript_title(&body);
        if estimate_tokens(&body) <= opts.target_max_tokens {
            chunks.push(make_chunk(
                file_path,
                "gdscript",
                "code_symbol",
                title,
                vec![],
                start + 1,
                end,
                body,
            ));
        } else {
            chunks.extend(split_large_code_unit(
                file_path,
                "gdscript",
                title,
                &body,
                start + 1,
                &opts,
            ));
        }
    }
    chunks
}

pub fn process_rust(file_path: &str, source: &str) -> Vec<Chunk> {
    let opts = ChunkOptions {
        target_min_tokens: 150,
        target_max_tokens: 550,
        hard_max_tokens: 1_100,
        overlap_tokens: 0,
    };
    let lines: Vec<&str> = source.lines().collect();
    let mut chunks = Vec::new();
    let mut units = Vec::new();
    let mut start = 0usize;
    let mut pending_attr_start: Option<usize> = None;
    let mut depth = 0isize;

    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if depth == 0 && (t.starts_with("#[") || t.starts_with("///") || t.starts_with("//!")) {
            pending_attr_start.get_or_insert(i);
        }
        if depth == 0 && is_rust_item_boundary(t) {
            if i > start {
                units.push((start, i));
            }
            start = pending_attr_start.take().unwrap_or(i);
        }
        depth += brace_delta_ignoring_simple_strings(line);
        if depth < 0 {
            depth = 0;
        }
    }

    if start < lines.len() {
        units.push((start, lines.len()));
    }

    for (start, end) in units {
        let body = line_range_text(&lines, start, end);
        let title = infer_rust_title(&body);
        if body.trim_start().starts_with("impl ") && estimate_tokens(&body) > opts.target_max_tokens
        {
            chunks.extend(split_rust_impl(file_path, &body, start + 1, title, &opts));
            continue;
        }
        chunks.push(make_chunk(
            file_path,
            "rust",
            "code_symbol",
            title,
            vec![],
            start + 1,
            end,
            body,
        ));
    }
    chunks
}

pub fn process_html(file_path: &str, source: &str) -> Vec<Chunk> {
    let opts = ChunkOptions {
        target_min_tokens: 200,
        target_max_tokens: 700,
        hard_max_tokens: 1_200,
        overlap_tokens: 0,
    };
    let lines: Vec<&str> = source.lines().collect();
    let mut chunks = Vec::new();
    let tags = [
        "main", "section", "article", "nav", "form", "template", "header", "footer",
    ];
    let mut starts = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim_start();
        if tags
            .iter()
            .any(|tag| t.starts_with(&format!("<{tag}")) || t.starts_with(&format!("<{tag} ")))
        {
            starts.push(i);
        }
        if t.starts_with("<script") || t.starts_with("<style") {
            starts.push(i);
        }
    }
    starts.sort_unstable();
    starts.dedup();
    if starts.is_empty() {
        return split_plain_blocks(file_path, "html", "html_block", vec![], source, &opts);
    }
    starts.push(lines.len());
    for pair in starts.windows(2) {
        let start = pair[0];
        let end = pair[1];
        let text = line_range_text(&lines, start, end);
        let title = infer_html_title(&text);
        if text.trim_start().starts_with("<script") {
            chunks.extend(process_js_ts_family(file_path, &strip_html_wrapper(&text)));
            continue;
        }
        if text.trim_start().starts_with("<style") {
            chunks.extend(process_css(file_path, &strip_html_wrapper(&text)));
            continue;
        }
        if estimate_tokens(&text) <= opts.target_max_tokens {
            chunks.push(make_chunk(
                file_path,
                "html",
                "html_subtree",
                title,
                vec![],
                start + 1,
                end,
                text,
            ));
        } else {
            chunks.extend(split_plain_blocks(
                file_path,
                "html",
                "html_subtree",
                vec![],
                &text,
                &opts,
            ));
        }
    }
    chunks
}

pub fn process_css(file_path: &str, source: &str) -> Vec<Chunk> {
    let opts = ChunkOptions {
        target_min_tokens: 150,
        target_max_tokens: 600,
        hard_max_tokens: 1_000,
        overlap_tokens: 0,
    };
    let lines: Vec<&str> = source.lines().collect();
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut depth = 0isize;
    for (i, line) in lines.iter().enumerate() {
        let before = depth;
        depth += brace_delta_ignoring_simple_strings(line);
        if before > 0 && depth == 0 {
            let end = i + 1;
            let text = line_range_text(&lines, start, end);
            if !text.trim().is_empty() {
                chunks.push(make_chunk(
                    file_path,
                    "css",
                    "css_rule_group",
                    infer_css_title(&text),
                    vec![],
                    start + 1,
                    end,
                    text,
                ));
            }
            start = end;
        }
        if depth < 0 {
            depth = 0;
        }
    }
    if start < lines.len() {
        let text = line_range_text(&lines, start, lines.len());
        if !text.trim().is_empty() {
            chunks.push(make_chunk(
                file_path,
                "css",
                "css_rule_group",
                infer_css_title(&text),
                vec![],
                start + 1,
                lines.len(),
                text,
            ));
        }
    }
    pack_small_css_chunks(chunks, &opts)
}

pub fn process_json(file_path: &str, source: &str) -> Vec<Chunk> {
    let value: Value = match serde_json::from_str(source) {
        Ok(value) => value,
        Err(_) => {
            return split_plain_blocks(
                file_path,
                "json",
                "json_invalid_text",
                vec![],
                source,
                &ChunkOptions::default(),
            );
        }
    };
    let opts = ChunkOptions {
        target_min_tokens: 100,
        target_max_tokens: 600,
        hard_max_tokens: 1_000,
        overlap_tokens: 0,
    };
    let mut chunks = Vec::new();
    chunk_json_value(file_path, "$", &value, &opts, &mut chunks);
    chunks
}

pub fn process_toml_like(file_path: &str, source: &str) -> Vec<Chunk> {
    let opts = ChunkOptions {
        target_min_tokens: 100,
        target_max_tokens: 600,
        hard_max_tokens: 1_000,
        overlap_tokens: 0,
    };
    let lines: Vec<&str> = source.lines().collect();
    let mut chunks = Vec::new();
    let mut sections = Vec::new();
    let mut start = 0usize;
    let mut pending_comment_start: Option<usize> = None;
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if t.starts_with('#') && pending_comment_start.is_none() {
            pending_comment_start = Some(i);
        }
        if is_toml_table_header(t) {
            if i > start {
                sections.push((start, i));
            }
            start = pending_comment_start.take().unwrap_or(i);
        }
        if !t.starts_with('#') && !t.is_empty() {
            pending_comment_start = None;
        }
    }
    if start < lines.len() {
        sections.push((start, lines.len()));
    }
    for (start, end) in sections {
        let text = line_range_text(&lines, start, end);
        let title = infer_toml_title(&text).unwrap_or_else(|| "top-level".to_string());
        if estimate_tokens(&text) <= opts.target_max_tokens {
            chunks.push(make_chunk(
                file_path,
                "toml",
                "toml_table",
                Some(title.clone()),
                vec![title],
                start + 1,
                end,
                text,
            ));
        } else {
            chunks.extend(split_plain_blocks(
                file_path,
                "toml",
                "toml_table",
                vec![title],
                &text,
                &opts,
            ));
        }
    }
    chunks
}

pub fn process_yaml(file_path: &str, source: &str) -> Vec<Chunk> {
    if source.lines().any(|line| line.trim() == "---") {
        return process_multi_document_yaml(file_path, source);
    }
    if source.contains("services:") {
        return process_compose_yaml(file_path, source);
    }
    if source.contains("jobs:") && source.contains("on:") {
        return process_github_actions_yaml(file_path, source);
    }
    process_generic_yaml(file_path, source)
}

pub fn process_python(file_path: &str, source: &str) -> Vec<Chunk> {
    if source.contains("# %%") {
        return process_python_cells(file_path, source);
    }
    let opts = ChunkOptions {
        target_min_tokens: 150,
        target_max_tokens: 550,
        hard_max_tokens: 1_100,
        overlap_tokens: 0,
    };
    let lines: Vec<&str> = source.lines().collect();
    let mut chunks = Vec::new();
    let mut starts = Vec::new();
    let mut pending_decorator_start: Option<usize> = None;
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim_start();
        if t.starts_with('@') {
            pending_decorator_start.get_or_insert(i);
        }
        if is_python_boundary(t) && !line.starts_with("    ") && !line.starts_with('\t') {
            starts.push(pending_decorator_start.take().unwrap_or(i));
        }
        if !t.starts_with('@') && !t.is_empty() {
            pending_decorator_start = None;
        }
    }
    starts.sort_unstable();
    starts.dedup();
    if starts.is_empty() {
        return split_plain_blocks(file_path, "python", "python_module", vec![], source, &opts);
    }
    starts.push(lines.len());
    for pair in starts.windows(2) {
        let start = pair[0];
        let end = pair[1];
        let text = line_range_text(&lines, start, end);
        let title = infer_python_title(&text);
        if text.trim_start().starts_with("class ")
            && estimate_tokens(&text) > opts.target_max_tokens
        {
            chunks.extend(split_python_class(
                file_path,
                &text,
                start + 1,
                title,
                &opts,
            ));
        } else {
            chunks.push(make_chunk(
                file_path,
                "python",
                "code_symbol",
                title,
                vec![],
                start + 1,
                end,
                text,
            ));
        }
    }
    chunks
}

pub fn process_go(file_path: &str, source: &str) -> Vec<Chunk> {
    let opts = ChunkOptions {
        target_min_tokens: 150,
        target_max_tokens: 550,
        hard_max_tokens: 1_100,
        overlap_tokens: 0,
    };
    let lines: Vec<&str> = source.lines().collect();
    let mut chunks = Vec::new();
    let mut starts = Vec::new();
    let mut pending_comment_start: Option<usize> = None;
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if t.starts_with("//") || t.starts_with("/*") {
            pending_comment_start.get_or_insert(i);
        }
        if is_go_boundary(t) {
            starts.push(pending_comment_start.take().unwrap_or(i));
        }
        if !t.starts_with("//") && !t.starts_with("/*") && !t.is_empty() {
            pending_comment_start = None;
        }
    }
    starts.sort_unstable();
    starts.dedup();
    if starts.is_empty() {
        return split_plain_blocks(file_path, "go", "go_file", vec![], source, &opts);
    }
    starts.push(lines.len());
    for pair in starts.windows(2) {
        let start = pair[0];
        let end = pair[1];
        let text = line_range_text(&lines, start, end);
        chunks.push(make_chunk(
            file_path,
            "go",
            "code_symbol",
            infer_go_title(&text),
            vec![],
            start + 1,
            end,
            text,
        ));
    }
    chunks
}

pub fn process_zig(file_path: &str, source: &str) -> Vec<Chunk> {
    let opts = ChunkOptions {
        target_min_tokens: 150,
        target_max_tokens: 550,
        hard_max_tokens: 1_100,
        overlap_tokens: 0,
    };
    let lines: Vec<&str> = source.lines().collect();
    let mut chunks = Vec::new();
    let mut starts = Vec::new();
    let mut depth = 0isize;
    let mut pending_comment_start: Option<usize> = None;
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if depth == 0 && t.starts_with("///") {
            pending_comment_start.get_or_insert(i);
        }
        if depth == 0 && is_zig_boundary(t) {
            starts.push(pending_comment_start.take().unwrap_or(i));
        }
        depth += brace_delta_ignoring_simple_strings(line);
        if depth < 0 {
            depth = 0;
        }
    }
    starts.sort_unstable();
    starts.dedup();
    if starts.is_empty() {
        return split_plain_blocks(file_path, "zig", "zig_file", vec![], source, &opts);
    }
    starts.push(lines.len());
    for pair in starts.windows(2) {
        let start = pair[0];
        let end = pair[1];
        let text = line_range_text(&lines, start, end);
        if estimate_tokens(&text) > opts.hard_max_tokens {
            chunks.extend(split_large_code_unit(
                file_path,
                "zig",
                infer_zig_title(&text),
                &text,
                start + 1,
                &opts,
            ));
        } else {
            chunks.push(make_chunk(
                file_path,
                "zig",
                "code_symbol",
                infer_zig_title(&text),
                vec![],
                start + 1,
                end,
                text,
            ));
        }
    }
    chunks
}

pub fn process_txt(file_path: &str, source: &str) -> Vec<Chunk> {
    let opts = ChunkOptions {
        target_min_tokens: 400,
        target_max_tokens: 900,
        hard_max_tokens: 1_500,
        overlap_tokens: 100,
    };
    let lines: Vec<&str> = source.lines().collect();
    let mut starts = Vec::new();
    for (i, _) in lines.iter().enumerate() {
        if looks_like_text_heading(&lines, i) {
            starts.push(i);
        }
    }
    if starts.is_empty() {
        return split_plain_blocks(file_path, "text", "plain_text", vec![], source, &opts);
    }
    starts.push(lines.len());
    let mut chunks = Vec::new();
    for pair in starts.windows(2) {
        let start = pair[0];
        let end = pair[1];
        let text = line_range_text(&lines, start, end);
        let title = lines[start].trim().to_string();
        if estimate_tokens(&text) <= opts.target_max_tokens {
            chunks.push(make_chunk(
                file_path,
                "text",
                "plain_text_section",
                Some(title.clone()),
                vec![title],
                start + 1,
                end,
                text,
            ));
        } else {
            chunks.extend(split_plain_blocks(
                file_path,
                "text",
                "plain_text_section",
                vec![title],
                &text,
                &opts,
            ));
        }
    }
    chunks
}

pub fn process_docx(file_path: &str, bytes: &[u8]) -> Vec<Chunk> {
    let xml = match extract_docx_document_xml(bytes) {
        Ok(xml) => xml,
        Err(_) => return Vec::new(),
    };
    let blocks = parse_docx_blocks_from_xml(&xml);
    let opts = ChunkOptions {
        target_min_tokens: 400,
        target_max_tokens: 900,
        hard_max_tokens: 1_500,
        overlap_tokens: 100,
    };
    let mut chunks = Vec::new();
    let mut heading_stack: Vec<(usize, String)> = Vec::new();
    let mut buf = String::new();
    let mut start_block = 0usize;
    let mut current_breadcrumb = Vec::new();

    for (i, block) in blocks.iter().enumerate() {
        match block {
            DocxBlock::Heading { level, text } => {
                if !buf.trim().is_empty() {
                    chunks.extend(chunk_docx_buffer(
                        file_path,
                        &buf,
                        start_block + 1,
                        i + 1,
                        current_breadcrumb.clone(),
                        &opts,
                    ));
                    buf.clear();
                }
                while heading_stack
                    .last()
                    .map(|(existing_level, _)| *existing_level >= *level)
                    .unwrap_or(false)
                {
                    heading_stack.pop();
                }
                heading_stack.push((*level, text.clone()));
                current_breadcrumb = heading_stack.iter().map(|(_, t)| t.clone()).collect();
                buf.push_str(text);
                buf.push_str("\n\n");
                start_block = i;
            }
            DocxBlock::Paragraph(text) => {
                buf.push_str(text);
                buf.push_str("\n\n");
            }
            DocxBlock::Table(text) => {
                if estimate_tokens(&buf) > opts.target_max_tokens {
                    chunks.extend(chunk_docx_buffer(
                        file_path,
                        &buf,
                        start_block + 1,
                        i + 1,
                        current_breadcrumb.clone(),
                        &opts,
                    ));
                    buf.clear();
                    start_block = i;
                }
                chunks.push(make_chunk(
                    file_path,
                    "docx",
                    "docx_table",
                    current_breadcrumb.last().cloned(),
                    current_breadcrumb.clone(),
                    i + 1,
                    i + 1,
                    text.clone(),
                ));
            }
        }
    }
    if !buf.trim().is_empty() {
        chunks.extend(chunk_docx_buffer(
            file_path,
            &buf,
            start_block + 1,
            blocks.len(),
            current_breadcrumb,
            &opts,
        ));
    }
    chunks
}

fn split_plain_blocks(
    file_path: &str,
    language: &str,
    chunk_type: &str,
    breadcrumb: Vec<String>,
    source: &str,
    opts: &ChunkOptions,
) -> Vec<Chunk> {
    let lines: Vec<&str> = source.lines().collect();
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut buf = String::new();
    for (i, line) in lines.iter().enumerate() {
        let would_exceed = estimate_tokens(&buf) > opts.target_max_tokens;
        let boundary = line.trim().is_empty();
        if would_exceed && boundary && !buf.trim().is_empty() {
            chunks.push(make_chunk(
                file_path,
                language,
                chunk_type,
                breadcrumb.last().cloned(),
                breadcrumb.clone(),
                start + 1,
                i + 1,
                buf.trim().to_string(),
            ));
            start = i + 1;
            buf.clear();
        } else {
            buf.push_str(line);
            buf.push('\n');
        }
    }
    if !buf.trim().is_empty() {
        chunks.push(make_chunk(
            file_path,
            language,
            chunk_type,
            breadcrumb.last().cloned(),
            breadcrumb,
            start + 1,
            lines.len(),
            buf.trim().to_string(),
        ));
    }
    chunks
}

#[derive(Debug)]
struct MarkdownBlock {
    text: String,
    line_count: usize,
}

fn parse_markdown_heading(line: &str) -> Option<(usize, String)> {
    let hashes = line.chars().take_while(|c| *c == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = line[hashes..].trim();
    if rest.is_empty() {
        return None;
    }
    Some((hashes, rest.trim_matches('#').trim().to_string()))
}

fn chunk_markdown_section(
    file_path: &str,
    section: &str,
    start_line: usize,
    end_line: usize,
    breadcrumb: Vec<String>,
    opts: &ChunkOptions,
) -> Vec<Chunk> {
    if estimate_tokens(section) <= opts.target_max_tokens {
        return vec![make_chunk(
            file_path,
            "markdown",
            "markdown_section",
            breadcrumb.last().cloned(),
            breadcrumb,
            start_line,
            end_line,
            section.trim().to_string(),
        )];
    }
    let blocks = split_markdown_blocks(section);
    let mut chunks = Vec::new();
    let mut buf = String::new();
    let mut chunk_start_offset = 0usize;
    let mut current_line_offset = 0usize;
    for block in blocks {
        let block_tokens = estimate_tokens(&block.text);
        let buf_tokens = estimate_tokens(&buf);
        if buf_tokens > 0 && buf_tokens + block_tokens > opts.target_max_tokens {
            chunks.push(make_chunk(
                file_path,
                "markdown",
                "markdown_section",
                breadcrumb.last().cloned(),
                breadcrumb.clone(),
                start_line + chunk_start_offset,
                start_line + current_line_offset,
                buf.trim().to_string(),
            ));
            buf.clear();
            chunk_start_offset = current_line_offset;
        }
        buf.push_str(&block.text);
        buf.push_str("\n\n");
        current_line_offset += block.line_count;
    }
    if !buf.trim().is_empty() {
        chunks.push(make_chunk(
            file_path,
            "markdown",
            "markdown_section",
            breadcrumb.last().cloned(),
            breadcrumb,
            start_line + chunk_start_offset,
            end_line,
            buf.trim().to_string(),
        ));
    }
    chunks
}

fn split_markdown_blocks(section: &str) -> Vec<MarkdownBlock> {
    let lines: Vec<&str> = section.lines().collect();
    let mut blocks = Vec::new();
    let mut i = 0usize;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            let fence = &trimmed[..3];
            let start = i;
            i += 1;
            while i < lines.len() && !lines[i].trim().starts_with(fence) {
                i += 1;
            }
            if i < lines.len() {
                i += 1;
            }
            blocks.push(MarkdownBlock {
                text: lines[start..i].join("\n"),
                line_count: i - start,
            });
            continue;
        }
        if is_markdown_table_start(&lines, i) {
            let start = i;
            i += 2;
            while i < lines.len() && lines[i].trim().starts_with('|') {
                i += 1;
            }
            blocks.push(MarkdownBlock {
                text: lines[start..i].join("\n"),
                line_count: i - start,
            });
            continue;
        }
        if trimmed.starts_with("- ")
            || trimmed.starts_with("* ")
            || trimmed.starts_with("+ ")
            || starts_ordered_list(trimmed)
        {
            let start = i;
            i += 1;
            while i < lines.len() {
                let t = lines[i].trim();
                if t.is_empty()
                    || t.starts_with("- ")
                    || t.starts_with("* ")
                    || t.starts_with("+ ")
                    || starts_ordered_list(t)
                    || lines[i].starts_with("  ")
                    || lines[i].starts_with('\t')
                {
                    i += 1;
                } else {
                    break;
                }
            }
            blocks.push(MarkdownBlock {
                text: lines[start..i].join("\n"),
                line_count: i - start,
            });
            continue;
        }
        let start = i;
        i += 1;
        while i < lines.len() && !lines[i].trim().is_empty() {
            i += 1;
        }
        if i < lines.len() {
            i += 1;
        }
        blocks.push(MarkdownBlock {
            text: lines[start..i].join("\n"),
            line_count: i - start,
        });
    }
    blocks
}

fn is_markdown_table_start(lines: &[&str], i: usize) -> bool {
    if i + 1 >= lines.len() {
        return false;
    }
    lines[i].trim().starts_with('|')
        && lines[i + 1]
            .trim()
            .chars()
            .all(|c| c == '|' || c == '-' || c == ':' || c == ' ')
}

fn starts_ordered_list(line: &str) -> bool {
    let mut seen_digit = false;
    for c in line.chars() {
        if c.is_ascii_digit() {
            seen_digit = true;
            continue;
        }
        return seen_digit && c == '.';
    }
    false
}

fn looks_like_code_declaration(line: &str) -> bool {
    let prefixes = [
        "function ",
        "export function ",
        "export async function ",
        "async function ",
        "class ",
        "export class ",
        "interface ",
        "type ",
        "enum ",
        "struct ",
        "trait ",
        "impl ",
        "fn ",
        "pub fn ",
        "def ",
        "func ",
        "pub ",
        "const ",
    ];
    prefixes.iter().any(|prefix| line.starts_with(prefix))
}

fn is_doc_comment(line: &str) -> bool {
    line.starts_with("///")
        || line.starts_with("//!")
        || line.starts_with("/**")
        || line.starts_with('*')
        || line.starts_with('#')
}

fn brace_delta_ignoring_simple_strings(line: &str) -> isize {
    let mut delta = 0isize;
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    for c in line.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if c == '\\' {
            escaped = true;
            continue;
        }
        match c {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '{' if !in_single && !in_double => delta += 1,
            '}' if !in_single && !in_double => delta -= 1,
            _ => {}
        }
    }
    delta
}

fn infer_code_title(body: &str) -> Option<String> {
    for line in body.lines() {
        let t = line.trim();
        for prefix in [
            "function ",
            "class ",
            "interface ",
            "type ",
            "enum ",
            "struct ",
            "trait ",
            "impl ",
            "fn ",
            "def ",
            "func ",
            "pub fn ",
            "export function ",
            "export class ",
        ] {
            if let Some(rest) = t.strip_prefix(prefix) {
                return Some(
                    rest.split(|c: char| c == '(' || c == '<' || c == ':' || c.is_whitespace())
                        .next()
                        .unwrap_or(rest)
                        .trim_end_matches('{')
                        .trim_end_matches('!')
                        .to_string(),
                );
            }
        }
    }
    None
}

fn split_large_code_unit(
    file_path: &str,
    language: &str,
    title: Option<String>,
    body: &str,
    start_line: usize,
    opts: &ChunkOptions,
) -> Vec<Chunk> {
    let lines: Vec<&str> = body.lines().collect();
    let mut chunks = Vec::new();
    let mut buf = String::new();
    let mut chunk_start = 0usize;
    for (i, line) in lines.iter().enumerate() {
        let boundary = line.trim().is_empty();
        if boundary && estimate_tokens(&buf) > opts.target_max_tokens {
            chunks.push(make_chunk(
                file_path,
                language,
                "code_block",
                title.clone(),
                vec![],
                start_line + chunk_start,
                start_line + i,
                buf.trim().to_string(),
            ));
            buf.clear();
            chunk_start = i + 1;
        } else {
            buf.push_str(line);
            buf.push('\n');
        }
    }
    if !buf.trim().is_empty() {
        chunks.push(make_chunk(
            file_path,
            language,
            "code_block",
            title,
            vec![],
            start_line + chunk_start,
            start_line + lines.len(),
            buf.trim().to_string(),
        ));
    }
    chunks
}

fn is_js_ts_boundary(line: &str) -> bool {
    line.starts_with("export function ")
        || line.starts_with("export async function ")
        || line.starts_with("function ")
        || line.starts_with("async function ")
        || line.starts_with("export const ")
        || line.starts_with("const use")
        || line.starts_with("const ")
        || line.starts_with("let ")
        || line.starts_with("export class ")
        || line.starts_with("class ")
        || line.starts_with("export interface ")
        || line.starts_with("interface ")
        || line.starts_with("export type ")
        || line.starts_with("type ")
        || line.starts_with("export enum ")
        || line.starts_with("enum ")
        || line.starts_with("describe(")
        || line.starts_with("test(")
        || line.starts_with("it(")
}

fn infer_js_ts_title(body: &str) -> Option<String> {
    for line in body.lines() {
        let t = line.trim();
        for prefix in [
            "export async function ",
            "export function ",
            "async function ",
            "function ",
            "export class ",
            "class ",
            "export interface ",
            "interface ",
            "export type ",
            "type ",
            "export enum ",
            "enum ",
            "export const ",
            "const ",
            "let ",
        ] {
            if let Some(rest) = t.strip_prefix(prefix) {
                return Some(
                    rest.split(|c: char| {
                        c == '('
                            || c == '<'
                            || c == ':'
                            || c == '='
                            || c == '{'
                            || c.is_whitespace()
                    })
                    .next()
                    .unwrap_or(rest)
                    .trim_end_matches(';')
                    .to_string(),
                );
            }
        }
    }
    None
}

fn split_large_js_ts_unit(
    file_path: &str,
    language: &str,
    title: Option<String>,
    body: &str,
    start_line: usize,
    opts: &ChunkOptions,
) -> Vec<Chunk> {
    let split_markers = [
        "useEffect(",
        "useMemo(",
        "useCallback(",
        "function ",
        "const handle",
        "const render",
        "return (",
        "return <",
    ];
    split_by_markers(
        file_path,
        language,
        "code_block",
        title,
        body,
        start_line,
        opts,
        &split_markers,
    )
}

#[allow(clippy::too_many_arguments)]
fn split_by_markers(
    file_path: &str,
    language: &str,
    chunk_type: &str,
    title: Option<String>,
    body: &str,
    start_line: usize,
    opts: &ChunkOptions,
    markers: &[&str],
) -> Vec<Chunk> {
    let lines: Vec<&str> = body.lines().collect();
    let mut cuts = vec![0usize];
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if i > 0
            && markers
                .iter()
                .any(|marker| t.starts_with(marker) || t.contains(marker))
        {
            cuts.push(i);
        }
    }
    cuts.push(lines.len());
    cuts.sort_unstable();
    cuts.dedup();
    let mut chunks = Vec::new();
    for pair in cuts.windows(2) {
        let start = pair[0];
        let end = pair[1];
        if start >= end {
            continue;
        }
        let text = line_range_text(&lines, start, end);
        if estimate_tokens(&text) > opts.hard_max_tokens {
            chunks.extend(split_large_code_unit(
                file_path,
                language,
                title.clone(),
                &text,
                start_line + start,
                opts,
            ));
        } else if !text.trim().is_empty() {
            chunks.push(make_chunk(
                file_path,
                language,
                chunk_type,
                title.clone(),
                vec![],
                start_line + start,
                start_line + end,
                text,
            ));
        }
    }
    chunks
}

fn attach_leading_comments(lines: &[&str], i: usize) -> usize {
    let mut start = i;
    while start > 0 {
        let prev = lines[start - 1].trim();
        if prev.starts_with('#') || prev.is_empty() {
            start -= 1;
        } else {
            break;
        }
    }
    start
}

fn infer_gdscript_title(body: &str) -> Option<String> {
    for line in body.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("func ") {
            return Some(rest.split('(').next().unwrap_or(rest).to_string());
        }
        if let Some(rest) = t.strip_prefix("static func ") {
            return Some(rest.split('(').next().unwrap_or(rest).to_string());
        }
    }
    None
}

fn is_rust_item_boundary(line: &str) -> bool {
    line.starts_with("pub struct ")
        || line.starts_with("struct ")
        || line.starts_with("pub enum ")
        || line.starts_with("enum ")
        || line.starts_with("pub trait ")
        || line.starts_with("trait ")
        || line.starts_with("impl ")
        || line.starts_with("pub fn ")
        || line.starts_with("fn ")
        || line.starts_with("pub mod ")
        || line.starts_with("mod ")
        || line.starts_with("macro_rules!")
}

fn infer_rust_title(body: &str) -> Option<String> {
    for line in body.lines() {
        let t = line.trim();
        for prefix in [
            "pub struct ",
            "struct ",
            "pub enum ",
            "enum ",
            "pub trait ",
            "trait ",
            "impl ",
            "pub fn ",
            "fn ",
            "pub mod ",
            "mod ",
            "macro_rules! ",
        ] {
            if let Some(rest) = t.strip_prefix(prefix) {
                return Some(
                    rest.split(|c: char| {
                        c == '<' || c == '(' || c == '{' || c == ':' || c.is_whitespace()
                    })
                    .next()
                    .unwrap_or(rest)
                    .trim_end_matches('!')
                    .to_string(),
                );
            }
        }
    }
    None
}

fn split_rust_impl(
    file_path: &str,
    body: &str,
    start_line: usize,
    impl_title: Option<String>,
    opts: &ChunkOptions,
) -> Vec<Chunk> {
    let lines: Vec<&str> = body.lines().collect();
    let mut chunks = Vec::new();
    let mut method_starts = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if t.starts_with("pub fn ") || t.starts_with("fn ") || t.starts_with("async fn ") {
            method_starts.push(i);
        }
    }
    if method_starts.is_empty() {
        return split_large_code_unit(file_path, "rust", impl_title, body, start_line, opts);
    }
    method_starts.push(lines.len());
    for pair in method_starts.windows(2) {
        let start = pair[0];
        let end = pair[1];
        let mut text = String::new();
        if let Some(title) = &impl_title {
            text.push_str(&format!("// Parent impl: {title}\n\n"));
        }
        text.push_str(&line_range_text(&lines, start, end));
        chunks.push(make_chunk(
            file_path,
            "rust",
            "code_symbol",
            infer_rust_title(&text).or_else(|| impl_title.clone()),
            vec![],
            start_line + start,
            start_line + end,
            text,
        ));
    }
    chunks
}

fn infer_html_title(text: &str) -> Option<String> {
    let first = text.lines().find(|line| !line.trim().is_empty())?.trim();
    let tag = first
        .trim_start_matches('<')
        .split([' ', '>', '/'])
        .next()
        .unwrap_or("html");
    let id = extract_attr(first, "id");
    let class = extract_attr(first, "class");
    match (id, class) {
        (Some(id), _) => Some(format!("{tag}#{id}")),
        (_, Some(class)) => Some(format!("{tag}.{class}")),
        _ => Some(tag.to_string()),
    }
}

fn extract_attr(line: &str, attr: &str) -> Option<String> {
    let needle = format!("{attr}=\"");
    let start = line.find(&needle)? + needle.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn strip_html_wrapper(text: &str) -> String {
    text.lines()
        .filter(|line| {
            let t = line.trim();
            !t.starts_with("<script")
                && !t.starts_with("</script")
                && !t.starts_with("<style")
                && !t.starts_with("</style")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn infer_css_title(text: &str) -> Option<String> {
    let first = text.lines().find(|line| !line.trim().is_empty())?.trim();
    Some(
        first
            .split('{')
            .next()
            .unwrap_or(first)
            .trim()
            .chars()
            .take(120)
            .collect(),
    )
}

fn pack_small_css_chunks(chunks: Vec<Chunk>, opts: &ChunkOptions) -> Vec<Chunk> {
    let mut packed = Vec::new();
    let mut current: Option<Chunk> = None;
    for chunk in chunks {
        if let Some(mut active) = current.take() {
            let combined_tokens = estimate_tokens(&active.text) + estimate_tokens(&chunk.text);
            if combined_tokens <= opts.target_max_tokens {
                active.end_line = chunk.end_line;
                active.text.push_str("\n\n");
                active.text.push_str(&chunk.text);
                current = Some(active);
            } else {
                packed.push(active);
                current = Some(chunk);
            }
        } else {
            current = Some(chunk);
        }
    }
    if let Some(active) = current {
        packed.push(active);
    }
    packed
}

fn chunk_json_value(
    file_path: &str,
    path: &str,
    value: &Value,
    opts: &ChunkOptions,
    chunks: &mut Vec<Chunk>,
) {
    let pretty = serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string());
    if estimate_tokens(&pretty) <= opts.target_max_tokens {
        let body = format!("JSON path: {path}\n\n{pretty}");
        chunks.push(make_chunk(
            file_path,
            "json",
            "json_object",
            Some(path.to_string()),
            vec![path.to_string()],
            1,
            1,
            body,
        ));
        return;
    }
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                let child_path = format!("{path}.{key}");
                chunk_json_value(file_path, &child_path, child, opts, chunks);
            }
        }
        Value::Array(items) => {
            for (i, child) in items.iter().enumerate() {
                let child_path = if let Some(id) = infer_json_id(child) {
                    format!("{path}[{i}]#{id}")
                } else {
                    format!("{path}[{i}]")
                };
                chunk_json_value(file_path, &child_path, child, opts, chunks);
            }
        }
        _ => {
            let body = format!("JSON path: {path}\n\n{pretty}");
            chunks.push(make_chunk(
                file_path,
                "json",
                "json_value",
                Some(path.to_string()),
                vec![path.to_string()],
                1,
                1,
                body,
            ));
        }
    }
}

fn infer_json_id(value: &Value) -> Option<String> {
    let obj = value.as_object()?;
    for key in ["id", "name", "slug", "key", "title"] {
        if let Some(Value::String(value)) = obj.get(key) {
            return Some(value.clone());
        }
    }
    None
}

fn is_toml_table_header(line: &str) -> bool {
    line.starts_with('[') && line.ends_with(']')
}

fn infer_toml_title(text: &str) -> Option<String> {
    for line in text.lines() {
        let t = line.trim();
        if is_toml_table_header(t) {
            return Some(t.trim_matches('[').trim_matches(']').to_string());
        }
    }
    None
}

fn process_generic_yaml(file_path: &str, source: &str) -> Vec<Chunk> {
    let opts = ChunkOptions {
        target_min_tokens: 100,
        target_max_tokens: 600,
        hard_max_tokens: 1_000,
        overlap_tokens: 0,
    };
    let lines: Vec<&str> = source.lines().collect();
    let mut chunks = Vec::new();
    let mut starts = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if is_yaml_top_level_key(line) {
            starts.push(i);
        }
    }
    if starts.is_empty() {
        return split_plain_blocks(file_path, "yaml", "yaml_document", vec![], source, &opts);
    }
    starts.push(lines.len());
    for pair in starts.windows(2) {
        let start = pair[0];
        let end = pair[1];
        let text = line_range_text(&lines, start, end);
        let title = lines[start]
            .split(':')
            .next()
            .unwrap_or("yaml")
            .trim()
            .to_string();
        chunks.push(make_chunk(
            file_path,
            "yaml",
            "yaml_object",
            Some(title.clone()),
            vec![title],
            start + 1,
            end,
            text,
        ));
    }
    chunks
}

fn process_multi_document_yaml(file_path: &str, source: &str) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut current = Vec::new();
    let mut start_line = 1usize;
    for (i, line) in source.lines().enumerate() {
        if line.trim() == "---" && !current.is_empty() {
            let text = current.join("\n");
            let title =
                infer_kubernetes_title(&text).unwrap_or_else(|| format!("document-{start_line}"));
            chunks.push(make_chunk(
                file_path,
                "yaml",
                "yaml_document",
                Some(title.clone()),
                vec![title],
                start_line,
                i,
                text,
            ));
            current.clear();
            start_line = i + 2;
        } else {
            current.push(line);
        }
    }
    if !current.is_empty() {
        let text = current.join("\n");
        let title =
            infer_kubernetes_title(&text).unwrap_or_else(|| format!("document-{start_line}"));
        chunks.push(make_chunk(
            file_path,
            "yaml",
            "yaml_document",
            Some(title.clone()),
            vec![title],
            start_line,
            source.lines().count(),
            text,
        ));
    }
    chunks
}

fn process_compose_yaml(file_path: &str, source: &str) -> Vec<Chunk> {
    chunk_yaml_nested_map(file_path, source, "services", "docker_compose_service")
}

fn process_github_actions_yaml(file_path: &str, source: &str) -> Vec<Chunk> {
    chunk_yaml_nested_map(file_path, source, "jobs", "github_actions_job")
}

fn chunk_yaml_nested_map(
    file_path: &str,
    source: &str,
    parent_key: &str,
    chunk_type: &str,
) -> Vec<Chunk> {
    let lines: Vec<&str> = source.lines().collect();
    let Some(parent_index) = lines
        .iter()
        .position(|line| line.trim() == format!("{parent_key}:"))
    else {
        return process_generic_yaml(file_path, source);
    };

    let mut chunks = Vec::new();
    let mut starts = Vec::new();
    for (i, line) in lines.iter().enumerate().skip(parent_index + 1) {
        if line.starts_with("  ") && !line.starts_with("    ") && line.trim_end().ends_with(':') {
            starts.push(i);
        }
    }
    starts.push(lines.len());
    for pair in starts.windows(2) {
        let start = pair[0];
        let end = pair[1];
        let name = lines[start].trim().trim_end_matches(':').to_string();
        let text = line_range_text(&lines, start, end);
        let path = format!("{parent_key}.{name}");
        chunks.push(make_chunk(
            file_path,
            "yaml",
            chunk_type,
            Some(path.clone()),
            vec![path],
            start + 1,
            end,
            text,
        ));
    }
    chunks
}

fn is_yaml_top_level_key(line: &str) -> bool {
    !line.starts_with(' ')
        && !line.starts_with('\t')
        && line.contains(':')
        && !line.trim_start().starts_with('#')
}

fn infer_kubernetes_title(text: &str) -> Option<String> {
    let mut kind = None;
    let mut name = None;
    for line in text.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("kind:") {
            kind = Some(rest.trim().to_string());
        }
        if let Some(rest) = t.strip_prefix("name:") {
            name = Some(rest.trim().to_string());
        }
    }
    match (kind, name) {
        (Some(kind), Some(name)) => Some(format!("{kind}/{name}")),
        (Some(kind), None) => Some(kind),
        _ => None,
    }
}

fn is_python_boundary(line: &str) -> bool {
    line.starts_with("def ") || line.starts_with("async def ") || line.starts_with("class ")
}

fn infer_python_title(text: &str) -> Option<String> {
    for line in text.lines() {
        let t = line.trim();
        for prefix in ["async def ", "def ", "class "] {
            if let Some(rest) = t.strip_prefix(prefix) {
                return Some(
                    rest.split(|c: char| c == '(' || c == ':' || c.is_whitespace())
                        .next()
                        .unwrap_or(rest)
                        .to_string(),
                );
            }
        }
    }
    None
}

fn split_python_class(
    file_path: &str,
    text: &str,
    start_line: usize,
    class_title: Option<String>,
    opts: &ChunkOptions,
) -> Vec<Chunk> {
    let lines: Vec<&str> = text.lines().collect();
    let mut starts = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim_start();
        if line.starts_with("    def ") || line.starts_with("    async def ") {
            starts.push(i);
        }
        if t.starts_with('@') && i + 1 < lines.len() {
            let next = lines[i + 1].trim_start();
            if next.starts_with("def ") || next.starts_with("async def ") {
                starts.push(i);
            }
        }
    }
    starts.sort_unstable();
    starts.dedup();
    if starts.is_empty() {
        return split_large_code_unit(file_path, "python", class_title, text, start_line, opts);
    }
    let class_header = lines
        .iter()
        .take_while(|line| !line.starts_with("    def ") && !line.starts_with("    async def "))
        .copied()
        .collect::<Vec<_>>()
        .join("\n");
    starts.push(lines.len());
    let mut chunks = Vec::new();
    for pair in starts.windows(2) {
        let start = pair[0];
        let end = pair[1];
        let mut body = String::new();
        body.push_str(&class_header);
        body.push_str("\n\n");
        body.push_str(&line_range_text(&lines, start, end));
        chunks.push(make_chunk(
            file_path,
            "python",
            "code_symbol",
            infer_python_title(&body).or_else(|| class_title.clone()),
            vec![],
            start_line + start,
            start_line + end,
            body,
        ));
    }
    chunks
}

fn process_python_cells(file_path: &str, source: &str) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut current = Vec::new();
    let mut start_line = 1usize;
    for (i, line) in source.lines().enumerate() {
        if line.trim() == "# %%" && !current.is_empty() {
            let text = current.join("\n");
            chunks.push(make_chunk(
                file_path,
                "python",
                "python_cell",
                Some(format!("cell-{start_line}")),
                vec![],
                start_line,
                i,
                text,
            ));
            current.clear();
            start_line = i + 1;
        }
        current.push(line);
    }
    if !current.is_empty() {
        chunks.push(make_chunk(
            file_path,
            "python",
            "python_cell",
            Some(format!("cell-{start_line}")),
            vec![],
            start_line,
            source.lines().count(),
            current.join("\n"),
        ));
    }
    chunks
}

fn is_go_boundary(line: &str) -> bool {
    line.starts_with("type ")
        || line.starts_with("func ")
        || line.starts_with("const ")
        || line.starts_with("var ")
}

fn infer_go_title(text: &str) -> Option<String> {
    for line in text.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("func ") {
            if rest.starts_with('(') {
                let after_receiver = rest.split(')').nth(1)?.trim();
                return Some(
                    after_receiver
                        .split('(')
                        .next()
                        .unwrap_or(after_receiver)
                        .trim()
                        .to_string(),
                );
            }
            return Some(rest.split('(').next().unwrap_or(rest).trim().to_string());
        }
        if let Some(rest) = t.strip_prefix("type ") {
            return Some(
                rest.split([' ', '\t', '{'])
                    .next()
                    .unwrap_or(rest)
                    .to_string(),
            );
        }
        if let Some(rest) = t.strip_prefix("const ") {
            return Some(format!(
                "const {}",
                rest.split('=').next().unwrap_or(rest).trim()
            ));
        }
        if let Some(rest) = t.strip_prefix("var ") {
            return Some(format!(
                "var {}",
                rest.split('=').next().unwrap_or(rest).trim()
            ));
        }
    }
    None
}

fn is_zig_boundary(line: &str) -> bool {
    line.starts_with("pub fn ")
        || line.starts_with("fn ")
        || line.starts_with("pub const ")
        || line.starts_with("const ")
        || line.starts_with("pub var ")
        || line.starts_with("var ")
        || line.starts_with("test ")
        || line.starts_with("comptime ")
}

fn infer_zig_title(text: &str) -> Option<String> {
    for line in text.lines() {
        let t = line.trim();
        for prefix in [
            "pub fn ",
            "fn ",
            "pub const ",
            "const ",
            "pub var ",
            "var ",
            "test ",
        ] {
            if let Some(rest) = t.strip_prefix(prefix) {
                return Some(
                    rest.split(|c: char| c == '(' || c == '=' || c == ':' || c.is_whitespace())
                        .next()
                        .unwrap_or(rest)
                        .trim_matches('"')
                        .to_string(),
                );
            }
        }
    }
    None
}

fn looks_like_text_heading(lines: &[&str], i: usize) -> bool {
    let line = lines[i].trim();
    if line.is_empty() {
        return false;
    }
    if i + 1 < lines.len() {
        let next = lines[i + 1].trim();
        if next.chars().all(|c| c == '-' || c == '=') && next.len() >= 3 {
            return true;
        }
    }
    if line.len() < 80
        && line.chars().all(|c| !c.is_lowercase())
        && line.chars().any(|c| c.is_alphabetic())
    {
        return true;
    }
    if starts_numbered_heading(line) {
        return true;
    }
    is_title_like(line)
}

fn starts_numbered_heading(line: &str) -> bool {
    let first = line.split_whitespace().next().unwrap_or("");
    first.chars().all(|c| c.is_ascii_digit() || c == '.') && first.contains('.')
}

fn is_title_like(line: &str) -> bool {
    let words: Vec<&str> = line.split_whitespace().collect();
    if words.is_empty() || words.len() > 10 {
        return false;
    }
    words.iter().all(|word| {
        word.chars()
            .next()
            .map(|c| c.is_uppercase() || c.is_ascii_digit())
            .unwrap_or(false)
    })
}

fn chunk_docx_buffer(
    file_path: &str,
    text: &str,
    start_line: usize,
    end_line: usize,
    breadcrumb: Vec<String>,
    opts: &ChunkOptions,
) -> Vec<Chunk> {
    if estimate_tokens(text) <= opts.target_max_tokens {
        return vec![make_chunk(
            file_path,
            "docx",
            "docx_section",
            breadcrumb.last().cloned(),
            breadcrumb,
            start_line,
            end_line,
            text.trim().to_string(),
        )];
    }
    split_plain_blocks(file_path, "docx", "docx_section", breadcrumb, text, opts)
}

fn extract_docx_document_xml(bytes: &[u8]) -> Result<String, Box<dyn std::error::Error>> {
    let reader = Cursor::new(bytes);
    let mut zip = ZipArchive::new(reader)?;
    let mut file = zip.by_name("word/document.xml")?;
    let mut xml = String::new();
    file.read_to_string(&mut xml)?;
    Ok(xml)
}

#[derive(Debug, Clone)]
enum DocxBlock {
    Heading { level: usize, text: String },
    Paragraph(String),
    Table(String),
}

fn parse_docx_blocks_from_xml(xml: &str) -> Vec<DocxBlock> {
    let mut blocks = Vec::new();
    for raw_para in xml.split("<w:p") {
        let text = extract_docx_text(raw_para);
        if text.trim().is_empty() {
            continue;
        }
        let level = infer_docx_heading_level(raw_para);
        if let Some(level) = level {
            blocks.push(DocxBlock::Heading {
                level,
                text: text.trim().to_string(),
            });
        } else {
            blocks.push(DocxBlock::Paragraph(text.trim().to_string()));
        }
    }
    for raw_table in xml.split("<w:tbl").skip(1) {
        let table_text = extract_docx_text(raw_table);
        if !table_text.trim().is_empty() {
            blocks.push(DocxBlock::Table(table_text));
        }
    }
    blocks
}

fn extract_docx_text(xml_fragment: &str) -> String {
    let mut out = String::new();
    let mut rest = xml_fragment;
    while let Some(start) = rest.find("<w:t") {
        rest = &rest[start..];
        let Some(close) = rest.find('>') else {
            break;
        };
        rest = &rest[close + 1..];
        let Some(end) = rest.find("</w:t>") else {
            break;
        };
        let text = &rest[..end];
        out.push_str(&decode_xml_entities(text));
        out.push(' ');
        rest = &rest[end + "</w:t>".len()..];
    }
    out
}

fn infer_docx_heading_level(xml_fragment: &str) -> Option<usize> {
    if xml_fragment.contains("Heading1") || xml_fragment.contains("heading 1") {
        return Some(1);
    }
    if xml_fragment.contains("Heading2") || xml_fragment.contains("heading 2") {
        return Some(2);
    }
    if xml_fragment.contains("Heading3") || xml_fragment.contains("heading 3") {
        return Some(3);
    }
    None
}

fn decode_xml_entities(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

fn chunk_window(lines: &[&str], start: usize, opts: &ChunkOptions) -> (usize, usize) {
    let mut end = start;
    let mut token_count = 0usize;
    let target = opts.target_min_tokens.max(1);
    let max_tokens = opts.target_max_tokens.max(target);
    let min_chars = 1usize;

    while end < lines.len() {
        let line_tokens = count_tokens(lines[end]);
        let would_exceed_max = end > start && token_count + line_tokens > max_tokens;
        if would_exceed_max {
            break;
        }
        token_count += line_tokens;
        end += 1;
        let chunk_chars = lines[start..end].join("\n").len();
        if token_count >= target && chunk_chars >= min_chars {
            break;
        }
    }

    if end == start {
        end = (start + 1).min(lines.len());
        token_count = count_tokens(lines[start]);
    }

    (end, token_count)
}

fn overlap_start(lines: &[&str], start: usize, end: usize, overlap_tokens: usize) -> usize {
    if overlap_tokens == 0 {
        return end;
    }

    let mut accumulated = 0usize;
    let mut overlap_start = end;
    while overlap_start > start {
        let candidate = overlap_start - 1;
        accumulated += count_tokens(lines[candidate]);
        overlap_start = candidate;
        if accumulated >= overlap_tokens {
            break;
        }
    }

    if overlap_start <= start {
        (start + 1).min(end)
    } else {
        overlap_start
    }
}

fn count_tokens(line: &str) -> usize {
    line.split_whitespace().count()
}

fn infer_code_language(file_path: &str) -> &'static str {
    let ext = file_path
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "rs" => "rust",
        "ts" => "typescript",
        "tsx" => "tsx",
        "js" => "javascript",
        "jsx" => "jsx",
        "py" => "python",
        "go" => "go",
        "gd" => "gdscript",
        "zig" => "zig",
        "html" | "htm" => "html",
        "css" => "css",
        "json" => "json",
        "toml" | "tmol" => "toml",
        "yml" | "yaml" => "yaml",
        _ => "code",
    }
}

fn looks_like_source_code(source: &str) -> bool {
    let code_markers = [
        "function ",
        "class ",
        "const ",
        "let ",
        "var ",
        "fn ",
        "def ",
        "func ",
        "import ",
        "use ",
        "package ",
        "{",
        "}",
        "=>",
    ];
    let hits = code_markers
        .iter()
        .filter(|marker| source.contains(**marker))
        .count();
    hits >= 3
}
