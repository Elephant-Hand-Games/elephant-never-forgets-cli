use anyhow::Result;

use crate::cli::SearchArgs;

pub fn run(args: SearchArgs, retrieve: bool) -> Result<()> {
    let config = crate::config::load()?;
    let cwd = std::env::current_dir()?;
    let conn = crate::db::open_or_create(&cwd.join(config.state.db_path))?;
    let query = format!("%{}%", args.query);
    let mut stmt = conn.prepare(
        "SELECT path, content FROM files WHERE content LIKE ?1 OR path LIKE ?1 ORDER BY path LIMIT ?2",
    )?;
    let rows = stmt.query_map(
        rusqlite::params![query, args.limit.unwrap_or(config.search.limit) as i64],
        |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
    )?;
    let mut results = Vec::new();
    for row in rows {
        let (path, content) = row?;
        results.push(serde_json::json!({
            "path": path,
            "snippet": content.unwrap_or_default().chars().take(config.search.snippet_chars).collect::<String>(),
            "score": 1.0
        }));
    }
    if args.json || retrieve {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "query": args.query,
                "results": results,
            }))?
        );
    } else {
        for result in results {
            println!("{}", result["path"].as_str().unwrap_or_default());
        }
    }
    Ok(())
}
