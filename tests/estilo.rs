//! Comprueba las reglas de redacción de los comentarios, de los mensajes y de
//! la documentación: sin rayas ni guiones largos (`—`, `–`) y sin comas antes o
//! después de la conjunción «y».
//!
//! Se revisa el texto de los comentarios (`//`, `///` y `//!`) uniendo las
//! líneas consecutivas de cada bloque, así como los literales de cadena. Los
//! fragmentos de código entre comillas invertidas y las tuplas como `(x, y)`
//! se ignoran.

use std::fs;
use std::path::{Path, PathBuf};

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// Sustituye los fragmentos entre comillas invertidas por un marcador.
fn strip_code(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut inside = false;
    for c in s.chars() {
        if c == '`' {
            inside = !inside;
            if inside {
                out.push_str("CODIGO");
            }
        } else if !inside {
            out.push(c);
        }
    }
    out
}

/// Elimina tuplas de identificadores como `(x, y)` o `(dx, dy, w)`.
fn strip_tuples(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('(') {
        out.push_str(&rest[..i]);
        let after = &rest[i + 1..];
        match after.find(')') {
            Some(j)
                if after[..j].split(',').all(|p| {
                    !p.trim().is_empty()
                        && p.trim()
                            .chars()
                            .all(|c| c.is_ascii_alphanumeric() || c == '_')
                }) && after[..j].contains(',') =>
            {
                rest = &after[j + 1..];
            }
            _ => {
                out.push('(');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Devuelve los problemas encontrados en un texto en prosa.
fn problems(text: &str) -> Vec<&'static str> {
    let t = strip_tuples(&strip_code(text));
    let words: Vec<&str> = t.split_whitespace().collect();
    let mut found = Vec::new();
    if t.contains('—') || t.contains('–') {
        found.push("raya o guion largo");
    }
    for (i, w) in words.iter().enumerate() {
        if *w == "y" && i > 0 && words[i - 1].ends_with(',') {
            found.push("coma antes de «y»");
        }
        if *w == concat!("y", ",") {
            found.push("coma después de «y»");
        }
    }
    found
}

/// Literales de cadena simples de una línea de código.
fn string_literals(code: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = code.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '"' {
            let mut s = String::new();
            while let Some(d) = chars.next() {
                match d {
                    '\\' => {
                        chars.next();
                    }
                    '"' => break,
                    _ => s.push(d),
                }
            }
            out.push(s);
        }
    }
    out
}

/// Revisa los comentarios `#` de `GDScript` (juego en Godot) y de los scripts de shell.
fn check_hash_comments(root: &Path, errors: &mut Vec<String>) {
    let mut hash_files: Vec<PathBuf> = ["godot", "scripts"]
        .iter()
        .flat_map(|d| fs::read_dir(root.join(d)).into_iter().flatten().flatten())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "gd" || x == "sh"))
        .collect();
    hash_files.sort();
    for f in &hash_files {
        let text = fs::read_to_string(f).expect("fuente legible");
        let mut block = String::new();
        let mut block_line = 0;
        for (n, line) in text.lines().enumerate() {
            let t = line.trim_start();
            if let Some(c) = t.strip_prefix('#').filter(|_| !t.starts_with("#!")) {
                if block.is_empty() {
                    block_line = n + 1;
                }
                block.push(' ');
                block.push_str(c);
                continue;
            }
            for p in problems(&block) {
                errors.push(format!("{}:{block_line}: {p}", f.display()));
            }
            block.clear();
            for lit in string_literals(line.split(" #").next().unwrap_or_default()) {
                for p in problems(&lit) {
                    errors.push(format!("{}:{}: {p} (cadena)", f.display(), n + 1));
                }
            }
        }
        for p in problems(&block) {
            errors.push(format!("{}:{block_line}: {p}", f.display()));
        }
    }
}

/// Revisa la documentación en Markdown de la raíz (fuera de los bloques de código).
fn check_markdown(root: &Path, errors: &mut Vec<String>) {
    let mut docs: Vec<PathBuf> = fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .collect();
    docs.sort();
    for f in &docs {
        let text = fs::read_to_string(f).expect("documento legible");
        let mut in_code = false;
        for (n, line) in text.lines().enumerate() {
            if line.trim_start().starts_with("```") {
                in_code = !in_code;
                continue;
            }
            if !in_code {
                for p in problems(line) {
                    errors.push(format!("{}:{}: {p}", f.display(), n + 1));
                }
            }
        }
    }
}

#[test]
fn comentarios_y_mensajes_siguen_las_reglas() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    for d in ["src", "tests", "examples", "benches"] {
        rust_files(&root.join(d), &mut files);
    }
    files.sort();
    let mut errors = Vec::new();
    for f in &files {
        let src = fs::read_to_string(f).expect("fuente legible");
        let mut block = String::new();
        let mut block_line = 0;
        for (n, line) in src.lines().enumerate() {
            let trimmed = line.trim_start();
            if let Some(c) = trimmed.strip_prefix("//") {
                if block.is_empty() {
                    block_line = n + 1;
                }
                block.push(' ');
                block.push_str(c.trim_start_matches(['/', '!']));
                continue;
            }
            for p in problems(&block) {
                errors.push(format!("{}:{block_line}: {p}", f.display()));
            }
            block.clear();
            let code = line.split("//").next().unwrap_or_default();
            for lit in string_literals(code) {
                for p in problems(&lit) {
                    errors.push(format!("{}:{}: {p} (cadena)", f.display(), n + 1));
                }
            }
            // Las líneas de cadenas de varias líneas (como la ayuda) también son prosa.
            if !trimmed.contains('"') && !trimmed.ends_with(';') && !trimmed.ends_with('{') {
                for p in problems(trimmed)
                    .into_iter()
                    .filter(|p| p.starts_with("raya"))
                {
                    errors.push(format!("{}:{}: {p}", f.display(), n + 1));
                }
            }
        }
        for p in problems(&block) {
            errors.push(format!("{}:{block_line}: {p}", f.display()));
        }
    }
    check_hash_comments(root, &mut errors);
    check_markdown(root, &mut errors);
    assert!(
        errors.is_empty(),
        "infracciones de estilo:\n{}",
        errors.join("\n")
    );
}

#[test]
fn el_detector_funciona() {
    assert_ne!(problems(concat!("uno,", " y dos")), [] as [&str; 0]);
    assert_ne!(problems(concat!("uno y", ", dos")), [] as [&str; 0]);
    assert_ne!(
        problems(&format!("rango 1{}4", '\u{2013}')),
        [] as [&str; 0]
    );
    assert_eq!(
        problems("la celda `(x, y)` y la tupla (x, y) están bien"),
        [] as [&str; 0]
    );
    assert_eq!(problems("rojo y verde"), [] as [&str; 0]);
}
