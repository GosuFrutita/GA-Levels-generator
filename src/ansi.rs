//! Visor de niveles en la terminal con códigos ANSI *truecolor*.
//!
//! Cada celda se dibuja como dos espacios con color de fondo (`ESC[48;2;r;g;bm`)
//! y un carácter indicativo; la animación usa la pantalla alternativa
//! (`ESC[?1049h`) y oculta el cursor (`ESC[?25l`).

use crate::image::Rgba;
use crate::level::{HEIGHT, Level, Tile};
use crate::render::Tilesets;
use std::fmt::Write as _;

fn glyph(t: Tile) -> &'static str {
    match t {
        Tile::Empty | Tile::Ground | Tile::Platform => "  ",
        Tile::Ladder => "╫╫",
        Tile::Hazard => "▲▲",
        Tile::Enemy => "ÒÓ",
        Tile::Coin => "()",
        Tile::Decor => "♣ ",
        Tile::Start => "▶ ",
        Tile::Goal => "⚑ ",
    }
}

/// Dibuja un fotograma: columnas `[x0, x0 + cols)`, con el jugador en `player`.
pub fn frame(
    level: &Level,
    sets: &Tilesets,
    x0: usize,
    cols: usize,
    player: Option<(usize, usize)>,
    trail: &[(usize, usize)],
) -> String {
    let mut s = String::with_capacity(cols * HEIGHT * 40);
    let sky: Rgba = sets.tile_color(level.style, Tile::Empty);
    for y in 0..HEIGHT {
        for x in x0..(x0 + cols).min(level.width()) {
            let t = level.get(x, y);
            let bg = if t.is_solid() {
                sets.tile_color(level.style, t)
            } else {
                sky
            };
            let _ = write!(s, "\x1b[48;2;{};{};{}m", bg[0], bg[1], bg[2]);
            if player == Some((x, y)) {
                s.push_str("\x1b[38;2;20;160;60m☻ ");
            } else if trail.contains(&(x, y)) && t == Tile::Empty {
                s.push_str("\x1b[38;2;230;180;20m··");
            } else {
                let fg = sets.tile_color(level.style, t);
                let _ = write!(
                    s,
                    "\x1b[38;2;{};{};{}m{}",
                    fg[0] / 2,
                    fg[1] / 2,
                    fg[2] / 2,
                    glyph(t)
                );
            }
        }
        s.push_str("\x1b[0m\n");
    }
    s
}

/// Anima la trayectoria en la terminal (`delay_ms` por paso).
///
/// # Errors
///
/// Si falla la escritura en la salida estándar.
pub fn animate(
    level: &Level,
    sets: &Tilesets,
    traj: &[(usize, usize)],
    cols: usize,
    delay_ms: u64,
) -> std::io::Result<()> {
    use std::io::Write;
    let mut out = std::io::stdout().lock();
    write!(out, "\x1b[?1049h\x1b[?25l")?;
    for (i, &p) in traj.iter().enumerate() {
        let x0 =
            p.0.saturating_sub(cols / 3)
                .min(level.width().saturating_sub(cols));
        let f = frame(level, sets, x0, cols, Some(p), &traj[..i]);
        writeln!(
            out,
            "\x1b[H{f}paso {}/{}  estilo {}",
            i + 1,
            traj.len(),
            level.style.name()
        )?;
        out.flush()?;
        std::thread::sleep(std::time::Duration::from_millis(delay_ms));
    }
    std::thread::sleep(std::time::Duration::from_millis(600));
    write!(out, "\x1b[?25h\x1b[?1049l")?;
    out.flush()
}
