//! Lectura de niveles del *Video Game Level Corpus* (VGLC; Summerville et al.,
//! 2016) de Super Mario Bros y proyección a nuestro alfabeto de tiles.
//!
//! Leyenda de `smb.json`: `X` suelo, `S` ladrillo, `?`/`Q` bloque de pregunta,
//! `-` vacío, `E` enemigo, `<>[]` tubería, `o` moneda, `B`/`b` cañón.

use crate::error::{Error, Result};
use crate::level::{HEIGHT, Level, Style, Tile};
use std::path::Path;

/// Traduce un carácter de VGLC a nuestro tile lógico.
///
/// Las tuberías son columnas sólidas apoyadas en el suelo, así que se tratan
/// como suelo; los cañones son obstáculos letales aproximados como peligro.
pub fn map_char(c: char) -> Option<Tile> {
    Some(match c {
        '-' => Tile::Empty,
        'X' | '<' | '>' | '[' | ']' => Tile::Ground,
        'S' | '?' | 'Q' => Tile::Platform,
        'E' => Tile::Enemy,
        'o' => Tile::Coin,
        'B' | 'b' => Tile::Hazard,
        _ => return None,
    })
}

/// Convierte el texto de un nivel VGLC en un [`Level`].
///
/// El suelo que no apoya en suelo (bloques "X" flotantes, escaleras de
/// bloques) se reclasifica como plataforma, con la misma regla que usa la
/// reparación del GA, para que ambos corpus compartan semántica.
///
/// # Errors
///
/// Si el nivel no tiene [`HEIGHT`] filas iguales o contiene caracteres desconocidos.
pub fn parse(text: &str) -> Result<Level> {
    let rows: Vec<Vec<char>> = text
        .lines()
        .map(|l| l.trim_end_matches('\r').chars().collect())
        .filter(|r: &Vec<char>| !r.is_empty())
        .collect();
    if rows.len() != HEIGHT {
        return Err(Error::format(
            "nivel VGLC",
            format!("{} filas en lugar de {HEIGHT}", rows.len()),
        ));
    }
    let w = rows[0].len();
    if rows.iter().any(|r| r.len() != w) {
        return Err(Error::format("nivel VGLC", "filas de distinta longitud"));
    }
    let mut tiles = vec![Tile::Empty; w * HEIGHT];
    for (y, row) in rows.iter().enumerate() {
        for (x, &c) in row.iter().enumerate() {
            tiles[x * HEIGHT + y] = map_char(c).ok_or_else(|| {
                Error::format("nivel VGLC", format!("carácter «{c}» desconocido"))
            })?;
        }
    }
    for col in tiles.as_chunks_mut::<HEIGHT>().0 {
        for y in (0..HEIGHT - 1).rev() {
            if col[y] == Tile::Ground && col[y + 1] != Tile::Ground {
                col[y] = Tile::Platform;
            }
        }
    }
    Level::from_tiles(tiles, Style::Base)
}

/// Carga todos los `*.txt` de un directorio (ordenados por nombre).
///
/// # Errors
///
/// Si el directorio no se puede leer o algún nivel es inválido.
pub fn load_dir(dir: impl AsRef<Path>) -> Result<Vec<(String, Level)>> {
    let mut paths: Vec<_> = std::fs::read_dir(dir.as_ref())?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt"))
        .collect();
    paths.sort();
    paths
        .into_iter()
        .map(|p| {
            let name = p
                .file_stem()
                .map_or_else(String::new, |s| s.to_string_lossy().into_owned());
            Ok((name, parse(&std::fs::read_to_string(&p)?)?))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traduce_y_reclasifica_suelo_flotante() {
        let mut text = String::new();
        for y in 0..HEIGHT {
            text.push_str(match y {
                9 => "--XX----\n",
                12 => "-E-o?---\n",
                13 => "XX[]XXXX\n",
                _ => "--------\n",
            });
        }
        let l = parse(&text).expect("válido");
        assert_eq!(l.get(2, 9), Tile::Platform);
        assert_eq!(l.get(2, 13), Tile::Ground);
        assert_eq!(l.get(1, 12), Tile::Enemy);
        assert_eq!(l.get(4, 12), Tile::Platform);
        assert!(parse("X\nX\n").is_err());
    }
}
