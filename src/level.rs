//! Representación de un nivel: matriz de tipos lógicos de tile + estilo visual.
//!
//! La matriz se guarda **por columnas** (`x * alto + y`, con `y = 0` arriba):
//! una columna es un slice contiguo, así el cruce por columnas es un `memcpy`
//! y el verificador de jugabilidad recorre memoria contigua.

use crate::error::{Error, Result};
use std::fmt;

/// Alto fijo de los niveles, en tiles.
pub const HEIGHT: usize = 14;
/// Columnas fijas al inicio (zona de salida) y al final (zona de meta).
pub const SAFE_COLS: usize = 3;

/// Tipo lógico de un tile.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub enum Tile {
    /// Aire: atravesable, sin apoyo.
    #[default]
    Empty = 0,
    /// Terreno sólido (pasto/tierra según estilo).
    Ground = 1,
    /// Plataforma sólida flotante (puente, bloque).
    Platform = 2,
    /// Escalera: atravesable, se puede trepar y sirve de apoyo.
    Ladder = 3,
    /// Pinchos/ácido: letal al tocarlo, no sirve de apoyo.
    Hazard = 4,
    /// Enemigo caminante: letal, hay que saltarlo.
    Enemy = 5,
    /// Moneda u objeto coleccionable: atravesable.
    Coin = 6,
    /// Decoración (planta, señal...): atravesable, puramente visual.
    Decor = 7,
    /// Punto de salida (fijo, no evoluciona).
    Start = 8,
    /// Meta (fija, no evoluciona).
    Goal = 9,
}

impl Tile {
    /// Todos los tiles, en orden de discriminante.
    pub const ALL: [Self; 10] = [
        Self::Empty,
        Self::Ground,
        Self::Platform,
        Self::Ladder,
        Self::Hazard,
        Self::Enemy,
        Self::Coin,
        Self::Decor,
        Self::Start,
        Self::Goal,
    ];
    /// Tiles que el algoritmo genético puede colocar libremente.
    pub const EVOLVABLE: [Self; 8] = [
        Self::Empty,
        Self::Ground,
        Self::Platform,
        Self::Ladder,
        Self::Hazard,
        Self::Enemy,
        Self::Coin,
        Self::Decor,
    ];
    /// Número de tipos.
    pub const COUNT: usize = 10;

    /// ¿Es un bloque estructural (suelo o plataforma)?
    pub const fn is_solid(self) -> bool {
        matches!(self, Self::Ground | Self::Platform)
    }

    /// ¿Bloquea el movimiento en todas direcciones? Solo el suelo: las
    /// plataformas son *de un solo sentido* (se atraviesan al subir o al
    /// caminar y sostienen al caer encima), como en el género y en el nivel de
    /// ejemplo `platformer-starter` (colisión `one_way` en Godot).
    pub const fn blocks(self) -> bool {
        matches!(self, Self::Ground)
    }

    /// ¿Se puede estar de pie encima?
    pub const fn is_support(self) -> bool {
        matches!(self, Self::Ground | Self::Platform | Self::Ladder)
    }

    /// ¿Mata al tocarlo?
    pub const fn is_lethal(self) -> bool {
        matches!(self, Self::Hazard | Self::Enemy)
    }

    /// ¿El jugador puede ocupar la celda sin morir?
    pub const fn is_passable(self) -> bool {
        !self.blocks() && !self.is_lethal()
    }

    /// Carácter en el formato de texto.
    pub const fn to_char(self) -> char {
        match self {
            Self::Empty => '-',
            Self::Ground => 'X',
            Self::Platform => 'S',
            Self::Ladder => 'L',
            Self::Hazard => '^',
            Self::Enemy => 'E',
            Self::Coin => 'o',
            Self::Decor => 'd',
            Self::Start => '@',
            Self::Goal => '!',
        }
    }

    /// Tile a partir de su carácter.
    pub const fn from_char(c: char) -> Option<Self> {
        Some(match c {
            '-' => Self::Empty,
            'X' => Self::Ground,
            'S' => Self::Platform,
            'L' => Self::Ladder,
            '^' => Self::Hazard,
            'E' => Self::Enemy,
            'o' => Self::Coin,
            'd' => Self::Decor,
            '@' => Self::Start,
            '!' => Self::Goal,
            _ => return None,
        })
    }
}

/// Estilo visual: uno de los cuatro packs *Pixel Platformer* de Kenney (CC0).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub enum Style {
    /// Pack base: pradera de pasto y tierra.
    #[default]
    Base = 0,
    /// Expansión granja.
    Farm = 1,
    /// Expansión comida.
    Food = 2,
    /// Expansión industrial.
    Industrial = 3,
}

impl Style {
    /// Todos los estilos.
    pub const ALL: [Self; 4] = [Self::Base, Self::Farm, Self::Food, Self::Industrial];
    /// Número de estilos.
    pub const COUNT: usize = 4;

    /// Índice del estilo.
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Nombre corto en español.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Base => "pradera",
            Self::Farm => "granja",
            Self::Food => "dulce",
            Self::Industrial => "industrial",
        }
    }

    /// Estilo a partir de su nombre.
    pub fn from_name(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|st| st.name() == s)
    }
}

/// Nivel: matriz de tiles por columnas más el gen de estilo.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Level {
    width: usize,
    tiles: Vec<Tile>,
    /// Estilo visual elegido.
    pub style: Style,
}

impl Level {
    /// Nivel vacío de `width` columnas.
    ///
    /// # Panics
    ///
    /// Si `width < 2 * SAFE_COLS + 1`.
    pub fn empty(width: usize, style: Style) -> Self {
        assert!(width > 2 * SAFE_COLS, "nivel demasiado estrecho");
        Self {
            width,
            tiles: vec![Tile::Empty; width * HEIGHT],
            style,
        }
    }

    /// Construye un nivel a partir de su vector de tiles por columnas.
    ///
    /// # Errors
    ///
    /// Si la longitud no es múltiplo de [`HEIGHT`] o el nivel es demasiado estrecho.
    pub fn from_tiles(tiles: Vec<Tile>, style: Style) -> Result<Self> {
        if !tiles.len().is_multiple_of(HEIGHT) || tiles.len() / HEIGHT <= 2 * SAFE_COLS {
            return Err(Error::format("nivel", "dimensiones inválidas"));
        }
        Ok(Self {
            width: tiles.len() / HEIGHT,
            tiles,
            style,
        })
    }

    /// Ancho en columnas.
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Alto en filas.
    pub const fn height(&self) -> usize {
        HEIGHT
    }

    /// Tiles por columnas.
    pub fn tiles(&self) -> &[Tile] {
        &self.tiles
    }

    /// Tiles por columnas, mutables.
    pub fn tiles_mut(&mut self) -> &mut [Tile] {
        &mut self.tiles
    }

    /// Tile en `(x, y)`.
    ///
    /// # Panics
    ///
    /// Si la coordenada está fuera del nivel.
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> Tile {
        debug_assert!(y < HEIGHT);
        self.tiles[x * HEIGHT + y]
    }

    /// Tile en coordenadas con signo: fuera de los lados o por arriba es
    /// `Empty`; por debajo del nivel también es `Empty` (se cae al vacío).
    #[inline]
    pub fn get_or_empty(&self, x: i32, y: i32) -> Tile {
        if x < 0 || y < 0 || x as usize >= self.width || y as usize >= HEIGHT {
            Tile::Empty
        } else {
            self.get(x as usize, y as usize)
        }
    }

    /// Copia del nivel con la zona de patrulla de cada enemigo marcada como
    /// enemigo: en el juego cada uno se mueve una casilla a cada lado si está
    /// libre y tiene apoyo, así que esas celdas también matan. La física del
    /// verificador y del agente trabaja sobre esta copia.
    #[must_use]
    pub fn with_patrols(&self) -> Self {
        let mut out = self.clone();
        for x in 0..self.width {
            for y in 0..HEIGHT {
                if self.get(x, y) != Tile::Enemy {
                    continue;
                }
                for nx in [x.wrapping_sub(1), x + 1] {
                    let free = nx < self.width && self.get(nx, y).is_passable();
                    if free && self.get_or_empty(nx as i32, y as i32 + 1).is_support() {
                        out.set(nx, y, Tile::Enemy);
                    }
                }
            }
        }
        out
    }

    /// Escribe el tile en `(x, y)`.
    ///
    /// # Panics
    ///
    /// Si la coordenada está fuera del nivel.
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, t: Tile) {
        debug_assert!(y < HEIGHT);
        self.tiles[x * HEIGHT + y] = t;
    }

    /// Columna `x` (de arriba abajo).
    ///
    /// # Panics
    ///
    /// Si `x` está fuera del nivel.
    pub fn column(&self, x: usize) -> &[Tile] {
        &self.tiles[x * HEIGHT..(x + 1) * HEIGHT]
    }

    /// Columna `x`, mutable.
    ///
    /// # Panics
    ///
    /// Si `x` está fuera del nivel.
    pub fn column_mut(&mut self, x: usize) -> &mut [Tile] {
        &mut self.tiles[x * HEIGHT..(x + 1) * HEIGHT]
    }

    /// Posición de la salida `(x, y)`, si existe.
    pub fn start(&self) -> Option<(usize, usize)> {
        self.find(Tile::Start)
    }

    /// Posición de la meta `(x, y)`, si existe.
    pub fn goal(&self) -> Option<(usize, usize)> {
        self.find(Tile::Goal)
    }

    fn find(&self, t: Tile) -> Option<(usize, usize)> {
        self.tiles
            .iter()
            .position(|&v| v == t)
            .map(|i| (i / HEIGHT, i % HEIGHT))
    }

    /// Altura de la superficie del suelo en la columna `x`: fila del primer
    /// `Ground` empezando desde abajo de un bloque contiguo que llega al fondo,
    /// o `HEIGHT` si la columna no tiene suelo (hueco).
    pub fn ground_top(&self, x: usize) -> usize {
        let col = self.column(x);
        let mut y = HEIGHT;
        while y > 0 && col[y - 1] == Tile::Ground {
            y -= 1;
        }
        y
    }

    /// Escribe una columna de suelo con la superficie en la fila `g` (la columna
    /// queda vacía por encima); con `g == HEIGHT` la columna es un hueco.
    ///
    /// # Panics
    ///
    /// Si `x` está fuera del nivel.
    pub fn set_ground_column(&mut self, x: usize, g: usize) {
        let col = self.column_mut(x);
        col.fill(Tile::Empty);
        if g < HEIGHT {
            col[g..].fill(Tile::Ground);
        }
    }

    /// ¿Se puede estar de pie en `(x, y)`? (celda atravesable con apoyo debajo).
    pub fn can_stand(&self, x: i32, y: i32) -> bool {
        y >= 0 && self.get_or_empty(x, y).is_passable() && self.get_or_empty(x, y + 1).is_support()
    }

    /// ¿Tiene salida una escalera cuya celda superior es `(x, y)`? Desde lo alto
    /// se sale a un lado o, si encima hay hueco, hacia arriba en diagonal.
    pub fn ladder_exit(&self, x: i32, y: i32) -> bool {
        self.can_stand(x - 1, y)
            || self.can_stand(x + 1, y)
            || self.can_stand(x - 1, y - 1)
            || self.can_stand(x + 1, y - 1)
    }

    /// Lee un nivel en formato texto: primera línea `estilo=<nombre>` opcional,
    /// después `HEIGHT` líneas de igual longitud con un carácter por tile.
    ///
    /// # Errors
    ///
    /// Si el texto no tiene el formato esperado.
    pub fn parse(text: &str) -> Result<Self> {
        let bad = |d: String| Error::format("nivel en texto", d);
        let mut lines = text
            .lines()
            .map(str::trim_end)
            .filter(|l| !l.is_empty())
            .peekable();
        let mut style = Style::Base;
        if let Some(first) = lines.peek()
            && let Some(name) = first.strip_prefix("estilo=")
        {
            style = Style::from_name(name.trim())
                .ok_or_else(|| bad(format!("estilo «{name}» desconocido")))?;
            lines.next();
        }
        let rows: Vec<Vec<char>> = lines.map(|l| l.chars().collect()).collect();
        if rows.len() != HEIGHT {
            return Err(bad(format!(
                "se esperaban {HEIGHT} filas, hay {}",
                rows.len()
            )));
        }
        let width = rows[0].len();
        if rows.iter().any(|r| r.len() != width) {
            return Err(bad("filas de distinta longitud".into()));
        }
        let mut tiles = vec![Tile::Empty; width * HEIGHT];
        for (y, row) in rows.iter().enumerate() {
            for (x, &c) in row.iter().enumerate() {
                tiles[x * HEIGHT + y] =
                    Tile::from_char(c).ok_or_else(|| bad(format!("carácter «{c}» desconocido")))?;
            }
        }
        Self::from_tiles(tiles, style)
    }
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "estilo={}", self.style.name())?;
        for y in 0..HEIGHT {
            let row: String = (0..self.width).map(|x| self.get(x, y).to_char()).collect();
            writeln!(f, "{row}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texto_ida_y_vuelta() {
        let mut lvl = Level::empty(10, Style::Food);
        for x in 0..10 {
            lvl.set(x, HEIGHT - 1, Tile::Ground);
        }
        lvl.set(1, HEIGHT - 2, Tile::Start);
        lvl.set(8, HEIGHT - 2, Tile::Goal);
        lvl.set(5, HEIGHT - 2, Tile::Enemy);
        let back = Level::parse(&lvl.to_string()).expect("válido");
        assert_eq!(back, lvl);
        assert_eq!(back.start(), Some((1, HEIGHT - 2)));
        assert_eq!(back.ground_top(3), HEIGHT - 1);
    }

    #[test]
    fn caracteres_unicos() {
        for a in Tile::ALL {
            assert_eq!(Tile::from_char(a.to_char()), Some(a));
        }
    }

    #[test]
    fn semantica() {
        assert!(Tile::Ladder.is_support() && Tile::Ladder.is_passable());
        assert!(!Tile::Enemy.is_passable() && !Tile::Enemy.is_support());
        assert!(Tile::Ground.is_solid() && !Tile::Coin.is_solid());
        // Plataformas de un solo sentido: apoyo, pero se pueden atravesar.
        assert!(
            Tile::Platform.is_support() && Tile::Platform.is_passable() && !Tile::Platform.blocks()
        );
    }
}
