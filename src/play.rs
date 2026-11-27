//! Verificador de jugabilidad sin motor físico.
//!
//! El jugador ocupa una celda. Un **estado de reposo** es una celda atravesable
//! en la que se puede estar de pie (apoyo debajo) o colgado de una escalera.
//! Desde cada estado hay cuatro tipos de movimiento:
//!
//! 1. **Caminar** una columna; si no hay apoyo, se cae en vertical.
//! 2. **Trepar** por escaleras (arriba/abajo).
//! 3. **Saltar** siguiendo arcos parabólicos precalculados una sola vez.
//! 4. **Soltarse** de una escalera (caída).
//!
//! Se busca con Dijkstra usando una cola de cubetas (algoritmo de Dial), con
//! coste 1 por paso y `columnas + 1` por salto, de modo que el camino solo salta cuando hace
//! falta (Summerville et al., 2017). Una sola pasada da: si la meta es
//! alcanzable, el camino óptimo, los saltos necesarios, el progreso máximo y el
//! conjunto de celdas alcanzables.

use crate::level::{HEIGHT, Level, Tile};

/// Parámetros de la física discreta del jugador.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Physics {
    /// Altura máxima de salto, en celdas.
    pub jump_height: u8,
    /// Distancia horizontal máxima de un salto que aterriza a la misma altura.
    pub jump_length: u8,
}

impl Default for Physics {
    fn default() -> Self {
        Self {
            jump_height: 3,
            jump_length: 4,
        }
    }
}

/// Un paso de un arco de salto: desplazamiento relativo al despegue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ArcStep {
    dx: i8,
    dy: i8,
    /// `true` si en este paso el jugador ya está bajando (puede aterrizar).
    falling: bool,
}

/// Arco de salto precalculado.
#[derive(Debug, Clone)]
pub struct JumpArc {
    /// Dirección horizontal (−1, 0, +1).
    pub dir: i8,
    /// Altura del vértice, en celdas.
    pub apex: u8,
    /// Distancia horizontal al volver a la altura de despegue.
    pub run: u8,
    steps: Vec<ArcStep>,
}

/// Genera los arcos de salto discretizando parábolas.
///
/// Para cada dirección, altura `h ∈ [1, jump_height]` y recorrido
/// `r ∈ [0, jump_length]` se integra `y(t) = −v₀t + g t²/2`, `x(t) = v_x t`,
/// con `v₀ = √(2gh)` y `v_x = r / T`, `T = 2v₀/g` (tiempo de vuelo hasta la
/// altura de despegue). Se registran las celdas visitadas, insertando la
/// celda intermedia en los pasos diagonales para no "cortar esquinas". La
/// caída continúa por debajo del despegue hasta `HEIGHT` filas.
pub fn build_arcs(p: Physics) -> Vec<JumpArc> {
    const G: f64 = 1.0;
    let mut arcs = Vec::new();
    for dir in [-1_i8, 0, 1] {
        for apex in 1..=p.jump_height {
            let runs: Vec<u8> = if dir == 0 {
                vec![0]
            } else {
                (1..=p.jump_length).collect()
            };
            for run in runs {
                let v0 = (2.0 * G * f64::from(apex)).sqrt();
                let flight = 2.0 * v0 / G;
                let vx = f64::from(run) / flight;
                let mut steps: Vec<ArcStep> = Vec::new();
                let (mut cx, mut cy) = (0_i32, 0_i32);
                let dt = 0.01;
                let mut t = 0.0;
                while cy < HEIGHT as i32 + 1 {
                    t += dt;
                    let x = vx * t + 0.5; // centro de la celda
                    let y = -v0 * t + 0.5 * G * t * t;
                    let nx = x.floor() as i32;
                    let ny = (y + 0.5).floor() as i32;
                    if (nx, ny) == (cx, cy) {
                        continue;
                    }
                    let falling = -v0 + G * t > 0.0;
                    if nx != cx && ny != cy {
                        // Celda intermedia: primero el eje vertical al subir y el
                        // horizontal al bajar (el más conservador).
                        let (ix, iy) = if falling { (nx, cy) } else { (cx, ny) };
                        steps.push(ArcStep {
                            dx: (ix * i32::from(dir)) as i8,
                            dy: iy as i8,
                            falling,
                        });
                    }
                    cx = nx;
                    cy = ny;
                    steps.push(ArcStep {
                        dx: (cx * i32::from(dir)) as i8,
                        dy: cy as i8,
                        falling,
                    });
                }
                if dir == 0 {
                    for s in &mut steps {
                        s.dx = 0;
                    }
                }
                arcs.push(JumpArc {
                    dir,
                    apex,
                    run,
                    steps,
                });
            }
        }
    }
    arcs
}

/// Tipo de movimiento de una arista del camino.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    /// Caminar (y caer si hace falta).
    Walk,
    /// Trepar o bajar por una escalera.
    Climb,
    /// Saltar con el arco indicado.
    Jump(u16),
}

/// Resultado del análisis de jugabilidad.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlayResult {
    /// ¿Se puede llegar a la meta?
    pub solvable: bool,
    /// Columna máxima alcanzada / columna de la meta, en `[0, 1]`.
    pub progress: f64,
    /// Celdas del camino óptimo, de la salida a la meta (vacío si no hay).
    pub path: Vec<(usize, usize)>,
    /// Movimientos del camino óptimo.
    pub moves: Vec<Move>,
    /// Saltos necesarios en el camino óptimo.
    pub jumps: u32,
    /// Saltos "ajustados" (cerca del límite de altura o longitud).
    pub tight_jumps: u32,
    /// Monedas alcanzables.
    pub reachable_coins: u32,
}

/// Memoria de trabajo reutilizable (una por hilo): evita asignar y limpiar
/// `ancho × alto` celdas por llamada gracias a sellos de época.
#[derive(Debug, Clone, Default)]
pub struct PlayScratch {
    epoch: u32,
    seen: Vec<u32>,
    touched: Vec<u32>,
    dist: Vec<u32>,
    prev: Vec<u32>,
    prev_move: Vec<Move>,
    buckets: Vec<Vec<u32>>,
}

impl PlayScratch {
    /// ¿Atravesó el jugador la celda `(x, y)` en el último análisis?
    pub fn touched(&self, x: i32, y: i32) -> bool {
        if x < 0 || y < 0 || y as usize >= HEIGHT {
            return false;
        }
        let i = x as usize * HEIGHT + y as usize;
        i < self.touched.len() && self.touched[i] == self.epoch
    }

    fn reset(&mut self, cells: usize, n_buckets: usize) {
        if self.buckets.len() != n_buckets {
            self.buckets = vec![Vec::new(); n_buckets];
        }
        if self.seen.len() < cells {
            self.seen.resize(cells, 0);
            self.touched.resize(cells, 0);
            self.dist.resize(cells, u32::MAX);
            self.prev.resize(cells, u32::MAX);
            self.prev_move.resize(cells, Move::Walk);
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.seen.fill(0);
            self.touched.fill(0);
            self.epoch = 1;
        }
        for b in &mut self.buckets {
            b.clear();
        }
    }
}

/// Verificador con arcos precalculados.
#[derive(Debug, Clone)]
pub struct Checker {
    physics: Physics,
    arcs: Vec<JumpArc>,
    /// Coste máximo de una arista + 1 (número de cubetas de Dial).
    n_buckets: usize,
}

/// Coste de caminar o trepar una celda.
pub const STEP_COST: u32 = 1;
/// Recargo de un salto sobre su desplazamiento horizontal: así un salto nunca
/// es más barato que caminar la misma distancia y el camino óptimo solo salta
/// cuando hace falta (idea de Summerville et al., 2017: coste 2 por salto).
pub const JUMP_PENALTY: u32 = 1;

/// Coste de un salto que aterriza `dx` columnas más allá.
#[inline]
fn jump_cost(dx: i32) -> u32 {
    dx.unsigned_abs().max(1) + JUMP_PENALTY
}

impl Default for Checker {
    fn default() -> Self {
        Self::new(Physics::default())
    }
}

#[inline]
fn idx(x: i32, y: i32) -> u32 {
    (x as u32) * HEIGHT as u32 + y as u32
}

impl Checker {
    /// Crea el verificador y precalcula los arcos.
    pub fn new(physics: Physics) -> Self {
        let arcs = build_arcs(physics);
        let max_dx = arcs
            .iter()
            .flat_map(|a| a.steps.iter().map(|s| i32::from(s.dx)))
            .map(i32::unsigned_abs)
            .max()
            .unwrap_or(1);
        Self {
            physics,
            arcs,
            n_buckets: (jump_cost(max_dx.cast_signed()) + 1) as usize,
        }
    }

    /// Física usada.
    pub fn physics(&self) -> Physics {
        self.physics
    }

    /// Arcos precalculados.
    pub fn arcs(&self) -> &[JumpArc] {
        &self.arcs
    }

    /// ¿Hay apoyo bajo `(x, y)`? El fondo del mapa no es apoyo (es un foso).
    #[inline]
    fn supported(level: &Level, x: i32, y: i32) -> bool {
        level.get_or_empty(x, y + 1).is_support()
    }

    /// ¿Es `(x, y)` un estado de reposo?
    #[inline]
    pub fn is_rest(level: &Level, x: i32, y: i32) -> bool {
        let t = level.get_or_empty(x, y);
        let inside = x >= 0 && y >= 0 && (x as usize) < level.width() && (y as usize) < HEIGHT;
        inside && t.is_passable() && (t == Tile::Ladder || Self::supported(level, x, y))
    }

    /// Cae en vertical desde `(x, y)` (celda ya atravesable) hasta reposar.
    /// Devuelve la celda de reposo o `None` si muere (foso o peligro).
    pub(crate) fn fall(
        level: &Level,
        x: i32,
        mut y: i32,
        mut visit: impl FnMut(i32, i32),
    ) -> Option<(i32, i32)> {
        loop {
            visit(x, y);
            if Self::is_rest(level, x, y) {
                return Some((x, y));
            }
            y += 1;
            if y >= HEIGHT as i32 {
                return None;
            }
            if !level.get_or_empty(x, y).is_passable() {
                return None; // cae sobre peligro/enemigo (los sólidos ya darían apoyo)
            }
        }
    }

    /// Recorre un arco desde `(x, y)`; devuelve dónde aterriza.
    pub(crate) fn follow_arc(
        level: &Level,
        arc: &JumpArc,
        x: i32,
        y: i32,
        mut visit: impl FnMut(i32, i32),
    ) -> Option<(i32, i32)> {
        // Hace falta espacio para despegar: la celda de encima debe ser atravesable.
        let mut last = (x, y);
        for s in &arc.steps {
            let (cx, cy) = (x + i32::from(s.dx), y + i32::from(s.dy));
            if cx < 0 || cx as usize >= level.width() || cy >= HEIGHT as i32 {
                return None;
            }
            let t = level.get_or_empty(cx, cy); // por encima del mapa: aire
            if t.is_lethal() {
                return None;
            }
            if t.blocks() {
                // Choque: si bajaba y la celda previa tiene apoyo, aterriza ahí;
                // si subía (cabezazo) o choca lateralmente, cae desde la previa.
                return if s.falling && Self::is_rest(level, last.0, last.1) {
                    Some(last)
                } else if last.1 >= 0 {
                    Self::fall(level, last.0, last.1, visit)
                } else {
                    None
                };
            }
            visit(cx, cy);
            // Al bajar se aterriza sobre cualquier apoyo (las plataformas solo
            // sostienen desde arriba; al subir se atraviesan).
            if s.falling && cy >= 0 && Self::is_rest(level, cx, cy) && t != Tile::Ladder {
                return Some((cx, cy));
            }
            if s.falling && t == Tile::Ladder {
                return Some((cx, cy)); // se agarra a la escalera
            }
            last = (cx, cy);
        }
        None
    }

    /// Reconstruye la trayectoria celda a celda del camino óptimo de `res`
    /// (incluye las celdas de los arcos de salto y de las caídas).
    pub fn trajectory(&self, level: &Level, res: &PlayResult) -> Vec<(usize, usize)> {
        let mut out = Vec::with_capacity(res.path.len() * 4);
        let Some(&first) = res.path.first() else {
            return out;
        };
        out.push(first);
        for (w, mv) in res.path.windows(2).zip(&res.moves) {
            let (x, y) = (w[0].0 as i32, w[0].1 as i32);
            let mut push = |cx: i32, cy: i32| {
                if cy >= 0 && out.last() != Some(&(cx as usize, cy as usize)) {
                    out.push((cx as usize, cy as usize));
                }
            };
            match *mv {
                Move::Walk | Move::Climb => {
                    let (tx, ty) = (w[1].0 as i32, w[1].1 as i32);
                    // Paso horizontal o vertical seguido de la caída cuando haga falta.
                    if tx == x {
                        for yy in if ty > y { y + 1..=ty } else { ty..=y - 1 } {
                            push(x, yy);
                        }
                    } else {
                        Self::fall(level, tx, y, &mut push);
                    }
                }
                Move::Jump(a) => {
                    Self::follow_arc(level, &self.arcs[usize::from(a)], x, y, &mut push);
                }
            }
            if out.last() != Some(&w[1]) {
                out.push(w[1]);
            }
        }
        out
    }

    /// Analiza la jugabilidad del nivel.
    pub fn check(&self, level: &Level, scratch: &mut PlayScratch) -> PlayResult {
        let (Some(start), Some(goal)) = (level.start(), level.goal()) else {
            return PlayResult::default();
        };
        scratch.reset(level.width() * HEIGHT, self.n_buckets);
        // La salida debe tener apoyo; si no, se deja caer.
        let Some(s0) = Self::fall(level, start.0 as i32, start.1 as i32, |_, _| {}) else {
            return PlayResult::default();
        };
        let mut search = Search {
            checker: self,
            level,
            sc: scratch,
            goal: (goal.0 as i32, goal.1 as i32),
            current: 0,
            pending: 0,
            max_x: s0.0,
            goal_state: None,
            coins: 0,
        };
        search.relax(s0, 0, u32::MAX, Move::Walk);
        search.run();
        search.finish(s0, goal)
    }
}

/// Estado de una búsqueda en curso (separa el bucle de Dijkstra de la
/// generación de vecinos y de la reconstrucción del resultado).
struct Search<'a> {
    checker: &'a Checker,
    level: &'a Level,
    sc: &'a mut PlayScratch,
    goal: (i32, i32),
    current: u32,
    pending: usize,
    max_x: i32,
    goal_state: Option<u32>,
    coins: u32,
}

impl Search<'_> {
    /// Marca una celda atravesada (alcanzabilidad, monedas y meta al vuelo).
    fn touch(&mut self, x: i32, y: i32) {
        if y < 0 {
            return;
        }
        let i = idx(x, y) as usize;
        if self.sc.touched[i] != self.sc.epoch {
            self.sc.touched[i] = self.sc.epoch;
            if self.level.get(x as usize, y as usize) == Tile::Coin {
                self.coins += 1;
            }
        }
    }

    /// Relaja la arista hacia `to` con coste `cost` desde el estado `from`.
    /// En caso de empate gana el primero: los arcos están ordenados de menor a
    /// mayor, así el camino usa el salto más pequeño que basta.
    fn relax(&mut self, to: (i32, i32), cost: u32, from: u32, mv: Move) {
        let ti = idx(to.0, to.1) as usize;
        let nd = self.current + cost;
        let sc = &mut *self.sc;
        if sc.seen[ti] != sc.epoch || nd < sc.dist[ti] {
            sc.seen[ti] = sc.epoch;
            sc.dist[ti] = nd;
            sc.prev[ti] = from;
            sc.prev_move[ti] = mv;
            let nb = sc.buckets.len() as u32;
            sc.buckets[(nd % nb) as usize].push(ti as u32);
            self.pending += 1;
        }
    }

    /// Bucle de Dijkstra con cubetas (Dial).
    fn run(&mut self) {
        let nb = self.sc.buckets.len() as u32;
        while self.pending > 0 {
            let b = (self.current % nb) as usize;
            let Some(u) = self.sc.buckets[b].pop() else {
                self.current += 1;
                continue;
            };
            self.pending -= 1;
            if self.sc.dist[u as usize] != self.current {
                continue; // entrada obsoleta
            }
            self.expand(u);
        }
    }

    /// Genera los vecinos del estado `u`.
    fn expand(&mut self, u: u32) {
        let level = self.level;
        let (ux, uy) = ((u as usize / HEIGHT) as i32, (u as usize % HEIGHT) as i32);
        self.max_x = self.max_x.max(ux);
        if (ux, uy) == self.goal && self.goal_state.is_none() {
            self.goal_state = Some(u);
        }
        self.touch(ux, uy);

        // 1. Caminar (y caer).
        for dx in [-1, 1] {
            let nx = ux + dx;
            if nx < 0
                || nx as usize >= level.width()
                || !level.get(nx as usize, uy as usize).is_passable()
            {
                continue;
            }
            let mut visited = [(0, 0); HEIGHT];
            let mut nv = 0;
            if let Some(to) = Checker::fall(level, nx, uy, |x, y| {
                if nv < HEIGHT {
                    visited[nv] = (x, y);
                    nv += 1;
                }
            }) {
                for &(x, y) in &visited[..nv] {
                    self.touch(x, y);
                }
                self.relax(to, STEP_COST, u, Move::Walk);
            }
        }

        // 2. Escaleras: subir (si estoy en una o la de encima lo es) y bajar.
        let on_ladder = level.get(ux as usize, uy as usize) == Tile::Ladder;
        let up = level.get_or_empty(ux, uy - 1);
        if uy > 0 && (on_ladder || up == Tile::Ladder) && Checker::is_rest(level, ux, uy - 1) {
            self.relax((ux, uy - 1), STEP_COST, u, Move::Climb);
        }
        let down = level.get_or_empty(ux, uy + 1);
        if down == Tile::Ladder {
            self.relax((ux, uy + 1), STEP_COST, u, Move::Climb);
        } else if on_ladder
            && down.is_passable()
            && uy + 1 < HEIGHT as i32
            && let Some(to) = Checker::fall(level, ux, uy + 1, |_, _| {})
        {
            self.relax(to, STEP_COST, u, Move::Climb);
        }

        // 3. Saltos (desde el suelo o desde una escalera).
        for (ai, arc) in self.checker.arcs.iter().enumerate() {
            let mut buf = [(0, 0); 64];
            let mut n = 0;
            let landing = Checker::follow_arc(level, arc, ux, uy, |x, y| {
                if n < buf.len() {
                    buf[n] = (x, y);
                    n += 1;
                }
            });
            if let Some(to) = landing
                && to != (ux, uy)
            {
                for &(x, y) in &buf[..n] {
                    self.touch(x, y);
                }
                self.relax(to, jump_cost(to.0 - ux), u, Move::Jump(ai as u16));
            }
        }
    }

    /// Reconstruye el camino y calcula las estadísticas finales.
    fn finish(self, s0: (i32, i32), goal: (usize, usize)) -> PlayResult {
        let sc = &*self.sc;
        let ep = sc.epoch;
        let (gx, gy) = self.goal;
        let mut res = PlayResult {
            reachable_coins: self.coins,
            progress: if gx > s0.0 {
                (f64::from(self.max_x - s0.0) / f64::from(gx - s0.0)).clamp(0.0, 1.0)
            } else {
                1.0
            },
            ..PlayResult::default()
        };
        // Destino del camino: el estado de meta o, si la meta solo se tocó al
        // vuelo, el estado de reposo alcanzado más cercano a ella.
        let goal_touched = sc.touched[idx(gx, gy) as usize] == ep;
        let target = self.goal_state.or_else(|| {
            goal_touched.then(|| {
                (0..self.level.width() as i32)
                    .flat_map(|x| (0..HEIGHT as i32).map(move |y| (x, y)))
                    .filter(|&(x, y)| sc.seen[idx(x, y) as usize] == ep)
                    .min_by_key(|&(x, y)| {
                        ((x - gx).abs() + (y - gy).abs(), sc.dist[idx(x, y) as usize])
                    })
                    .map_or(idx(s0.0, s0.1), |(x, y)| idx(x, y))
            })
        });
        let Some(t) = target else {
            return res;
        };
        res.solvable = true;
        res.progress = 1.0;
        let mut cur = t;
        while cur != u32::MAX {
            res.path
                .push(((cur as usize) / HEIGHT, (cur as usize) % HEIGHT));
            if sc.prev[cur as usize] != u32::MAX {
                res.moves.push(sc.prev_move[cur as usize]);
            }
            cur = sc.prev[cur as usize];
        }
        res.path.reverse();
        res.moves.reverse();
        let ph = self.checker.physics;
        for m in &res.moves {
            if let Move::Jump(a) = *m {
                res.jumps += 1;
                // "Ajustado": el salto más pequeño que basta está cerca del límite
                // (≥ 80 % de la altura y de la longitud máximas).
                let arc = &self.checker.arcs[usize::from(a)];
                let tight_h = u32::from(arc.apex) * 5 >= u32::from(ph.jump_height) * 4;
                let tight_r = u32::from(arc.run) * 5 >= u32::from(ph.jump_length) * 4;
                if tight_h || tight_r {
                    res.tight_jumps += 1;
                }
            }
        }
        if self.goal_state.is_none() && res.path.last() != Some(&goal) {
            res.path.push(goal);
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lvl(rows: &[&str]) -> Level {
        let mut text = String::new();
        let pad = HEIGHT - rows.len();
        let w = rows[0].len();
        for _ in 0..pad {
            text.push_str(&"-".repeat(w));
            text.push('\n');
        }
        for r in rows {
            text.push_str(r);
            text.push('\n');
        }
        Level::parse(&text).expect("nivel de prueba válido")
    }

    fn check(l: &Level) -> PlayResult {
        Checker::default().check(l, &mut PlayScratch::default())
    }

    #[test]
    fn llano_se_camina_sin_saltar() {
        let r = check(&lvl(&["-@--------!-", "XXXXXXXXXXXX"]));
        assert!(r.solvable);
        assert_eq!(r.jumps, 0);
    }

    #[test]
    fn hueco_saltable_y_no_saltable() {
        // Hueco de 3: con jump_length = 4 se salta.
        let r = check(&lvl(&["-@-------!--", "XXXX---XXXXX"]));
        assert!(r.solvable, "{r:?}");
        assert_eq!(r.jumps, 1);
        // Hueco de 7: imposible; el progreso se queda antes del hueco.
        let r = check(&lvl(&["-@--------!-", "XXX-------XX"]));
        assert!(!r.solvable);
        assert!(r.progress < 0.5);
    }

    #[test]
    fn muro_alto_requiere_escalera() {
        let alto = [
            "----------X--",
            "----------X--",
            "----------X--",
            "----------X--",
            "-@--------X!-",
            "XXXXXXXXXXXXX",
        ];
        assert!(!check(&lvl(&alto)).solvable);
        let con_escalera = [
            "----------X--",
            "---------LX--",
            "---------LX--",
            "---------LX--",
            "-@-------LX!-",
            "XXXXXXXXXXXXX",
        ];
        // La escalera llega hasta arriba; desde su tope se salta al otro lado.
        let r = check(&lvl(&con_escalera));
        assert!(r.solvable, "{r:?}");
    }

    #[test]
    fn muro_de_tres_se_salta() {
        let r = check(&lvl(&[
            "------XX----",
            "------XX----",
            "-@----XX--!-",
            "XXXXXXXXXXXX",
        ]));
        assert!(r.solvable);
        assert!(r.jumps >= 1);
    }

    #[test]
    fn enemigo_hay_que_saltarlo_y_pinchos_matan() {
        let r = check(&lvl(&["-@---E----!-", "XXXXXXXXXXXX"]));
        assert!(r.solvable);
        assert_eq!(r.jumps, 1);
        // Techo bajo en todo el ancho: no se puede saltar al enemigo.
        let r = check(&lvl(&["XXXXXXXXXXXX", "-@---E----!-", "XXXXXXXXXXXX"]));
        assert!(!r.solvable);
        // Techo parcial: se sube encima del techo y se rodea (solución legítima).
        let r = check(&lvl(&["--XXXXXXX---", "-@---E----!-", "XXXXXXXXXXXX"]));
        assert!(r.solvable);
        // Pinchos en un pasillo de una fila de alto: imposible.
        let r = check(&lvl(&["XXXXXXXXXXXX", "-@---^----!-", "XXXXXXXXXXXX"]));
        assert!(!r.solvable);
    }

    #[test]
    fn arcos_simetricos_y_acotados() {
        let arcs = build_arcs(Physics::default());
        assert_eq!(arcs.len(), 3 + 2 * 3 * 4);
        for a in &arcs {
            let top = a.steps.iter().map(|s| s.dy).min().unwrap_or(0);
            assert_eq!(-i32::from(top), i32::from(a.apex), "{a:?}");
            // Al volver a la altura de despegue ha recorrido `run` columnas
            // (o una menos: el cuerpo redondea a la celda más cercana).
            let back = a
                .steps
                .iter()
                .find(|s| s.falling && s.dy == 0)
                .expect("vuelve a bajar");
            let (dx, run) = (i32::from(back.dx).abs(), i32::from(a.run));
            assert!(dx == run || dx + 1 == run, "{a:?}");
        }
    }

    #[test]
    fn trayectoria_continua() {
        let l = lvl(&["-@--------!-", "XXXX---XXXXX"]);
        let c = Checker::default();
        let r = c.check(&l, &mut PlayScratch::default());
        let t = c.trajectory(&l, &r);
        assert_eq!(t.first(), r.path.first());
        assert_eq!(t.last(), r.path.last());
        // Celdas consecutivas adyacentes (8-vecindad).
        for w in t.windows(2) {
            let (dx, dy) = (w[0].0.abs_diff(w[1].0), w[0].1.abs_diff(w[1].1));
            assert!(dx <= 1 && dy <= 1, "{w:?}");
        }
    }

    #[test]
    fn determinista_y_reutiliza_memoria() {
        let l = lvl(&["-@---E-o--!-", "XXXX--XXXXXX"]);
        let c = Checker::default();
        let mut s = PlayScratch::default();
        let a = c.check(&l, &mut s);
        let b = c.check(&l, &mut s);
        assert_eq!(a, b);
        assert_eq!(a.reachable_coins, 1);
    }
}
