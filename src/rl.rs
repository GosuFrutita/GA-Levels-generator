//! Aprendizaje por refuerzo: un agente que **juega** los niveles.
//!
//! El entorno usa exactamente los movimientos del verificador de jugabilidad
//! (caminar con caída, escaleras y los arcos de salto precalculados), así que
//! todo lo que el agente puede hacer también lo puede hacer el BFS.
//!
//! El agente es un actor-crítico entrenado con A2C (Mnih et al., 2016): `N`
//! entornos en paralelo, retornos de `t_max` pasos, `γ = 0,99` y bono de
//! entropía `β = 0,01`. La pérdida de la política es la cross-entropy entre la
//! acción tomada (one-hot) y la salida softmax, ponderada por la ventaja:
//! `L = −Â · log π(a|s) − β·H(π)`; su gradiente respecto a los logits es
//! `Â·(π − onehot(a))`, el mismo código que la red de naturalidad.
//!
//! Usos: medir la **dificultad empírica** de un nivel (tasa de éxito del
//! agente) y servir de probador automático junto al BFS.

use crate::error::{Error, Result};
use crate::fitness::Scorer;
use crate::level::{HEIGHT, Level, Tile};
use crate::nn::loss::{cross_entropy, entropy_bonus, softmax};
use crate::nn::{Activation, Adam, Input, Net, NetSpec, Tape};
use crate::play::Checker;
use crate::rng::{Rng, mix_seed};

/// Columnas de la ventana de observación detrás del jugador.
pub const BACK: i32 = 3;
/// Columnas de la ventana de observación delante del jugador.
pub const AHEAD: i32 = 5;
/// Columnas de la ventana.
pub const COLS: usize = (BACK + 1 + AHEAD) as usize;
/// Posiciones categóricas de la observación (todo el alto × ventana).
pub const OBS_POS: usize = HEIGHT * COLS;
/// Categorías por posición: los tipos de tile más el borde.
pub const OBS_CATS: usize = Tile::COUNT + 1;
/// Rasgos densos: fila one-hot y si está en una escalera. No se incluyen el
/// progreso ni la distancia a la meta: permitían memorizar cada nivel y
/// empeoraban la generalización (las observaciones locales generalizan mejor,
/// como observan Earle et al. en PCGRL+).
pub const OBS_DENSE: usize = HEIGHT + 1;

/// Acción del agente.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Caminar una columna (−1 o +1).
    Walk(i32),
    /// Subir por una escalera.
    Up,
    /// Bajar por una escalera o soltarse.
    Down,
    /// Saltar con el arco de índice dado.
    Jump(usize),
}

/// Construye la tabla de acciones: caminar a cada lado, subir, bajar y **todos**
/// los arcos de salto del verificador. Así el agente puede hacer exactamente
/// lo mismo que el BFS: si el BFS dice que un nivel es jugable, existe una
/// secuencia de acciones del agente que lo resuelve.
///
/// # Panics
///
/// Si la física no produce exactamente `ACTIONS − 4` arcos (la arquitectura
/// de la red depende de ese número; vale para la física por defecto).
pub fn action_table(checker: &Checker) -> Vec<Action> {
    assert_eq!(
        checker.arcs().len(),
        ACTIONS - 4,
        "el agente asume la física por defecto"
    );
    let mut a = vec![Action::Walk(-1), Action::Walk(1), Action::Up, Action::Down];
    a.extend((0..checker.arcs().len()).map(Action::Jump));
    a
}

/// Número de acciones con la física por defecto (4 + 27 arcos).
pub const ACTIONS: usize = 4 + 27;

/// Resultado de un paso.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Step {
    /// Recompensa.
    pub reward: f64,
    /// ¿Terminó el episodio?
    pub done: bool,
}

/// Entorno de un nivel.
#[derive(Debug, Clone)]
pub struct Env<'a> {
    level: Level,
    checker: &'a Checker,
    actions: &'a [Action],
    /// Posición actual.
    pub pos: (i32, i32),
    start_x: i32,
    goal: (i32, i32),
    max_x: i32,
    /// Pasos dados.
    pub steps: usize,
    max_steps: usize,
    /// ¿Llegó a la meta?
    pub won: bool,
    /// ¿Murió?
    pub died: bool,
    /// Celdas recorridas en el último paso (para animar).
    pub trail: Vec<(i32, i32)>,
}

/// Recompensa por paso (coste del tiempo).
pub const STEP_PENALTY: f64 = -0.01;

impl<'a> Env<'a> {
    /// Crea el entorno; `None` si el nivel no tiene salida o meta válidas.
    pub fn new(level: &Level, checker: &'a Checker, actions: &'a [Action]) -> Option<Self> {
        // Misma física que el verificador: las zonas de patrulla matan.
        let level = level.with_patrols();
        let s = level.start()?;
        let g = level.goal()?;
        let start = Checker::fall(&level, s.0 as i32, s.1 as i32, |_, _| {})?;
        let max_steps = 3 * level.width();
        Some(Self {
            level,
            checker,
            actions,
            pos: start,
            start_x: start.0,
            goal: (g.0 as i32, g.1 as i32),
            max_x: start.0,
            steps: 0,
            max_steps,
            won: false,
            died: false,
            trail: Vec::new(),
        })
    }

    /// Progreso horizontal en `[0, 1]`.
    pub fn progress(&self) -> f64 {
        let span = (self.goal.0 - self.start_x).max(1);
        (f64::from(self.max_x - self.start_x) / f64::from(span)).clamp(0.0, 1.0)
    }

    /// Escribe la observación: ventana egocéntrica de todo el alto y
    /// [`COLS`] columnas (más tiles por delante que por detrás) y rasgos densos.
    pub fn observe(&self, cats: &mut [u16; OBS_POS], dense: &mut [f32; OBS_DENSE]) {
        let w = self.level.width() as i32;
        let mut k = 0;
        for dx in -BACK..=AHEAD {
            let x = self.pos.0 + dx;
            for y in 0..HEIGHT {
                cats[k] = if x < 0 || x >= w {
                    Tile::COUNT as u16
                } else {
                    self.level.get(x as usize, y) as u16
                };
                k += 1;
            }
        }
        dense.fill(0.0);
        dense[self.pos.1 as usize] = 1.0;
        dense[HEIGHT] = f32::from(u8::from(
            self.level.get(self.pos.0 as usize, self.pos.1 as usize) == Tile::Ladder,
        ));
    }

    /// Aplica una acción.
    pub fn step(&mut self, action: usize) -> Step {
        self.steps += 1;
        self.trail.clear();
        let (x, y) = self.pos;
        let level = &self.level;
        let goal = self.goal;
        let mut touched_goal = false;
        let mut trail = std::mem::take(&mut self.trail);
        let mut visit = |cx: i32, cy: i32| {
            trail.push((cx, cy));
            touched_goal |= (cx, cy) == goal;
        };
        // `Some(Some(p))` = se mueve a p; `Some(None)` = muere; `None` = bloqueado.
        let outcome: Option<Option<(i32, i32)>> = match self.actions[action] {
            Action::Walk(dx) => {
                let nx = x + dx;
                if nx < 0
                    || nx as usize >= level.width()
                    || level.get(nx as usize, y as usize).is_solid()
                {
                    None
                } else if level.get(nx as usize, y as usize).is_lethal() {
                    Some(None)
                } else {
                    Some(Checker::fall(level, nx, y, &mut visit))
                }
            }
            Action::Up => {
                let here = level.get(x as usize, y as usize);
                let up = level.get_or_empty(x, y - 1);
                (y > 0
                    && (here == Tile::Ladder || up == Tile::Ladder)
                    && Checker::is_rest(level, x, y - 1))
                .then_some(Some((x, y - 1)))
            }
            Action::Down => {
                let here = level.get(x as usize, y as usize);
                let down = level.get_or_empty(x, y + 1);
                if down == Tile::Ladder {
                    Some(Some((x, y + 1)))
                } else if here == Tile::Ladder && down.is_passable() && y + 1 < HEIGHT as i32 {
                    Some(Checker::fall(level, x, y + 1, &mut visit))
                } else {
                    None
                }
            }
            Action::Jump(i) => Some(Checker::follow_arc(
                level,
                &self.checker.arcs()[i],
                x,
                y,
                &mut visit,
            )),
        };
        self.trail = trail;
        let mut reward = STEP_PENALTY;
        match outcome {
            None => {}
            Some(None) => {
                self.died = true;
                return Step {
                    reward: reward - 1.0,
                    done: true,
                };
            }
            Some(Some(p)) => {
                self.pos = p;
                touched_goal |= p == self.goal;
            }
        }
        let before = self.progress();
        self.max_x = self.max_x.max(self.pos.0);
        if touched_goal {
            self.max_x = self.goal.0;
        }
        reward += self.progress() - before;
        if touched_goal {
            self.won = true;
            return Step {
                reward: reward + 1.0,
                done: true,
            };
        }
        Step {
            reward,
            done: self.steps >= self.max_steps,
        }
    }
}

/// Agente actor-crítico.
#[derive(Debug, Clone)]
pub struct Agent {
    /// Red con cabeza de política y de valor.
    pub net: Net<f32>,
}

impl Agent {
    /// Agente nuevo con pesos aleatorios.
    pub fn new(hidden: Vec<usize>, rng: &mut Rng) -> Self {
        let spec = NetSpec {
            positions: OBS_POS,
            categories: OBS_CATS,
            dense: OBS_DENSE,
            hidden,
            outputs: ACTIONS,
            value_head: true,
            activation: Activation::Relu,
        };
        Self {
            net: Net::new(spec, rng),
        }
    }

    /// Elige una acción muestreando la política (o la más probable si `greedy`).
    pub fn act(&self, env: &Env<'_>, tape: &mut Tape<f32>, rng: &mut Rng, greedy: bool) -> usize {
        let mut cats = [0; OBS_POS];
        let mut dense = [0.0; OBS_DENSE];
        env.observe(&mut cats, &mut dense);
        self.net.forward(
            Input {
                cats: &cats,
                dense: &dense,
            },
            tape,
        );
        if greedy {
            return tape
                .logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |p| p.0);
        }
        let mut p = [0.0_f32; ACTIONS];
        softmax(&tape.logits, &mut p);
        sample(&p, rng)
    }

    /// Juega un episodio; devuelve `(ganó, progreso, pasos)`.
    pub fn play(
        &self,
        level: &Level,
        checker: &Checker,
        actions: &[Action],
        rng: &mut Rng,
        greedy: bool,
    ) -> (bool, f64, usize) {
        let Some(mut env) = Env::new(level, checker, actions) else {
            return (false, 0.0, 0);
        };
        let mut tape = self.net.tape();
        loop {
            let a = self.act(&env, &mut tape, rng, greedy);
            if env.step(a).done {
                return (env.won, env.progress(), env.steps);
            }
        }
    }

    /// Trayectoria completa de un episodio (para visualizarla).
    pub fn trajectory(
        &self,
        level: &Level,
        checker: &Checker,
        rng: &mut Rng,
    ) -> (Vec<(usize, usize)>, bool) {
        let actions = action_table(checker);
        let mut out = Vec::new();
        let Some(mut env) = Env::new(level, checker, &actions) else {
            return (out, false);
        };
        let mut tape = self.net.tape();
        out.push((env.pos.0 as usize, env.pos.1 as usize));
        loop {
            let a = self.act(&env, &mut tape, rng, false);
            let s = env.step(a);
            for &(x, y) in &env.trail {
                if y >= 0 {
                    out.push((x as usize, y as usize));
                }
            }
            out.push((env.pos.0 as usize, env.pos.1 as usize));
            if s.done {
                out.dedup();
                return (out, env.won);
            }
        }
    }

    /// Serializa el agente.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = b"RLA1".to_vec();
        out.extend_from_slice(&self.net.to_bytes());
        out
    }

    /// Lee un agente escrito con [`Agent::to_bytes`].
    ///
    /// # Errors
    ///
    /// Si el formato es inválido o la arquitectura no coincide con el entorno.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let rest = bytes
            .strip_prefix(b"RLA1")
            .ok_or_else(|| Error::format("agente", "firma incorrecta"))?;
        let net: Net<f32> = Net::from_bytes(rest)?;
        let s = net.spec();
        if s.positions != OBS_POS
            || s.categories != OBS_CATS
            || s.dense != OBS_DENSE
            || s.outputs != ACTIONS
        {
            return Err(Error::format(
                "agente",
                "arquitectura incompatible con el entorno",
            ));
        }
        Ok(Self { net })
    }
}

/// Muestrea un índice de una distribución discreta.
fn sample(p: &[f32], rng: &mut Rng) -> usize {
    let mut r = rng.f64() as f32;
    for (i, &pi) in p.iter().enumerate() {
        r -= pi;
        if r <= 0.0 {
            return i;
        }
    }
    p.len() - 1
}

/// Hiperparámetros de A2C (valores de Mnih et al., 2016, salvo el optimizador).
#[derive(Debug, Clone)]
pub struct A2cConfig {
    /// Entornos simultáneos.
    pub envs: usize,
    /// Pasos por actualización (`t_max`).
    pub t_max: usize,
    /// Descuento.
    pub gamma: f64,
    /// Coeficiente del bono de entropía `β`.
    pub entropy: f64,
    /// Peso de la pérdida del crítico.
    pub value_coef: f64,
    /// Tasa de aprendizaje (Adam).
    pub lr: f64,
    /// Norma máxima del gradiente.
    pub clip: f64,
    /// Pasos de entorno totales.
    pub total_steps: usize,
    /// Neuronas ocultas.
    pub hidden: Vec<usize>,
    /// Semilla.
    pub seed: u64,
    /// Ancho de los niveles al empezar el currículo.
    pub min_width: usize,
    /// Ancho final de los niveles.
    pub max_width: usize,
    /// Probabilidad de aceptar un nivel injugable como episodio.
    pub unsolvable: f64,
    /// Normalizar las ventajas de cada lote.
    pub normalize_advantages: bool,
}

impl Default for A2cConfig {
    fn default() -> Self {
        Self {
            envs: 16,
            t_max: 5,
            gamma: 0.99,
            entropy: 0.01,
            value_coef: 0.5,
            lr: 7e-4,
            clip: 0.5,
            total_steps: 2_000_000,
            hidden: vec![64, 64],
            seed: 0,
            min_width: 20,
            max_width: 60,
            unsolvable: 0.1,
            normalize_advantages: true,
        }
    }
}

/// Registro periódico del entrenamiento.
#[derive(Debug, Clone, Copy, Default)]
pub struct TrainLog {
    /// Pasos de entorno acumulados.
    pub steps: usize,
    /// Episodios terminados en la ventana.
    pub episodes: usize,
    /// Tasa de victorias en la ventana.
    pub win_rate: f64,
    /// Progreso medio en la ventana.
    pub progress: f64,
    /// Retorno medio en la ventana.
    pub ret: f64,
    /// Entropía media de la política.
    pub entropy: f64,
}

/// Una transición guardada para la actualización.
struct Transition {
    cats: [u16; OBS_POS],
    dense: [f32; OBS_DENSE],
    action: usize,
    reward: f64,
    done: bool,
    value: f64,
}

/// Aumento de datos: aplica a un nivel del GA de 2 a 4 mutaciones macro y unos
/// 10 cambios puntuales antes de repararlo. Sin esto el agente memorizaba los
/// niveles del GA (48 % de victorias en los vistos frente a 2 % en los nuevos).
fn augment(level: &mut Level, rng: &mut Rng) {
    use crate::ga::variation::{MacroOp, macro_mutation, point_mutation};
    for _ in 0..2 + rng.index(3) {
        let op = MacroOp::ALL[rng.index(MacroOp::ALL.len())];
        macro_mutation(level, op, rng);
    }
    let cells = ((level.width() - 2 * crate::level::SAFE_COLS) * HEIGHT) as f64;
    point_mutation(level, 10.0 / cells, rng);
    crate::ga::init::repair(level, true);
}

/// Estado del entrenamiento A2C.
struct Trainer<'a> {
    levels: &'a [Level],
    checker: &'a Checker,
    actions: &'a [Action],
    cfg: &'a A2cConfig,
    rng: Rng,
    agent: Agent,
    tape: Tape<f32>,
    grads: Vec<f32>,
    returns: Vec<f64>,
    buf: Vec<Vec<Transition>>,
    window: TrainLog,
    ent_sum: f64,
    ent_n: usize,
    steps: usize,
    scratch: crate::play::PlayScratch,
}

impl<'a> Trainer<'a> {
    /// Nivel del siguiente episodio. Currículo de **longitud**: el ancho crece
    /// linealmente de `min_width` a `max_width` durante la primera mitad del
    /// entrenamiento. La mitad de los niveles son ventanas recortadas de los
    /// niveles del GA con mutaciones (ver [`augment`]). La otra mitad se genera
    /// al azar en el momento. Así el agente casi nunca ve dos veces el mismo
    /// nivel y no puede memorizarlos.
    /// Se descartan los injugables (según el BFS) salvo una fracción
    /// `unsolvable`, para que también aprenda a progresar donde no se puede ganar.
    fn next_level(&mut self) -> Level {
        let half = (self.cfg.total_steps / 2).max(1) as f64;
        let t = (self.steps as f64 / half).min(1.0);
        let (lo, hi) = (self.cfg.min_width as f64, self.cfg.max_width as f64);
        let width = (lo + (hi - lo) * t).round() as usize;
        loop {
            let level = if !self.levels.is_empty() && self.rng.chance(0.5) {
                let src = &self.levels[self.rng.index(self.levels.len())];
                let w = width.min(src.width());
                let x0 = self.rng.index(src.width() - w + 1);
                let mut l = crate::ga::init::window(src, x0, w);
                augment(&mut l, &mut self.rng);
                l
            } else {
                crate::ga::init::random_level(width, &mut self.rng)
            };
            let solvable = self.checker.check(&level, &mut self.scratch).solvable;
            if solvable || self.rng.chance(self.cfg.unsolvable) {
                return level;
            }
        }
    }

    fn spawn(&mut self) -> Env<'a> {
        loop {
            let level = self.next_level();
            if let Some(e) = Env::new(&level, self.checker, self.actions) {
                return e;
            }
        }
    }

    fn observe_value(&mut self, env: &Env<'_>) -> f64 {
        let mut cats = [0; OBS_POS];
        let mut dense = [0.0; OBS_DENSE];
        env.observe(&mut cats, &mut dense);
        self.agent.net.forward(
            Input {
                cats: &cats,
                dense: &dense,
            },
            &mut self.tape,
        );
        f64::from(self.tape.value)
    }

    /// Recoge hasta `t_max` pasos de cada entorno (se corta al terminar un episodio).
    fn collect(&mut self, envs: &mut [Env<'a>]) {
        let mut probs = [0.0_f32; ACTIONS];
        for e in 0..envs.len() {
            self.buf[e].clear();
            for _ in 0..self.cfg.t_max {
                let mut tr = Transition {
                    cats: [0; OBS_POS],
                    dense: [0.0; OBS_DENSE],
                    action: 0,
                    reward: 0.0,
                    done: false,
                    value: 0.0,
                };
                envs[e].observe(&mut tr.cats, &mut tr.dense);
                self.agent.net.forward(
                    Input {
                        cats: &tr.cats,
                        dense: &tr.dense,
                    },
                    &mut self.tape,
                );
                softmax(&self.tape.logits, &mut probs);
                tr.action = sample(&probs, &mut self.rng);
                tr.value = f64::from(self.tape.value);
                let s = envs[e].step(tr.action);
                tr.reward = s.reward;
                tr.done = s.done;
                self.returns[e] += s.reward;
                self.steps += 1;
                self.buf[e].push(tr);
                if s.done {
                    let env = &envs[e];
                    self.window.episodes += 1;
                    self.window.win_rate += f64::from(u8::from(env.won));
                    self.window.progress += env.progress();
                    self.window.ret += self.returns[e];
                    self.returns[e] = 0.0;
                    envs[e] = self.spawn();
                    break;
                }
            }
        }
    }

    /// Retornos de `n` pasos con *bootstrap* del crítico, gradientes y paso de Adam.
    fn update(&mut self, envs: &[Env<'a>], opt: &mut Adam<f32>) {
        // 1. Retornos y ventajas de cada transición.
        let mut targets: Vec<Vec<(f64, f64)>> = Vec::with_capacity(envs.len());
        for (e, env) in envs.iter().enumerate() {
            let mut ret = if self.buf[e].last().is_some_and(|t| t.done) {
                0.0
            } else {
                self.observe_value(env)
            };
            let mut t: Vec<(f64, f64)> = self.buf[e]
                .iter()
                .rev()
                .map(|tr| {
                    ret = tr.reward + self.cfg.gamma * ret * f64::from(u8::from(!tr.done));
                    (ret, ret - tr.value)
                })
                .collect();
            t.reverse();
            targets.push(t);
        }
        // 2. Normalización de las ventajas del lote (media 0, desviación 1):
        //    hace que la escala del término de política no dependa de la de
        //    las recompensas y no quede ahogada por el bono de entropía.
        let all: Vec<f64> = targets.iter().flatten().map(|t| t.1).collect();
        let n = all.len() as f64;
        let mean = all.iter().sum::<f64>() / n;
        let sd = (all.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / n).sqrt();
        let (mean, sd) = if self.cfg.normalize_advantages {
            (mean, sd.max(1e-8))
        } else {
            (0.0, 1.0)
        };
        // 3. Gradientes.
        self.grads.fill(0.0);
        let scale = 1.0 / n as f32;
        for (e, tgt) in targets.iter().enumerate() {
            let tail = std::mem::take(&mut self.buf[e]);
            for (tr, &(ret, adv)) in tail.iter().zip(tgt) {
                let adv = ((adv - mean) / sd) as f32;
                self.agent.net.forward(
                    Input {
                        cats: &tr.cats,
                        dense: &tr.dense,
                    },
                    &mut self.tape,
                );
                let mut dz = [0.0_f32; ACTIONS];
                // Política: cross-entropy ponderada por la ventaja.
                cross_entropy(&self.tape.logits, tr.action, 0.0, adv * scale, &mut dz);
                self.ent_sum += f64::from(entropy_bonus(
                    &self.tape.logits,
                    self.cfg.entropy as f32 * scale,
                    &mut dz,
                ));
                self.ent_n += 1;
                // Crítico: ½·c·(V − R)².
                let dv = self.cfg.value_coef as f32 * (self.tape.value - ret as f32) * scale;
                self.agent
                    .net
                    .backward(&mut self.tape, &dz, dv, &mut self.grads);
            }
            self.buf[e] = tail;
        }
        opt.step(&mut self.agent.net.params, &mut self.grads);
    }

    fn take_log(&mut self) -> TrainLog {
        let n = self.window.episodes.max(1) as f64;
        let entry = TrainLog {
            steps: self.steps,
            episodes: self.window.episodes,
            win_rate: self.window.win_rate / n,
            progress: self.window.progress / n,
            ret: self.window.ret / n,
            entropy: self.ent_sum / self.ent_n.max(1) as f64,
        };
        self.window = TrainLog::default();
        self.ent_sum = 0.0;
        self.ent_n = 0;
        entry
    }
}

/// Entrena el agente con A2C. Cada episodio usa un nivel nuevo: una ventana de
/// uno de `levels` (p. ej. niveles del GA) o un nivel aleatorio, con un
/// currículo de longitud creciente (ver `Trainer::next_level`).
///
pub fn train(
    levels: &[Level],
    checker: &Checker,
    cfg: &A2cConfig,
    mut log: impl FnMut(&TrainLog),
) -> (Agent, Vec<TrainLog>) {
    const LOG_EVERY: usize = 50_000;
    let actions = action_table(checker);
    let mut rng = Rng::seed_from_u64(cfg.seed);
    let agent = Agent::new(cfg.hidden.clone(), &mut rng);
    let mut opt = Adam::new(agent.net.num_params(), cfg.lr);
    opt.clip_norm = Some(cfg.clip);
    let mut t = Trainer {
        levels,
        checker,
        actions: &actions,
        cfg,
        rng,
        tape: agent.net.tape(),
        grads: vec![0.0; agent.net.num_params()],
        agent,
        returns: vec![0.0; cfg.envs],
        buf: (0..cfg.envs)
            .map(|_| Vec::with_capacity(cfg.t_max))
            .collect(),
        window: TrainLog::default(),
        ent_sum: 0.0,
        ent_n: 0,
        steps: 0,
        scratch: crate::play::PlayScratch::default(),
    };
    let mut envs: Vec<Env<'_>> = (0..cfg.envs).map(|_| t.spawn()).collect();
    let mut history = Vec::new();
    let mut next_log = LOG_EVERY.min(cfg.total_steps);
    while t.steps < cfg.total_steps {
        t.collect(&mut envs);
        t.update(&envs, &mut opt);
        if t.steps >= next_log {
            next_log += LOG_EVERY;
            let entry = t.take_log();
            log(&entry);
            history.push(entry);
        }
    }
    (t.agent, history)
}

/// Resultado de evaluar el agente en un nivel.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Playtest {
    /// Fracción de episodios ganados.
    pub win_rate: f64,
    /// Progreso medio.
    pub progress: f64,
    /// Pasos medios.
    pub steps: f64,
}

/// Semilla determinista derivada del contenido del nivel (FNV-1a).
pub fn level_seed(level: &Level) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for &t in level.tiles() {
        h = (h ^ u64::from(t as u8)).wrapping_mul(0x0100_0000_01b3);
    }
    h ^ level.style as u64
}

/// Juega `episodes` episodios estocásticos en un nivel.
pub fn playtest(
    agent: &Agent,
    level: &Level,
    checker: &Checker,
    episodes: usize,
    seed: u64,
) -> Playtest {
    let actions = action_table(checker);
    let mut rng = Rng::seed_from_u64(mix_seed(&[seed, level_seed(level)]));
    let mut t = Playtest::default();
    for _ in 0..episodes {
        let (won, prog, steps) = agent.play(level, checker, &actions, &mut rng, false);
        t.win_rate += f64::from(u8::from(won));
        t.progress += prog;
        t.steps += steps as f64;
    }
    let n = episodes.max(1) as f64;
    Playtest {
        win_rate: t.win_rate / n,
        progress: t.progress / n,
        steps: t.steps / n,
    }
}

/// Puntuación de "dificultad empírica moderada" para el GA: campana sobre la
/// tasa de victorias del agente, centrada en `target` (González-Duque et al.,
/// 2020, buscan niveles con un 60 % de victorias).
#[derive(Debug, Clone)]
pub struct PlaytesterScorer {
    /// Agente.
    pub agent: Agent,
    /// Verificador (define los arcos de las acciones).
    pub checker: Checker,
    /// Episodios por nivel.
    pub episodes: usize,
    /// Tasa de victorias objetivo.
    pub target: f64,
    /// Tolerancia de la campana.
    pub sigma: f64,
}

impl Scorer for PlaytesterScorer {
    fn score(&self, level: &Level) -> f64 {
        let t = playtest(&self.agent, level, &self.checker, self.episodes, 0);
        crate::metrics::bell(t.win_rate, self.target, self.sigma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::level::Style;

    fn flat(width: usize, gap: Option<usize>) -> Level {
        let mut l = Level::empty(width, Style::Base);
        for x in 0..width {
            if gap.is_none_or(|g| !(g..g + 3).contains(&x)) {
                l.set(x, HEIGHT - 1, Tile::Ground);
            }
        }
        l.set(1, HEIGHT - 2, Tile::Start);
        l.set(width - 2, HEIGHT - 2, Tile::Goal);
        l
    }

    #[test]
    fn el_entorno_respeta_la_fisica() {
        let checker = Checker::default();
        let actions = action_table(&checker);
        assert_eq!(actions.len(), ACTIONS);
        let l = flat(20, Some(8));
        let mut env = Env::new(&l, &checker, &actions).expect("válido");
        // Caminar a la derecha hasta el borde del hueco.
        while env.pos.0 < 7 {
            assert!(!env.step(1).done);
        }
        // Caer al hueco mata.
        let s = env.step(1);
        assert!(s.done && env.died && s.reward < -0.9);
        // Saltar el hueco y llegar a la meta.
        let mut env = Env::new(&l, &checker, &actions).expect("válido");
        while env.pos.0 < 7 {
            env.step(1);
        }
        let long_right = 4 + checker
            .arcs()
            .iter()
            .position(|a| a.dir == 1 && a.apex == 3 && a.run == 4)
            .expect("arco largo");
        assert!(!env.step(long_right).done);
        assert!(env.pos.0 >= 11, "{:?}", env.pos);
        while !env.won {
            assert!(!env.died);
            env.step(1);
        }
        assert!(env.progress() >= 1.0 - 1e-12);
    }

    #[test]
    fn a2c_aprende_a_cruzar_un_nivel_facil() {
        let checker = Checker::default();
        let levels = vec![flat(24, Some(10)), flat(24, None)];
        let cfg = A2cConfig {
            total_steps: 60_000,
            hidden: vec![32],
            seed: 3,
            min_width: 24,
            max_width: 24,
            unsolvable: 0.0,
            ..A2cConfig::default()
        };
        let (agent, hist) = train(&levels, &checker, &cfg, |_| {});
        assert!(!hist.is_empty());
        let t = playtest(&agent, &levels[0], &checker, 20, 1);
        assert!(t.win_rate > 0.5, "{t:?}");
        let back = Agent::from_bytes(&agent.to_bytes()).expect("válido");
        assert_eq!(back.net.params, agent.net.params);
    }
}
