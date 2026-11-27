//! Perceptrón multicapa con entrada categórica por posición.
//!
//! La primera capa recibe, para cada posición `p`, una categoría `c` (un tile)
//! y suma la fila `W[p][c]`: equivale a multiplicar un vector *one-hot* por la
//! matriz, pero en `O(posiciones · h)` en vez de `O(posiciones · categorías · h)`.
//! Opcionalmente se añaden rasgos densos. Después vienen capas ocultas densas,
//! una cabeza de *logits* y una cabeza de valor opcional (actor-crítico).
//!
//! Todos los parámetros viven en un único `Vec<T>` contiguo; así el optimizador,
//! el recorte de gradiente y la serialización operan sobre un solo slice.

use super::Float;
use crate::error::{Error, Result};
use crate::rng::Rng;

/// Categoría ausente: la posición no aporta nada (relleno).
pub const NONE: u16 = u16::MAX;

/// Función de activación de las capas ocultas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Tangente hiperbólica.
    Tanh,
    /// Rectificador lineal.
    Relu,
}

/// Arquitectura de la red.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetSpec {
    /// Número de posiciones categóricas de entrada.
    pub positions: usize,
    /// Número de categorías por posición.
    pub categories: usize,
    /// Número de rasgos densos de entrada.
    pub dense: usize,
    /// Tamaños de las capas ocultas (al menos una).
    pub hidden: Vec<usize>,
    /// Número de logits de salida.
    pub outputs: usize,
    /// Si hay cabeza de valor escalar.
    pub value_head: bool,
    /// Activación de las capas ocultas.
    pub activation: Activation,
}

/// Desplazamientos de cada bloque dentro del vector de parámetros.
#[derive(Debug, Clone)]
struct Layout {
    emb: usize,
    dense_w: usize,
    b0: usize,
    /// `(w, b)` de las capas ocultas 1.. .
    layers: Vec<(usize, usize)>,
    head_w: usize,
    head_b: usize,
    value_w: usize,
    value_b: usize,
    total: usize,
}

impl Layout {
    fn new(s: &NetSpec) -> Self {
        let h0 = s.hidden[0];
        let mut off = 0;
        let mut take = |n: usize| {
            let o = off;
            off += n;
            o
        };
        let emb = take(s.positions * s.categories * h0);
        let dense_w = take(s.dense * h0);
        let b0 = take(h0);
        let layers = s
            .hidden
            .windows(2)
            .map(|w| (take(w[1] * w[0]), take(w[1])))
            .collect();
        let last = *s.hidden.last().expect("al menos una capa oculta");
        let head_w = take(s.outputs * last);
        let head_b = take(s.outputs);
        let (value_w, value_b) = if s.value_head {
            (take(last), take(1))
        } else {
            (0, 0)
        };
        Self {
            emb,
            dense_w,
            b0,
            layers,
            head_w,
            head_b,
            value_w,
            value_b,
            total: off,
        }
    }
}

/// Entrada de una muestra.
#[derive(Debug, Clone, Copy)]
pub struct Input<'a, T> {
    /// Categoría por posición (`NONE` = ausente).
    pub cats: &'a [u16],
    /// Rasgos densos.
    pub dense: &'a [T],
}

/// Activaciones guardadas de una muestra, necesarias para retropropagar.
/// Se reutiliza entre muestras para no asignar memoria.
#[derive(Debug, Clone, Default)]
pub struct Tape<T> {
    cats: Vec<u16>,
    dense: Vec<T>,
    acts: Vec<Vec<T>>,
    dh: Vec<T>,
    dprev: Vec<T>,
    /// Logits de la última pasada hacia delante.
    pub logits: Vec<T>,
    /// Valor estimado de la última pasada (0 si no hay cabeza de valor).
    pub value: T,
}

/// Red neuronal.
#[derive(Debug, Clone)]
pub struct Net<T> {
    spec: NetSpec,
    layout: Layout,
    /// Parámetros aplanados.
    pub params: Vec<T>,
}

#[inline]
fn dot<T: Float>(a: &[T], b: &[T]) -> T {
    a.iter().zip(b).fold(T::ZERO, |acc, (&x, &y)| acc + x * y)
}

#[inline]
fn axpy<T: Float>(alpha: T, x: &[T], y: &mut [T]) {
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi += alpha * xi;
    }
}

impl<T: Float> Net<T> {
    /// Crea la red con inicialización aleatoria: varianza `1/n_entrada` (LeCun) para
    /// `tanh` y `2/n_entrada` (He) para ReLU; las cabezas se escalan ×0,01 para que
    /// la pérdida inicial sea ≈ `ln K`.
    ///
    /// # Panics
    ///
    /// Si `spec.hidden` está vacío o `spec.outputs == 0`.
    pub fn new(spec: NetSpec, rng: &mut Rng) -> Self {
        assert!(
            !spec.hidden.is_empty() && spec.outputs > 0,
            "arquitectura vacía"
        );
        let layout = Layout::new(&spec);
        let mut params = vec![T::ZERO; layout.total];
        let gain = match spec.activation {
            Activation::Tanh => 1.0,
            Activation::Relu => 2.0,
        };
        let mut fill = |range: std::ops::Range<usize>, std: f64| {
            for p in &mut params[range] {
                *p = T::from_f64(rng.normal() * std);
            }
        };
        let h0 = spec.hidden[0];
        let fan0 = (spec.positions + spec.dense).max(1) as f64;
        fill(layout.emb..layout.dense_w, (gain / fan0).sqrt());
        fill(layout.dense_w..layout.b0, (gain / fan0).sqrt());
        for (i, &(w, b)) in layout.layers.iter().enumerate() {
            fill(w..b, (gain / spec.hidden[i] as f64).sqrt());
        }
        let last = *spec.hidden.last().expect("comprobado arriba") as f64;
        fill(layout.head_w..layout.head_b, 0.01 / last.sqrt());
        if spec.value_head {
            fill(layout.value_w..layout.value_b, 0.01 / last.sqrt());
        }
        debug_assert!(h0 > 0);
        Self {
            spec,
            layout,
            params,
        }
    }

    /// Arquitectura.
    pub fn spec(&self) -> &NetSpec {
        &self.spec
    }

    /// Número total de parámetros.
    pub fn num_params(&self) -> usize {
        self.params.len()
    }

    /// Crea una cinta con la capacidad adecuada.
    pub fn tape(&self) -> Tape<T> {
        Tape {
            acts: self.spec.hidden.iter().map(|&h| vec![T::ZERO; h]).collect(),
            logits: vec![T::ZERO; self.spec.outputs],
            ..Tape::default()
        }
    }

    #[inline]
    fn activate(&self, v: T) -> T {
        match self.spec.activation {
            Activation::Tanh => v.tanh(),
            Activation::Relu => v.max(T::ZERO),
        }
    }

    #[inline]
    fn activation_grad(&self, out: T) -> T {
        match self.spec.activation {
            Activation::Tanh => T::ONE - out * out,
            Activation::Relu => {
                if out > T::ZERO {
                    T::ONE
                } else {
                    T::ZERO
                }
            }
        }
    }

    /// Pasada hacia delante; deja logits, valor y activaciones en `tape`.
    pub fn forward(&self, x: Input<'_, T>, tape: &mut Tape<T>) {
        let s = &self.spec;
        let l = &self.layout;
        debug_assert_eq!(x.cats.len(), s.positions);
        debug_assert_eq!(x.dense.len(), s.dense);
        let h0 = s.hidden[0];
        if tape.acts.len() != s.hidden.len() {
            *tape = self.tape();
        }
        tape.cats.clear();
        tape.cats.extend_from_slice(x.cats);
        tape.dense.clear();
        tape.dense.extend_from_slice(x.dense);

        let a0 = &mut tape.acts[0];
        a0.copy_from_slice(&self.params[l.b0..l.b0 + h0]);
        for (p, &c) in x.cats.iter().enumerate() {
            if c != NONE {
                let o = l.emb + (p * s.categories + usize::from(c)) * h0;
                axpy(T::ONE, &self.params[o..o + h0], a0);
            }
        }
        for (f, &v) in x.dense.iter().enumerate() {
            // Los rasgos one-hot son casi todos cero: se saltan sin coste.
            if v != T::ZERO {
                let o = l.dense_w + f * h0;
                axpy(v, &self.params[o..o + h0], a0);
            }
        }
        for v in a0.iter_mut() {
            *v = self.activate(*v);
        }
        for (i, &(w, b)) in l.layers.iter().enumerate() {
            let (prev, next) = tape.acts.split_at_mut(i + 1);
            let (input, out) = (&prev[i], &mut next[0]);
            let n_in = input.len();
            for (j, o) in out.iter_mut().enumerate() {
                let row = &self.params[w + j * n_in..w + (j + 1) * n_in];
                *o = self.activate(self.params[b + j] + dot(row, input));
            }
        }
        let last = &tape.acts[s.hidden.len() - 1];
        let n = last.len();
        for (j, z) in tape.logits.iter_mut().enumerate() {
            let row = &self.params[l.head_w + j * n..l.head_w + (j + 1) * n];
            *z = self.params[l.head_b + j] + dot(row, last);
        }
        tape.value = if s.value_head {
            self.params[l.value_b] + dot(&self.params[l.value_w..l.value_w + n], last)
        } else {
            T::ZERO
        };
    }

    /// Retropropaga `∂L/∂logits` y `∂L/∂valor` de la última pasada de `tape`,
    /// **acumulando** en `grads` (misma disposición que `params`).
    pub fn backward(&self, tape: &mut Tape<T>, dlogits: &[T], dvalue: T, grads: &mut [T]) {
        let s = &self.spec;
        let l = &self.layout;
        debug_assert_eq!(grads.len(), self.params.len());
        let last_i = s.hidden.len() - 1;
        let n = s.hidden[last_i];
        tape.dh.clear();
        tape.dh.resize(n, T::ZERO);
        {
            let last = &tape.acts[last_i];
            for (j, &dz) in dlogits.iter().enumerate() {
                let o = l.head_w + j * n;
                axpy(dz, last, &mut grads[o..o + n]);
                grads[l.head_b + j] += dz;
                axpy(dz, &self.params[o..o + n], &mut tape.dh);
            }
            if s.value_head {
                axpy(dvalue, last, &mut grads[l.value_w..l.value_w + n]);
                grads[l.value_b] += dvalue;
                axpy(dvalue, &self.params[l.value_w..l.value_w + n], &mut tape.dh);
            }
        }
        for i in (1..s.hidden.len()).rev() {
            let (w, b) = l.layers[i - 1];
            let out = &tape.acts[i];
            let input = &tape.acts[i - 1];
            let n_in = input.len();
            tape.dprev.clear();
            tape.dprev.resize(n_in, T::ZERO);
            for (j, (&dh, &o)) in tape.dh.iter().zip(out).enumerate() {
                let da = dh * self.activation_grad(o);
                let row = w + j * n_in;
                axpy(da, input, &mut grads[row..row + n_in]);
                grads[b + j] += da;
                axpy(da, &self.params[row..row + n_in], &mut tape.dprev);
            }
            std::mem::swap(&mut tape.dh, &mut tape.dprev);
        }
        let h0 = s.hidden[0];
        for (d, &o) in tape.dh.iter_mut().zip(&tape.acts[0]) {
            *d *= self.activation_grad(o);
        }
        axpy(T::ONE, &tape.dh, &mut grads[l.b0..l.b0 + h0]);
        for (p, &c) in tape.cats.iter().enumerate() {
            if c != NONE {
                let o = l.emb + (p * s.categories + usize::from(c)) * h0;
                axpy(T::ONE, &tape.dh, &mut grads[o..o + h0]);
            }
        }
        for (f, &v) in tape.dense.iter().enumerate() {
            let o = l.dense_w + f * h0;
            axpy(v, &tape.dh, &mut grads[o..o + h0]);
        }
    }

    /// Serializa la red (cabecera de enteros + parámetros `f32` little-endian).
    pub fn to_bytes(&self) -> Vec<u8> {
        let s = &self.spec;
        let mut out = b"NNG1".to_vec();
        let mut put = |v: usize| out.extend_from_slice(&(v as u32).to_le_bytes());
        put(s.positions);
        put(s.categories);
        put(s.dense);
        put(s.outputs);
        put(usize::from(s.value_head));
        put(usize::from(s.activation == Activation::Relu));
        put(s.hidden.len());
        for &h in &s.hidden {
            put(h);
        }
        for p in &self.params {
            out.extend_from_slice(&(p.to_f64() as f32).to_le_bytes());
        }
        out
    }

    /// Deserializa una red escrita con [`Net::to_bytes`].
    ///
    /// # Errors
    ///
    /// Si los bytes no tienen el formato esperado.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let bad = |d: &str| Error::format("red serializada", d);
        let rest = bytes
            .strip_prefix(b"NNG1")
            .ok_or_else(|| bad("firma incorrecta"))?;
        let mut words = rest
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w) as usize);
        let mut next = || words.next().ok_or_else(|| bad("truncada"));
        let (positions, categories, dense, outputs) = (next()?, next()?, next()?, next()?);
        let value_head = next()? == 1;
        let activation = if next()? == 1 {
            Activation::Relu
        } else {
            Activation::Tanh
        };
        let n_hidden = next()?;
        if n_hidden == 0 || n_hidden > 16 {
            return Err(bad("número de capas inválido"));
        }
        let hidden = (0..n_hidden).map(|_| next()).collect::<Result<Vec<_>>>()?;
        let spec = NetSpec {
            positions,
            categories,
            dense,
            hidden,
            outputs,
            value_head,
            activation,
        };
        let layout = Layout::new(&spec);
        let header = 4 + 4 * (7 + n_hidden);
        let body = bytes.get(header..).ok_or_else(|| bad("truncada"))?;
        if body.len() != layout.total * 4 {
            return Err(bad("número de parámetros incorrecto"));
        }
        let params = body
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| T::from_f64(f64::from(f32::from_le_bytes(*w))))
            .collect();
        Ok(Self {
            spec,
            layout,
            params,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::loss::cross_entropy;
    use super::*;

    fn spec(activation: Activation, value_head: bool) -> NetSpec {
        NetSpec {
            positions: 5,
            categories: 4,
            dense: 3,
            hidden: vec![7, 6],
            outputs: 4,
            value_head,
            activation,
        }
    }

    /// Pérdida = CE(logits, objetivo) + 0,5·(valor − 0,3)².
    fn loss(
        net: &Net<f64>,
        x: Input<'_, f64>,
        tape: &mut Tape<f64>,
        grads: Option<&mut [f64]>,
    ) -> f64 {
        net.forward(x, tape);
        let mut dz = vec![0.0; 4];
        let ce = cross_entropy(&tape.logits, 2, 0.1, 1.0, &mut dz);
        let dv = if net.spec.value_head {
            tape.value - 0.3
        } else {
            0.0
        };
        if let Some(g) = grads {
            net.backward(tape, &dz, dv, g);
        }
        ce + 0.5 * dv * dv
    }

    #[test]
    fn chequeo_de_gradiente() {
        for act in [Activation::Tanh, Activation::Relu] {
            let mut rng = Rng::seed_from_u64(11);
            let mut net: Net<f64> = Net::new(spec(act, true), &mut rng);
            // Pesos de cabeza más grandes para que el gradiente no sea diminuto.
            for p in &mut net.params {
                *p *= 3.0;
            }
            let cats = [0_u16, 3, NONE, 1, 2];
            let dense = [0.5, -1.0, 2.0];
            let x = Input {
                cats: &cats,
                dense: &dense,
            };
            let mut tape = net.tape();
            let mut grads = vec![0.0; net.num_params()];
            loss(&net, x, &mut tape, Some(&mut grads));
            let h = 1e-5;
            let mut worst = 0.0_f64;
            for i in 0..net.num_params() {
                let orig = net.params[i];
                net.params[i] = orig + h;
                let a = loss(&net, x, &mut tape, None);
                net.params[i] = orig - h;
                let b = loss(&net, x, &mut tape, None);
                net.params[i] = orig;
                let num = (a - b) / (2.0 * h);
                let rel = (num - grads[i]).abs() / (num.abs() + grads[i].abs()).max(1e-8);
                // ReLU no es derivable en 0; se ignoran diferencias en torno a ese punto.
                if (num - grads[i]).abs() > 1e-7 {
                    worst = worst.max(rel);
                }
            }
            assert!(worst < 1e-6, "{act:?}: error relativo {worst}");
        }
    }

    #[test]
    fn perdida_inicial_cerca_de_ln_k() {
        let mut rng = Rng::seed_from_u64(1);
        let net: Net<f32> = Net::new(spec(Activation::Tanh, false), &mut rng);
        let mut tape = net.tape();
        net.forward(
            Input {
                cats: &[1, 2, 3, 0, 1],
                dense: &[0.1, 0.2, 0.3],
            },
            &mut tape,
        );
        let mut dz = [0.0_f32; 4];
        let l = cross_entropy(&tape.logits, 0, 0.0, 1.0, &mut dz);
        assert!((l - 4.0_f32.ln()).abs() < 0.05, "{l}");
    }

    #[test]
    fn serializacion_ida_y_vuelta() {
        let mut rng = Rng::seed_from_u64(5);
        let net: Net<f32> = Net::new(spec(Activation::Relu, true), &mut rng);
        let back: Net<f32> = Net::from_bytes(&net.to_bytes()).expect("válida");
        assert_eq!(back.spec, net.spec);
        assert_eq!(back.params, net.params);
    }
}
