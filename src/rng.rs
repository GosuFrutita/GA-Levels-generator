//! Generadores pseudoaleatorios hechos a mano.
//!
//! - [`SplitMix64`]: generador de 64 bits de estado mínimo, usado para sembrar.
//! - [`Rng`]: xoshiro256** (Blackman y Vigna), generador de propósito general.
//!
//! Referencias: <https://prng.di.unimi.it/xoshiro256starstar.c> y
//! <https://prng.di.unimi.it/splitmix64.c>. Para enteros acotados sin sesgo se
//! sigue a Lemire (2019), *Fast Random Integer Generation in an Interval*
//! (<https://arxiv.org/abs/1805.10941>).

/// `SplitMix64`: expande una semilla de 64 bits en una secuencia bien mezclada.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Crea el generador con la semilla dada.
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Devuelve el siguiente valor de 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// Mezcla varias palabras en una semilla, para derivar flujos independientes
/// a partir de `(semilla, generación, índice)` sin depender del hilo.
pub fn mix_seed(parts: &[u64]) -> u64 {
    let mut acc = 0x6a09_e667_f3bc_c909_u64;
    for &p in parts {
        acc = SplitMix64::new(acc ^ p).next_u64();
    }
    acc
}

/// Generador xoshiro256**.
#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
    /// Segunda normal pendiente del método polar.
    spare_normal: Option<f64>,
}

impl Rng {
    /// Siembra el estado con cuatro salidas de [`SplitMix64`], como recomiendan
    /// los autores (el estado nunca queda en cero).
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut sm = SplitMix64::new(seed);
        let s = [sm.next_u64(), sm.next_u64(), sm.next_u64(), sm.next_u64()];
        Self {
            s,
            spare_normal: None,
        }
    }

    /// Devuelve el siguiente valor de 64 bits.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Avanza el estado 2^128 pasos: produce subsecuencias que no se solapan.
    pub fn jump(&mut self) {
        const JUMP: [u64; 4] = [
            0x180e_c6d3_3cfd_0aba,
            0xd5a6_1266_f0c9_392c,
            0xa958_2618_e03f_c9aa,
            0x39ab_dc45_29b1_661c,
        ];
        let mut acc = [0_u64; 4];
        for word in JUMP {
            for b in 0..64 {
                if word & (1_u64 << b) != 0 {
                    for (a, s) in acc.iter_mut().zip(self.s) {
                        *a ^= s;
                    }
                }
                self.next_u64();
            }
        }
        self.s = acc;
        self.spare_normal = None;
    }

    /// Uniforme en `[0, 1)` con 53 bits de precisión.
    #[inline]
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
    }

    /// Devuelve `true` con probabilidad `p`.
    #[inline]
    pub fn chance(&mut self, p: f64) -> bool {
        self.f64() < p
    }

    /// Entero uniforme en `[0, n)` sin sesgo (método de Lemire).
    ///
    /// # Panics
    ///
    /// Si `n == 0`.
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "below(0) no tiene valores posibles");
        let mut m = u128::from(self.next_u64()) * u128::from(n);
        let mut low = m as u64;
        if low < n {
            let threshold = n.wrapping_neg() % n;
            while low < threshold {
                m = u128::from(self.next_u64()) * u128::from(n);
                low = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// Índice uniforme en `[0, n)`.
    ///
    /// # Panics
    ///
    /// Si `n == 0`.
    #[inline]
    pub fn index(&mut self, n: usize) -> usize {
        self.below(n as u64) as usize
    }

    /// Entero uniforme en el rango cerrado `[lo, hi]`.
    ///
    /// # Panics
    ///
    /// Si `lo > hi`.
    #[inline]
    pub fn range_i32(&mut self, lo: i32, hi: i32) -> i32 {
        assert!(lo <= hi, "rango vacío");
        let span = (i64::from(hi) - i64::from(lo) + 1) as u64;
        (i64::from(lo) + self.below(span).cast_signed()) as i32
    }

    /// Normal estándar por el método polar de Marsaglia.
    pub fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare_normal.take() {
            return z;
        }
        loop {
            let u = 2.0 * self.f64() - 1.0;
            let v = 2.0 * self.f64() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let k = (-2.0 * s.ln() / s).sqrt();
                self.spare_normal = Some(v * k);
                return u * k;
            }
        }
    }

    /// Mezcla un slice en sitio (Fisher-Yates).
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.index(i + 1);
            xs.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_vector_conocido() {
        // Valor de referencia calculado con el código C de Vigna.
        assert_eq!(SplitMix64::new(0).next_u64(), 0xe220_a839_7b1d_cdaf);
    }

    #[test]
    fn xoshiro_vector_conocido() {
        let mut r = Rng::seed_from_u64(0);
        assert_eq!(r.next_u64(), 0x99ec_5f36_cb75_f2b4);
    }

    #[test]
    fn below_es_uniforme() {
        let mut r = Rng::seed_from_u64(42);
        let mut counts = [0_u32; 6];
        let n = 600_000;
        for _ in 0..n {
            counts[r.index(6)] += 1;
        }
        let expected = f64::from(n) / 6.0;
        let chi2: f64 = counts
            .iter()
            .map(|&c| (f64::from(c) - expected).powi(2) / expected)
            .sum();
        // Valor crítico de chi² con 5 g.l. al 0.1 %: 20.52.
        assert!(chi2 < 20.52, "chi2 = {chi2}");
    }

    #[test]
    fn f64_en_rango() {
        let mut r = Rng::seed_from_u64(7);
        for _ in 0..10_000 {
            let x = r.f64();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn normal_media_y_varianza() {
        let mut r = Rng::seed_from_u64(3);
        let n = 200_000;
        let xs: Vec<f64> = (0..n).map(|_| r.normal()).collect();
        let mean = xs.iter().sum::<f64>() / f64::from(n);
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / f64::from(n);
        assert!(mean.abs() < 0.01, "media {mean}");
        assert!((var - 1.0).abs() < 0.02, "varianza {var}");
    }

    #[test]
    fn jump_cambia_el_flujo() {
        let mut a = Rng::seed_from_u64(1);
        let mut b = a.clone();
        b.jump();
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn shuffle_es_permutacion() {
        let mut r = Rng::seed_from_u64(9);
        let mut v: Vec<u32> = (0..100).collect();
        r.shuffle(&mut v);
        let mut s = v.clone();
        s.sort_unstable();
        assert_eq!(s, (0..100).collect::<Vec<_>>());
        assert_ne!(v, s);
    }
}
