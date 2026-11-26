//! Estadística no paramétrica para comparar configuraciones del GA, siguiendo
//! a Arcuri y Briand (2011) y Derrac et al. (2011):
//!
//! - Mann-Whitney U (aproximación normal con corrección por empates y por
//!   continuidad) y tamaño de efecto Â12 de Vargha y Delaney (2000).
//! - Prueba exacta de Fisher (2×2, bilateral) y *odds ratio* para tasas de éxito.
//! - Kruskal-Wallis y Friedman (χ²) para más de dos grupos.
//! - Corrección de Holm para comparaciones múltiples.

/// Resumen descriptivo de una muestra.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Summary {
    /// Tamaño.
    pub n: usize,
    /// Media.
    pub mean: f64,
    /// Desviación típica muestral.
    pub std: f64,
    /// Mínimo.
    pub min: f64,
    /// Primer cuartil.
    pub q1: f64,
    /// Mediana.
    pub median: f64,
    /// Tercer cuartil.
    pub q3: f64,
    /// Máximo.
    pub max: f64,
}

/// Calcula el resumen (cuartiles tipo 7 de Hyndman-Fan).
pub fn summary(xs: &[f64]) -> Summary {
    let n = xs.len();
    if n == 0 {
        return Summary::default();
    }
    let mut v = xs.to_vec();
    v.sort_by(f64::total_cmp);
    let mean = v.iter().sum::<f64>() / n as f64;
    let var = if n > 1 {
        v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    Summary {
        n,
        mean,
        std: var.sqrt(),
        min: v[0],
        q1: crate::plot::quantile(&v, 0.25),
        median: crate::plot::quantile(&v, 0.5),
        q3: crate::plot::quantile(&v, 0.75),
        max: v[n - 1],
    }
}

/// `ln Γ(x)` por la aproximación de Lanczos (g = 7, 9 coeficientes).
pub fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflexión: Γ(x)Γ(1−x) = π / sen(πx).
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + G + 0.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Gamma incompleta regularizada superior `Q(a, x)` (serie o fracción continua).
pub fn gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let gln = ln_gamma(a);
    if x < a + 1.0 {
        // Serie para P(a, x).
        let (mut sum, mut del, mut ap) = (1.0 / a, 1.0 / a, a);
        for _ in 0..1000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        1.0 - sum * (-x + a * x.ln() - gln).exp()
    } else {
        // Fracción continua de Lentz para Q(a, x).
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..1000 {
            let an = -f64::from(i) * (f64::from(i) - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-15 {
                break;
            }
        }
        (-x + a * x.ln() - gln).exp() * h
    }
}

/// Probabilidad de cola superior de una χ² con `k` grados de libertad.
pub fn chi2_sf(x: f64, k: f64) -> f64 {
    gamma_q(k / 2.0, x / 2.0)
}

/// Función de distribución normal estándar complementaria, `P(Z > z)`.
pub fn normal_sf(z: f64) -> f64 {
    // Φ̄(z) = Q(1/2, z²/2) / 2 para z ≥ 0.
    if z >= 0.0 {
        0.5 * gamma_q(0.5, z * z / 2.0)
    } else {
        1.0 - normal_sf(-z)
    }
}

/// Rangos promedio (empates comparten la media de sus rangos), base 1.
/// Devuelve los rangos y `Σ(t³ − t)` de los grupos empatados.
#[expect(clippy::float_cmp, reason = "un empate es una igualdad exacta")]
pub fn ranks(xs: &[f64]) -> (Vec<f64>, f64) {
    let mut idx: Vec<usize> = (0..xs.len()).collect();
    idx.sort_by(|&a, &b| xs[a].total_cmp(&xs[b]));
    let mut r = vec![0.0; xs.len()];
    let mut ties = 0.0;
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && xs[idx[j + 1]] == xs[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            r[k] = avg;
        }
        let t = (j - i + 1) as f64;
        ties += t * t * t - t;
        i = j + 1;
    }
    (r, ties)
}

/// Resultado de Mann-Whitney.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MannWhitney {
    /// Estadístico U de la primera muestra.
    pub u: f64,
    /// Estadístico z (con corrección por continuidad).
    pub z: f64,
    /// Valor p bilateral.
    pub p: f64,
}

/// Prueba U de Mann-Whitney bilateral (aproximación normal, corrección por
/// empates y por continuidad, como `scipy.stats.mannwhitneyu(method="asymptotic")`).
pub fn mann_whitney(a: &[f64], b: &[f64]) -> MannWhitney {
    let (n1, n2) = (a.len() as f64, b.len() as f64);
    let all: Vec<f64> = a.iter().chain(b).copied().collect();
    let (r, ties) = ranks(&all);
    let r1: f64 = r[..a.len()].iter().sum();
    let u = r1 - n1 * (n1 + 1.0) / 2.0;
    let n = n1 + n2;
    let mu = n1 * n2 / 2.0;
    let sigma = (n1 * n2 / 12.0 * ((n + 1.0) - ties / (n * (n - 1.0)))).sqrt();
    if sigma == 0.0 {
        return MannWhitney { u, z: 0.0, p: 1.0 };
    }
    let z = ((u - mu).abs() - 0.5).max(0.0) / sigma * (u - mu).signum();
    MannWhitney {
        u,
        z,
        p: (2.0 * normal_sf(z.abs())).min(1.0),
    }
}

/// Â12 de Vargha y Delaney: probabilidad de que un valor de `a` supere a uno
/// de `b` (los empates cuentan 0,5). 0,5 = sin efecto.
#[expect(clippy::float_cmp, reason = "un empate es una igualdad exacta")]
pub fn a12(a: &[f64], b: &[f64]) -> f64 {
    let mut wins = 0.0;
    for &x in a {
        for &y in b {
            wins += if x > y {
                1.0
            } else if x == y {
                0.5
            } else {
                0.0
            };
        }
    }
    wins / (a.len() * b.len()).max(1) as f64
}

/// Magnitud cualitativa de Â12 (umbrales 0,56 / 0,64 / 0,71, simétricos).
pub fn a12_magnitude(a: f64) -> &'static str {
    let d = (a - 0.5).abs();
    if d < 0.06 {
        "despreciable"
    } else if d < 0.14 {
        "pequeño"
    } else if d < 0.21 {
        "mediano"
    } else {
        "grande"
    }
}

fn ln_choose(n: u64, k: u64) -> f64 {
    ln_gamma(n as f64 + 1.0) - ln_gamma(k as f64 + 1.0) - ln_gamma((n - k) as f64 + 1.0)
}

/// Prueba exacta de Fisher bilateral para la tabla `[[a, b], [c, d]]`
/// (suma de probabilidades de las tablas no más probables que la observada).
/// Devuelve `(p, odds_ratio)`; el *odds ratio* usa la corrección de Haldane
/// (+0,5) si alguna celda es 0.
pub fn fisher_exact(a: u64, b: u64, c: u64, d: u64) -> (f64, f64) {
    let (r1, c1, n) = (a + b, a + c, a + b + c + d);
    let p_of = |x: u64| (ln_choose(r1, x) + ln_choose(n - r1, c1 - x) - ln_choose(n, c1)).exp();
    let p_obs = p_of(a);
    let lo = c1.saturating_sub(n - r1);
    let hi = r1.min(c1);
    let p: f64 = (lo..=hi)
        .map(p_of)
        .filter(|&p| p <= p_obs * (1.0 + 1e-7))
        .sum();
    let (fa, fb, fc, fd) = if a == 0 || b == 0 || c == 0 || d == 0 {
        (
            a as f64 + 0.5,
            b as f64 + 0.5,
            c as f64 + 0.5,
            d as f64 + 0.5,
        )
    } else {
        (a as f64, b as f64, c as f64, d as f64)
    };
    (p.min(1.0), (fa * fd) / (fb * fc))
}

/// Kruskal-Wallis con corrección por empates. Devuelve `(H, p)`.
pub fn kruskal_wallis(groups: &[Vec<f64>]) -> (f64, f64) {
    let all: Vec<f64> = groups.iter().flatten().copied().collect();
    let n = all.len() as f64;
    let (r, ties) = ranks(&all);
    let mut off = 0;
    let mut h = 0.0;
    for g in groups {
        let rs: f64 = r[off..off + g.len()].iter().sum();
        h += rs * rs / g.len() as f64;
        off += g.len();
    }
    h = 12.0 / (n * (n + 1.0)) * h - 3.0 * (n + 1.0);
    let corr = 1.0 - ties / (n * n * n - n);
    if corr > 0.0 {
        h /= corr;
    }
    (h, chi2_sf(h, groups.len() as f64 - 1.0))
}

/// Prueba de Friedman: `data[bloque][tratamiento]` (p. ej. instancias ×
/// algoritmos). Devuelve `(χ², p, rangos medios por tratamiento)`.
pub fn friedman(data: &[Vec<f64>]) -> (f64, f64, Vec<f64>) {
    let n = data.len() as f64;
    let k = data.first().map_or(0, Vec::len);
    let mut rank_sum = vec![0.0; k];
    for row in data {
        let (r, _) = ranks(row);
        for (s, v) in rank_sum.iter_mut().zip(r) {
            *s += v;
        }
    }
    let kf = k as f64;
    let chi2 = 12.0 / (n * kf * (kf + 1.0)) * rank_sum.iter().map(|r| r * r).sum::<f64>()
        - 3.0 * n * (kf + 1.0);
    let mean_ranks = rank_sum.iter().map(|r| r / n).collect();
    (chi2, chi2_sf(chi2, kf - 1.0), mean_ranks)
}

/// Valores p ajustados por el método de Holm (paso descendente).
pub fn holm(p: &[f64]) -> Vec<f64> {
    let m = p.len();
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    let mut adj = vec![0.0; m];
    let mut running: f64 = 0.0;
    for (rank, &i) in idx.iter().enumerate() {
        running = running.max(((m - rank) as f64 * p[i]).min(1.0));
        adj[i] = running;
    }
    adj
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1e-12)
    }

    #[test]
    fn gamma_y_normal() {
        assert!(close(ln_gamma(5.0), 24.0_f64.ln(), 1e-12));
        assert!(close(
            ln_gamma(0.5),
            std::f64::consts::PI.sqrt().ln(),
            1e-12
        ));
        assert!(close(normal_sf(1.959_963_984_540_054), 0.025, 1e-9));
        assert!(close(chi2_sf(3.841_458_820_694_124, 1.0), 0.05, 1e-9));
    }

    // Valores de referencia obtenidos una vez con SciPy 1.18.1:
    // mannwhitneyu(a, b, method="asymptotic"), fisher_exact, kruskal, friedmanchisquare.
    #[test]
    fn contra_scipy() {
        let a = [1.1, 2.3, 2.3, 4.0, 5.5, 6.1, 7.2, 8.8];
        let b = [0.5, 1.1, 1.9, 2.3, 3.0, 3.3, 4.1];
        let mw = mann_whitney(&a, &b);
        assert!(close(mw.u, 42.5, 1e-12), "{mw:?}");
        assert!(close(mw.p, 0.103_637_964_013_962_3, 1e-6), "{mw:?}");
        let (p, or) = fisher_exact(8, 2, 1, 5);
        assert!(close(p, 0.034_965_034_965_035_01, 1e-9), "{p}");
        assert!(close(or, 20.0, 1e-12));
        let (h, p) = kruskal_wallis(&[a.to_vec(), b.to_vec(), vec![9.0, 9.5, 10.0, 2.3]]);
        assert!(close(h, 6.277_821_713_273_434, 1e-9), "{h}");
        assert!(close(p, 0.043_329_964_755_115_09, 1e-6), "{p}");
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 1.0],
            vec![1.0, 3.0, 2.0],
            vec![1.0, 2.0, 3.0],
        ];
        let (chi2, p, _) = friedman(&data);
        assert!(close(chi2, 3.5, 1e-12), "{chi2}");
        assert!(close(p, 0.173_773_943_450_445_04, 1e-9), "{p}");
    }

    #[test]
    fn a12_y_holm() {
        assert!(close(a12(&[3.0, 4.0], &[1.0, 2.0]), 1.0, 1e-12));
        assert!(close(a12(&[1.0], &[1.0]), 0.5, 1e-12));
        let adj = holm(&[0.01, 0.04, 0.03]);
        assert_eq!(adj, vec![0.03, 0.06, 0.06]);
    }
}
