//! Gráficas SVG escritas a mano: curvas con banda, diagramas de caja y mapas de calor.
//!
//! Se genera texto SVG 1.1 directamente; cualquier navegador o visor lo abre.

use std::fmt::Write as _;

/// Paleta categórica (alto contraste, apta para daltonismo; Okabe-Ito).
pub const PALETTE: [&str; 8] = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000",
];

const W: f64 = 760.0;
const H: f64 = 420.0;
const ML: f64 = 70.0;
const MR: f64 = 170.0;
const MT: f64 = 40.0;
const MB: f64 = 55.0;

/// Una serie de una gráfica de líneas, con banda opcional (p. ej. IQR).
#[derive(Debug, Clone)]
pub struct Series {
    /// Nombre en la leyenda.
    pub name: String,
    /// Puntos `(x, y)`.
    pub points: Vec<(f64, f64)>,
    /// Banda opcional `(x, y_bajo, y_alto)`.
    pub band: Vec<(f64, f64, f64)>,
}

/// Escapa texto para XML.
pub fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Marcas "bonitas" (1-2-5 × 10^k) que cubren `[lo, hi]`.
pub fn nice_ticks(lo: f64, hi: f64, target: usize) -> Vec<f64> {
    if !lo.is_finite() || !hi.is_finite() || hi <= lo {
        return vec![lo];
    }
    let raw = (hi - lo) / target.max(1) as f64;
    let mag = 10_f64.powf(raw.log10().floor());
    let step = [1.0, 2.0, 5.0, 10.0]
        .into_iter()
        .map(|m| m * mag)
        .find(|s| *s >= raw)
        .unwrap_or(10.0 * mag);
    // Se multiplica en vez de acumular para no arrastrar error de redondeo.
    let first = (lo / step).floor() as i64;
    let last = (hi / step).ceil() as i64;
    (first..=last)
        .map(|k| k as f64 * step)
        .filter(|t| *t >= lo - step * 1e-9 && *t <= hi + step * 1e-9)
        .collect()
}

fn fmt_tick(v: f64) -> String {
    if v == 0.0 {
        "0".into()
    } else if v.abs() >= 1000.0 || v.fract() == 0.0 {
        format!("{v:.0}")
    } else if v.abs() >= 1.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}

/// Transformación lineal de datos a píxeles.
struct Axes {
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
}

impl Axes {
    fn px(&self, x: f64) -> f64 {
        ML + (x - self.x0) / (self.x1 - self.x0) * (W - ML - MR)
    }
    fn py(&self, y: f64) -> f64 {
        H - MB - (y - self.y0) / (self.y1 - self.y0) * (H - MT - MB)
    }
}

fn header(svg: &mut String, title: &str) {
    let _ = write!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="sans-serif" font-size="12">"#
    );
    let _ = write!(
        svg,
        r##"<rect width="100%" height="100%" fill="#ffffff"/>"##
    );
    let _ = write!(
        svg,
        r#"<text x="{}" y="24" font-size="15" font-weight="bold">{}</text>"#,
        ML,
        escape(title)
    );
}

fn grid_and_axes(svg: &mut String, ax: &Axes, xlabel: &str, ylabel: &str, xticks: bool) {
    for t in nice_ticks(ax.y0, ax.y1, 6) {
        let y = ax.py(t);
        let _ = write!(
            svg,
            r##"<line x1="{ML}" x2="{}" y1="{y:.1}" y2="{y:.1}" stroke="#e5e5e5"/><text x="{}" y="{:.1}" text-anchor="end">{}</text>"##,
            W - MR,
            ML - 6.0,
            y + 4.0,
            fmt_tick(t)
        );
    }
    if xticks {
        for t in nice_ticks(ax.x0, ax.x1, 8) {
            let x = ax.px(t);
            let _ = write!(
                svg,
                r##"<line x1="{x:.1}" x2="{x:.1}" y1="{}" y2="{}" stroke="#999"/><text x="{x:.1}" y="{}" text-anchor="middle">{}</text>"##,
                H - MB,
                H - MB + 5.0,
                H - MB + 18.0,
                fmt_tick(t)
            );
        }
    }
    let _ = write!(
        svg,
        r##"<line x1="{ML}" x2="{ML}" y1="{MT}" y2="{}" stroke="#333"/><line x1="{ML}" x2="{}" y1="{}" y2="{}" stroke="#333"/>"##,
        H - MB,
        W - MR,
        H - MB,
        H - MB
    );
    let _ = write!(
        svg,
        r#"<text x="{}" y="{}" text-anchor="middle">{}</text>"#,
        (ML + W - MR) / 2.0,
        H - 12.0,
        escape(xlabel)
    );
    let _ = write!(
        svg,
        r#"<text transform="translate(18 {}) rotate(-90)" text-anchor="middle">{}</text>"#,
        (MT + H - MB) / 2.0,
        escape(ylabel)
    );
}

fn legend(svg: &mut String, names: &[&str]) {
    for (i, name) in names.iter().enumerate() {
        let y = MT + 10.0 + i as f64 * 20.0;
        let c = PALETTE[i % PALETTE.len()];
        let _ = write!(
            svg,
            r#"<rect x="{}" y="{}" width="14" height="4" fill="{c}"/><text x="{}" y="{}">{}</text>"#,
            W - MR + 14.0,
            y - 4.0,
            W - MR + 34.0,
            y + 1.0,
            escape(name)
        );
    }
}

fn extent(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let (lo, hi) = values
        .filter(|v| v.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        });
    if lo > hi {
        (0.0, 1.0)
    } else if (hi - lo).abs() < 1e-12 {
        (lo - 0.5, hi + 0.5)
    } else {
        (lo, hi)
    }
}

/// Gráfica de líneas con bandas opcionales.
pub fn line_chart(title: &str, xlabel: &str, ylabel: &str, series: &[Series]) -> String {
    let (x0, x1) = extent(series.iter().flat_map(|s| s.points.iter().map(|p| p.0)));
    let (y0, y1) = extent(series.iter().flat_map(|s| {
        s.points
            .iter()
            .map(|p| p.1)
            .chain(s.band.iter().flat_map(|b| [b.1, b.2]))
    }));
    let pad = (y1 - y0) * 0.05;
    let ax = Axes {
        x0,
        x1,
        y0: y0 - pad,
        y1: y1 + pad,
    };
    let mut svg = String::with_capacity(16_384);
    header(&mut svg, title);
    grid_and_axes(&mut svg, &ax, xlabel, ylabel, true);
    for (i, s) in series.iter().enumerate() {
        let c = PALETTE[i % PALETTE.len()];
        if !s.band.is_empty() {
            let mut d = String::new();
            for (k, &(x, _, hi)) in s.band.iter().enumerate() {
                let _ = write!(
                    d,
                    "{}{:.1},{:.1} ",
                    if k == 0 { 'M' } else { 'L' },
                    ax.px(x),
                    ax.py(hi)
                );
            }
            for &(x, lo, _) in s.band.iter().rev() {
                let _ = write!(d, "L{:.1},{:.1} ", ax.px(x), ax.py(lo));
            }
            let _ = write!(
                svg,
                r#"<path d="{d}Z" fill="{c}" fill-opacity="0.18" stroke="none"/>"#
            );
        }
        let mut d = String::new();
        for (k, &(x, y)) in s.points.iter().enumerate() {
            let _ = write!(
                d,
                "{}{:.1},{:.1} ",
                if k == 0 { 'M' } else { 'L' },
                ax.px(x),
                ax.py(y)
            );
        }
        let _ = write!(
            svg,
            r#"<path d="{d}" fill="none" stroke="{c}" stroke-width="2"/>"#
        );
    }
    let names: Vec<&str> = series.iter().map(|s| s.name.as_str()).collect();
    legend(&mut svg, &names);
    svg.push_str("</svg>");
    svg
}

/// Cuartiles por interpolación lineal (tipo 7 de Hyndman-Fan, el de R por defecto).
pub fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let h = (sorted.len() - 1) as f64 * q.clamp(0.0, 1.0);
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

/// Diagrama de cajas (bigotes a 1,5·IQR) de varios grupos.
pub fn box_plot(title: &str, ylabel: &str, groups: &[(String, Vec<f64>)]) -> String {
    let (y0, y1) = extent(groups.iter().flat_map(|g| g.1.iter().copied()));
    let pad = (y1 - y0) * 0.05;
    let n = groups.len().max(1) as f64;
    let ax = Axes {
        x0: 0.0,
        x1: n,
        y0: y0 - pad,
        y1: y1 + pad,
    };
    let mut svg = String::with_capacity(8192);
    header(&mut svg, title);
    grid_and_axes(&mut svg, &ax, "", ylabel, false);
    let bw = (W - ML - MR) / n * 0.5;
    for (i, (name, values)) in groups.iter().enumerate() {
        let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
        if v.is_empty() {
            continue;
        }
        v.sort_by(f64::total_cmp);
        let (q1, med, q3) = (quantile(&v, 0.25), quantile(&v, 0.5), quantile(&v, 0.75));
        let iqr = q3 - q1;
        let lo = v
            .iter()
            .copied()
            .find(|&x| x >= q1 - 1.5 * iqr)
            .unwrap_or(q1);
        let hi = v
            .iter()
            .rev()
            .copied()
            .find(|&x| x <= q3 + 1.5 * iqr)
            .unwrap_or(q3);
        let cx = ax.px(i as f64 + 0.5);
        let c = PALETTE[i % PALETTE.len()];
        let _ = write!(
            svg,
            r##"<line x1="{cx:.1}" x2="{cx:.1}" y1="{:.1}" y2="{:.1}" stroke="#333"/>"##,
            ax.py(lo),
            ax.py(hi)
        );
        let _ = write!(
            svg,
            r#"<rect x="{:.1}" y="{:.1}" width="{bw:.1}" height="{:.1}" fill="{c}" fill-opacity="0.35" stroke="{c}"/>"#,
            cx - bw / 2.0,
            ax.py(q3),
            (ax.py(q1) - ax.py(q3)).max(1.0)
        );
        let _ = write!(
            svg,
            r##"<line x1="{:.1}" x2="{:.1}" y1="{:.1}" y2="{:.1}" stroke="#000" stroke-width="2"/>"##,
            cx - bw / 2.0,
            cx + bw / 2.0,
            ax.py(med),
            ax.py(med)
        );
        for &x in v.iter().filter(|&&x| x < lo || x > hi) {
            let _ = write!(
                svg,
                r##"<circle cx="{cx:.1}" cy="{:.1}" r="2.5" fill="none" stroke="#333"/>"##,
                ax.py(x)
            );
        }
        let _ = write!(
            svg,
            r#"<text x="{cx:.1}" y="{}" text-anchor="middle">{}</text>"#,
            H - MB + 18.0,
            escape(name)
        );
    }
    svg.push_str("</svg>");
    svg
}

/// Interpola un color secuencial (viridis aproximado con 5 paradas).
fn viridis(t: f64) -> String {
    const STOPS: [(f64, f64, f64); 5] = [
        (68.0, 1.0, 84.0),
        (59.0, 82.0, 139.0),
        (33.0, 145.0, 140.0),
        (94.0, 201.0, 98.0),
        (253.0, 231.0, 37.0),
    ];
    let t = if t.is_finite() {
        t.clamp(0.0, 1.0)
    } else {
        0.0
    } * 4.0;
    let i = (t.floor() as usize).min(3);
    let f = t - i as f64;
    let (a, b) = (STOPS[i], STOPS[i + 1]);
    let mix = |u: f64, v: f64| (u + (v - u) * f).round() as u8;
    format!(
        "#{:02x}{:02x}{:02x}",
        mix(a.0, b.0),
        mix(a.1, b.1),
        mix(a.2, b.2)
    )
}

/// Mapa de calor de una matriz `values[fila][columna]` con etiquetas.
pub fn heatmap(
    title: &str,
    row_labels: &[String],
    col_labels: &[String],
    values: &[Vec<f64>],
    (xlabel, ylabel): (&str, &str),
) -> String {
    let (lo, hi) = extent(values.iter().flatten().copied());
    let rows = values.len().max(1) as f64;
    let cols = values.first().map_or(1, Vec::len).max(1) as f64;
    let (cw, ch) = ((W - ML - MR) / cols, (H - MT - MB) / rows);
    let mut svg = String::with_capacity(8192);
    header(&mut svg, title);
    for (r, row) in values.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            let t = (v - lo) / (hi - lo);
            let (x, y) = (ML + c as f64 * cw, MT + r as f64 * ch);
            let text = if t > 0.6 { "#000" } else { "#fff" };
            let _ = write!(
                svg,
                r#"<rect x="{x:.1}" y="{y:.1}" width="{cw:.1}" height="{ch:.1}" fill="{}"/><text x="{:.1}" y="{:.1}" text-anchor="middle" fill="{text}">{}</text>"#,
                viridis(t),
                x + cw / 2.0,
                y + ch / 2.0 + 4.0,
                if v.is_finite() {
                    format!("{v:.3}")
                } else {
                    "-".into()
                }
            );
        }
    }
    for (c, l) in col_labels.iter().enumerate() {
        let _ = write!(
            svg,
            r#"<text x="{:.1}" y="{}" text-anchor="middle">{}</text>"#,
            ML + (c as f64 + 0.5) * cw,
            H - MB + 18.0,
            escape(l)
        );
    }
    for (r, l) in row_labels.iter().enumerate() {
        let _ = write!(
            svg,
            r#"<text x="{}" y="{:.1}" text-anchor="end">{}</text>"#,
            ML - 6.0,
            MT + (r as f64 + 0.5) * ch + 4.0,
            escape(l)
        );
    }
    let _ = write!(
        svg,
        r#"<text x="{}" y="{}" text-anchor="middle">{}</text><text transform="translate(18 {}) rotate(-90)" text-anchor="middle">{}</text>"#,
        (ML + W - MR) / 2.0,
        H - 12.0,
        escape(xlabel),
        (MT + H - MB) / 2.0,
        escape(ylabel)
    );
    svg.push_str("</svg>");
    svg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_bonitos() {
        let t = nice_ticks(0.0, 1.0, 5);
        let want = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        assert_eq!(t.len(), want.len());
        assert!(t.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-9));
        let t = nice_ticks(0.0, 200.0, 8);
        assert_eq!(t.first(), Some(&0.0));
        assert_eq!(t.last(), Some(&200.0));
    }

    #[test]
    fn cuantiles_tipo7() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert!((quantile(&v, 0.25) - 1.75).abs() < 1e-12);
        assert!((quantile(&v, 0.5) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn svg_bien_formado() {
        let s = Series {
            name: "a<b".into(),
            points: vec![(0.0, 1.0), (1.0, 2.0)],
            band: vec![],
        };
        let svg = line_chart("t", "x", "y", &[s]);
        assert!(svg.starts_with("<svg") && svg.ends_with("</svg>"));
        assert!(svg.contains("a&lt;b"));
    }
}
