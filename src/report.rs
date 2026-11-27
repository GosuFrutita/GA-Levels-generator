//! Escritura de resultados de una ejecución: CSV, gráficas SVG, niveles en
//! texto y PNG con la trayectoria del verificador.

use crate::error::Result;
use crate::fitness::Evaluation;
use crate::ga::{GaConfig, GenStats, RunResult};
use crate::image::{Image, png};
use crate::level::{Level, Style};
use crate::play::{Checker, PlayScratch};
use crate::plot::{Series, line_chart};
use crate::render::{RenderOptions, Tilesets, render};
use std::fmt::Write as _;
use std::path::Path;

/// Cabecera del CSV del historial.
pub const HISTORY_HEADER: &str = "generacion,evaluaciones,mejor,mejor_historico,media,peor,desv,jugables,diversidad,entropia_celdas,entropia_estilo,pradera,granja,dulce,industrial,intensidad_seleccion,mejor_q,mejor_d,mejor_v,mejor_c,uso_hueco,uso_altura,uso_patron,uso_enemigo,uso_copia,uso_monedas,uso_decoracion";

/// Historial en CSV (separador coma, punto decimal).
pub fn history_csv(history: &[GenStats]) -> String {
    let mut s = String::with_capacity(history.len() * 160);
    s.push_str(HISTORY_HEADER);
    s.push('\n');
    for h in history {
        let _ = write!(
            s,
            "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.4},{:.6},{:.6},{:.6}",
            h.generation,
            h.evaluations,
            h.best,
            h.best_ever,
            h.mean,
            h.worst,
            h.std,
            h.feasible,
            h.diversity,
            h.cell_entropy,
            h.style_entropy
        );
        for c in h.style_counts {
            let _ = write!(s, ",{c}");
        }
        let _ = write!(s, ",{:.6}", h.selection_intensity);
        for o in h.best_objectives {
            let _ = write!(s, ",{o:.6}");
        }
        // Uso de cada operador macro según el bandido (ceros sin bandido).
        for u in h.macro_usage {
            let _ = write!(s, ",{u:.4}");
        }
        s.push('\n');
    }
    s
}

fn series(name: &str, history: &[GenStats], f: impl Fn(&GenStats) -> f64) -> Series {
    Series {
        name: name.into(),
        points: history
            .iter()
            .map(|h| (h.generation as f64, f(h)))
            .collect(),
        band: vec![],
    }
}

/// Resumen legible de una evaluación.
pub fn describe(e: &Evaluation) -> String {
    let f = &e.features;
    format!(
        "F={:.4} jugable={} Q={:.3} D={:.3} (f_dif={:.3}) V={:.3} C={:.3} estilo_fit={:.3} nat={:.3}\n\
         enemigos={} peligros={} huecos={} (máx {}) saltos={} (ajustados {}) monedas={} (trampa {}) escaleras={} \
         plataformas={} deco={} violaciones={} inalcanzable={:.0}% callejones={:.0}% linealidad={:.2} leniencia={:.2} patrones={:.3} entropía={:.3}",
        e.fitness,
        if e.feasible { "sí" } else { "no" },
        e.quality,
        e.difficulty,
        e.f_difficulty,
        e.variety,
        e.coherence,
        e.style_fit,
        e.naturalness,
        f.enemies,
        f.hazards,
        f.gaps,
        f.gap_max,
        f.jumps,
        f.tight_jumps,
        f.coins,
        f.unreachable_coins,
        f.ladders,
        f.platforms,
        f.decor,
        f.violations,
        100.0 * f.unreachable_structure,
        100.0 * f.trapped,
        f.linearity,
        f.leniency,
        f.pattern_ratio,
        f.entropy
    )
}

/// Dibuja un nivel con su trayectoria óptima.
pub fn render_with_path(level: &Level, checker: &Checker, sets: &Tilesets, scale: u32) -> Image {
    let res = checker.check(level, &mut PlayScratch::default());
    let path = checker.trajectory(level, &res);
    render(level, sets, &RenderOptions { scale, path })
}

/// Apila imágenes verticalmente con un margen.
pub fn vstack(images: &[Image], gap: u32, bg: [u8; 4]) -> Image {
    let w = images.iter().map(Image::width).max().unwrap_or(1);
    let h = images.iter().map(|i| i.height() + gap).sum::<u32>().max(1);
    let mut out = Image::new(w, h, bg);
    let mut y = 0;
    for img in images {
        out.draw(img, 0, i64::from(y), 1);
        y += img.height() + gap;
    }
    out
}

/// Guarda todos los resultados de una ejecución en `dir`.
///
/// # Errors
///
/// Si no se puede escribir algún archivo o cargar los tilesets.
pub fn save_run(
    dir: &Path,
    cfg: &GaConfig,
    result: &RunResult,
    checker: &Checker,
    assets: &Path,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("configuracion.txt"), format!("{cfg:#?}\n"))?;
    std::fs::write(dir.join("historial.csv"), history_csv(&result.history))?;
    let h = &result.history;
    let curves = line_chart(
        "Aptitud por generación",
        "generación",
        "aptitud",
        &[
            series("mejor", h, |s| s.best),
            series("media", h, |s| s.mean),
            series("peor", h, |s| s.worst),
            series("jugables", h, |s| s.feasible),
        ],
    );
    std::fs::write(dir.join("aptitud.svg"), curves)?;
    let objectives = line_chart(
        "Objetivos del mejor individuo",
        "generación",
        "valor",
        &[
            series("calidad Q", h, |s| s.best_objectives[0]),
            series("dificultad D", h, |s| s.best_objectives[1]),
            series("variedad V", h, |s| s.best_objectives[2]),
            series("coherencia C", h, |s| s.best_objectives[3]),
        ],
    );
    std::fs::write(dir.join("objetivos.svg"), objectives)?;
    let div = line_chart(
        "Diversidad de la población",
        "generación",
        "valor normalizado",
        &[
            series("Hamming medio", h, |s| s.diversity),
            series("entropía por celda", h, |s| s.cell_entropy),
            series("entropía del estilo", h, |s| s.style_entropy),
        ],
    );
    std::fs::write(dir.join("diversidad.svg"), div)?;
    let styles = line_chart(
        "Individuos por tileset",
        "generación",
        "individuos",
        &Style::ALL.map(|st| series(st.name(), h, |s| s.style_counts[st.index()] as f64)),
    );
    std::fs::write(dir.join("estilos.svg"), styles)?;

    let sets = Tilesets::load(assets)?;
    std::fs::write(dir.join("mejor.txt"), result.best.to_string())?;
    std::fs::write(
        dir.join("mejor_evaluacion.txt"),
        describe(&result.best_eval) + "\n",
    )?;
    png::save(
        &render_with_path(&result.best, checker, &sets, 2),
        dir.join("mejor.png"),
    )?;
    let mut gallery = Vec::new();
    for (l, e) in result.best_by_style.iter().flatten() {
        let name = l.style.name();
        std::fs::write(dir.join(format!("mejor_{name}.txt")), l.to_string())?;
        std::fs::write(
            dir.join(format!("mejor_{name}_evaluacion.txt")),
            describe(e) + "\n",
        )?;
        let img = render_with_path(l, checker, &sets, 1);
        png::save(&img, dir.join(format!("mejor_{name}.png")))?;
        gallery.push(img);
    }
    if !gallery.is_empty() {
        png::save(
            &vstack(&gallery, 6, [30, 30, 40, 255]),
            dir.join("galeria_estilos.png"),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_con_cabecera_y_columnas_consistentes() {
        let h = vec![GenStats::default(); 3];
        let csv = history_csv(&h);
        let cols = HISTORY_HEADER.split(',').count();
        for line in csv.lines() {
            assert_eq!(line.split(',').count(), cols, "{line}");
        }
        assert_eq!(csv.lines().count(), 4);
    }
}
