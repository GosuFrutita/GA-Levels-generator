# GA Levels Generator (`nivelga`)

Generación automática de niveles de plataformas 2D con un algoritmo genético. El
cromosoma es la matriz de tipos lógicos de tiles más un gen que elige el estilo
visual entre los cuatro packs *Pixel Platformer* de Kenney. La función de
adaptación busca niveles jugables con dificultad moderada, variedad estructural
y coherencia visual.

Todo está escrito a mano en Rust, sin dependencias externas: generador
aleatorio, lectura y escritura de PNG (incluido DEFLATE), verificador de
jugabilidad, métricas, operadores genéticos, redes neuronales, estadística y
gráficas.

## Requisitos

- Rust 1.100 o posterior (edición 2024).

## Uso

```sh
cargo run --release -- ayuda
cargo run --release -- evolucionar --semilla 3 --salida resultados/prueba
cargo run --release -- ver resultados/prueba/mejor.txt
cargo run --release -- dibujar resultados/prueba/mejor.txt mejor.png --escala 2
cargo run --release -- comprobar resultados/prueba/mejor.txt
```

`evolucionar` guarda en el directorio de salida el historial por generación
(`historial.csv`), las gráficas (`aptitud.svg`, `objetivos.svg`,
`diversidad.svg` y `estilos.svg`) junto con el mejor nivel en texto y en PNG
con su trayectoria óptima. También guarda el mejor nivel jugable de cada estilo.

## Red de naturalidad (cross-entropy)

Un MLP aprende a predecir cada tile a partir de su contexto en los niveles de
Super Mario Bros del VGLC. Su log-verosimilitud negativa se usa como término de
aptitud para que los niveles se parezcan a niveles diseñados por personas.

```sh
sh scripts/descargar_vglc.sh            # datos (no se versionan)
cargo run --release -- entrenar-nn      # guarda modelos/naturalidad.nng
cargo run --release -- evolucionar --naturalidad modelos/naturalidad.nng
```

## Experimentos

```sh
cargo run --release -- experimentos     # lista e1 a e10
cargo run --release -- experimento e1 --repeticiones 30
cargo run --release -- experimento todos --repeticiones 30 --naturalidad modelos/naturalidad.nng
```

Cada experimento usa las mismas semillas para todas las variantes y guarda el
CSV por ejecución, el resumen, las curvas de convergencia, los diagramas de
caja y un informe con Kruskal-Wallis, Mann-Whitney con Holm, Â12, Fisher y
Friedman.

## Componentes

| Módulo | Contenido |
| --- | --- |
| `level` | Tiles lógicos, estilos, nivel guardado por columnas, formato de texto |
| `play` | Verificador de jugabilidad: arcos de salto, Dijkstra con cubetas, trayectoria |
| `metrics` | Rasgos, dificultad, variedad y coherencia |
| `fitness` | Agregación multiobjetivo con la jugabilidad como restricción dura |
| `ga` | Selección (torneo y ruleta con escalado y SUS), cruce, mutación, elitismo, reparación, bandido D-MAB |
| `nn` | Red neuronal con cross-entropy, Adam y chequeo de gradientes |
| `naturalness`, `vglc` | Modelo de naturalidad de tiles y lector del corpus VGLC |
| `baselines`, `experiment` | Búsqueda aleatoria, (1+1)-EA y ejecutor de experimentos |
| `stats` | Mann-Whitney, Â12, Fisher, Kruskal-Wallis, Friedman y Holm |
| `image`, `render`, `plot`, `ansi` | PNG, dibujo con tilesets, gráficas SVG y visor en terminal |

## Calidad

```sh
cargo fmt --check
cargo clippy --all-targets
cargo test
cargo bench --bench evaluacion
```

El código usa `clippy::pedantic` sin avisos y prohíbe `unsafe`. Los
comentarios y mensajes están en español. El test `tests/estilo.rs` comprueba
que no haya rayas ni comas antes o después de la conjunción «y».

## Créditos

Tilesets *Pixel Platformer* y sus expansiones Farm, Food e Industrial de
[Kenney](https://kenney.nl) (CC0).
