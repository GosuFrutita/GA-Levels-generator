# Resultados experimentales

Todas las cifras se obtienen con `cargo run --release -- experimento todos
--repeticiones 30 --naturalidad modelos/naturalidad.nng`. Cada variante se
ejecuta con las mismas 30 semillas (números aleatorios comunes) y un
presupuesto fijo de 20 000 evaluaciones (población 100, 200 generaciones,
2 élites). Se informa la mediana con su rango intercuartílico. Las variantes se
comparan con Kruskal-Wallis, con Mann-Whitney ajustado por Holm frente a la
referencia y con el tamaño de efecto Â12 de Vargha y Delaney (0,5 = sin efecto;
umbrales 0,56, 0,64 y 0,71 para efecto pequeño, mediano y grande). Los informes
completos, los CSV y las gráficas quedan en `resultados/experimentos/eN/`.

## Algoritmo genético

| Exp. | Pregunta | Resultado |
|---|---|---|
| E1 | ¿Aporta el GA? | El GA (0,9947) supera a la búsqueda aleatoria (0,9652; Â12 = 1,00) y al GA sin cruce (0,9929; Â12 = 0,67). El (1+1)-EA queda ligeramente por encima (0,9958; Â12 = 0,28 a su favor). |
| E2 | Ruleta frente a torneo | El torneo gana a todas las ruletas (Â12 ≥ 0,94). Torneo de 5 o 7 (0,996) > torneo de 3 (0,995) > torneo de 2 (0,989) > ruletas (0,976 a 0,980). |
| E3 | Elitismo | Sin élites la aptitud cae (0,9733; Â12 = 1,00 frente a 2 élites). Entre 1 y 10 élites las diferencias son pequeñas. |
| E4 | Cruce | Columnas (0,9947) y bloque 2D (0,9953) empatan (p = 0,33). El cruce uniforme es peor (0,9854; Â12 = 0,91). |
| E5 | Sensibilidad | La probabilidad de cruce entre 0,6 y 1,0 apenas influye. Con 5 mutaciones por hijo la aptitud baja (0,984 a 0,986 frente a 0,995 a 0,996). |
| E6 | Agregación | Suma ponderada, media geométrica, penalización y reglas de Deb no difieren (Kruskal-Wallis p = 0,10). |
| E7 | Nueve instancias (D* × ancho) | Friedman p = 9,8·10⁻⁶. Rangos medios: (1+1)-EA 4,0; torneo-3 3,0; ruleta 1,9; aleatoria 1,1. |
| E9 | Rango expresivo | La dificultad del mejor nivel cae entre 0,486 y 0,514 con objetivo 0,5. Estilos elegidos: pradera 15, granja 13, dulce 2. |
| E10 | Bandido D-MAB | Sin diferencia frente al operador uniforme (p = 0,72). |
| E11 | Diversidad | Fitness sharing multiplica la diversidad final por 8 a 9 (0,18 a 0,23 frente a 0,024) con menor aptitud (0,971 a 0,980). El crowding alcanza la mejor aptitud (0,9953 con emparejamiento aleatorio de Mahfoud; 0,9961 con torneo). |

**Intensidad de selección (E2).** La medida al seleccionar es de 0,39 a 0,47 en
los torneos y de 0,34 a 0,38 en las ruletas. Ordena bien los métodos pero queda
por debajo de la teórica de Blickle y Thiele (0,58 con t = 2 y 1,34 con t = 7)
y apenas crece con t. La fórmula supone una aptitud normal y continua. Aquí la
población converge pronto, acumula clones con la misma aptitud (elitismo) y se
concentra cerca de 1. Con empates un torneo grande no distingue más que uno
pequeño.

## Componentes de aprendizaje

**Red de naturalidad (cross-entropy).** MLP de 7 846 parámetros entrenado
sobre 12 niveles de Super Mario Bros del VGLC y validado con otros 3. NLL de
validación por tile: 0,129, frente a 0,139 de una cadena de Markov 2D y 0,467
del unigrama. En E8 el término de naturalidad sube la naturalidad del mejor
nivel de 0,544 a 0,788 (Â12 = 0,99) sin perder jugabilidad, dificultad ni
coherencia.

**Agente A2C (cross-entropy ponderada por la ventaja).** 20 millones de pasos
en unos 5 minutos de CPU con niveles nuevos en cada episodio. Gana el 78,9 % de
los niveles aleatorios nuevos jugables (0 % sin entrenar) y su tasa de
victorias correlaciona con la dificultad D (Spearman −0,45). En los niveles
evolucionados por el GA solo gana el 4 %: están optimizados hacia la
dificultad objetivo con muchos saltos ajustados encadenados. Nunca gana un
nivel que el verificador declara injugable, lo que confirma que ambos usan la
misma física.

**Programación Genética.** Clasificador simbólico de jugabilidad entrenado con
cross-entropy binaria, escalado lineal de Keijzer y presión de parsimonia,
sobre 20 000 niveles equilibrados etiquetados por el verificador. En prueba:

| Modelo | AUC | Exactitud | Tamaño |
|---|---|---|---|
| Regresión logística | 0,928 | 0,858 | 22 parámetros |
| MLP | 0,940 | 0,869 | 737 parámetros |
| PG | 0,919 | 0,853 | 21 nodos |

La expresión de la PG es legible: un nivel es menos jugable cuantos más
pinchos, huecos y enemigos junto a los huecos tiene, cuanto mayor es su vano
sin apoyo y cuanto más alta es su mayor subida.

## Hallazgos sobre el diseño de niveles

- Las primeras versiones de la aptitud premiaban el ruido. Una penalización de
  coherencia con tope formaba una meseta (F = 0,8785 en las 30 semillas) y un
  objetivo de variedad fijado a mano pedía seis veces más patrones distintos
  que un nivel humano (mediana 0,056 en el VGLC). Calibrar con niveles humanos
  y medir la variedad con patrones de diseño resolvió ambos problemas.
- El gen de tileset convergía casi siempre a pradera (27 de 30) porque los
  niveles iniciales encajaban mejor con su personalidad. Con el generador de
  patrones el reparto es más equilibrado (15, 13 y 2), aunque industrial sigue
  sin elegirse.

## Limitaciones

- **Efecto techo.** Con la inicialización por patrones casi todas las variantes
  llegan a F ≈ 0,99, de modo que varias comparaciones discriminan poco. Un
  objetivo más exigente (por ejemplo la naturalidad o el probador de RL como
  términos de la aptitud) daría más margen.
- **El probador de RL** generaliza a niveles aleatorios pero no a los niveles
  difíciles que produce el GA; haría falta un currículo con más niveles
  evolucionados o más entrenamiento.
- **Las cifras de diseño** (envolvente de salto, mezcla objetivo de patrones,
  rangos de cada estilo) son propuestas razonadas a partir de las fuentes y del
  corpus, no valores publicados.

## Fuentes

Las notas de investigación con todas las referencias están en
`research_notes/` (fuera del repositorio): diseño de niveles de plataformas,
análisis de proyectos de Godot, PCG evolutivo, operadores y evaluación de GA,
redes con cross-entropy, aprendizaje por refuerzo y convenciones de Rust.
