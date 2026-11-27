# Juego de plataformas sencillo para probar los niveles que genera nivelga.
#
# No es un simulador: es un control clásico con aceleración, gravedad, salto
# variable (soltar el botón corta el salto), tiempo coyote y búfer de salto.
# Los parámetros siguen a la física discreta del verificador de Rust: el salto
# alcanza unas 3,2 casillas de altura y unas 4,2 de distancia a la misma
# altura, así un nivel que el verificador declara jugable también se puede pasar.
#
# Al empezar se elige el modo. En el infinito los niveles se generan con
# `nivelga generar` en un hilo mientras se juega, de modo que siempre hay uno
# nuevo y distinto preparado. En el de evolución se juegan las etapas guardadas
# en godot/evolucion/ de una misma ejecución del GA, que parte de una población
# de ruido (tiles al azar): un individuo típico de ella y el mejor de varias
# generaciones hasta la última, con los datos de cada etapa en el marcador. La
# serie se calcula antes con `nivelga generar --evolucion`, nunca mientras se
# juega. Cada partida se
# registra en resultados/partidas.csv para comparar la dificultad medida (D)
# con la experiencia real de quien juega.
extends Node2D

# Unidades: píxeles del tile de 18 px y ticks de 1/60 s.
const T := 18.0
const GRAVEDAD := 0.5
const VEL_SALTO := 7.6
const VEL_CARRERA := 2.5
const CAIDA_MAX := 9.0
const VEL_ESCALERA := 1.6
const VEL_ENEMIGO := 0.5
const COYOTE := 6
const BUFER_SALTO := 6
const ANCHO_J := 12.0
const ALTO_J := 16.0
const TICKS_MUERTE := 45

var raiz := ""
var binario := ""
var dir_juego := ""

# Nivel actual.
var nivel: Dictionary = {}
var filas: PackedStringArray = PackedStringArray()
var ancho := 0
var alto := 14
var numero_nivel := 0
var nivel_fijo := ""
# "" mientras se muestra el menú, "infinito" o "evolucion".
var modo := ""
var dir_evolucion := ""
var etapas: PackedStringArray = PackedStringArray()
var etapa := 0

# Jugador.
var px := 0.0
var py := 0.0
var vx := 0.0
var vy := 0.0
var en_suelo := false
var en_escalera := false
var coyote := 0
var bufer := 0
var salto_pulsado := false
var mira_izquierda := false
var aparicion := Vector2.ZERO

# Partida.
var enemigos: Array = []
var monedas: Dictionary = {}
var muertes := 0
var ticks := 0
var muerto := 0
var ganado := false
var marcado := false
var depuracion := false
var anim := 0

# Nodos.
var capa_nivel: Node2D
var capa_depuracion: Node2D
var sprite_jugador: Sprite2D
var sprite_bandera: Sprite2D
var texturas_jugador: Array = []
var texturas_enemigo: Array = []
var texturas_bandera: Array = []
var camara: Camera2D
var hud: Label
var aviso: Label

# Generación en segundo plano.
var hilo: Thread = null
var siguiente := ""
var contador_generados := 0


func _ready() -> void:
	raiz = ProjectSettings.globalize_path("res://").path_join("..").simplify_path()
	binario = raiz.path_join("target/release/nivelga")
	dir_juego = raiz.path_join("resultados/juego")
	dir_evolucion = raiz.path_join("godot/evolucion")
	DirAccess.make_dir_recursive_absolute(dir_juego)
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--nivel="):
			nivel_fijo = a.trim_prefix("--nivel=")
	_configurar_controles()
	_crear_nodos()
	if nivel_fijo != "":
		_cargar(nivel_fijo)
	elif not FileAccess.file_exists(binario):
		aviso.text = "No encuentro %s.\nCompila antes con: cargo build --release" % binario
	else:
		_mostrar_menu()


# --- Menú y modos --------------------------------------------------------------

func _mostrar_menu() -> void:
	modo = ""
	nivel = {}
	for hijo in capa_nivel.get_children():
		hijo.queue_free()
	enemigos.clear()
	monedas.clear()
	sprite_jugador.visible = false
	capa_depuracion.visible = false
	hud.visible = false
	aviso.text = "Elige un modo:\n\n1  Infinito: un nivel nuevo y distinto tras otro\n" \
		+ "2  Evolución: etapas guardadas de una ejecución del GA, desde ruido hasta el nivel final\n\nEsc salir"


func _empezar_infinito() -> void:
	modo = "infinito"
	if _siguiente_listo():
		_cargar(siguiente)
		siguiente = ""
		_preparar_siguiente()
		return
	aviso.text = "Generando el primer nivel..."
	await get_tree().process_frame
	var ruta := _generar(_nuevo_destino())
	if ruta == "":
		aviso.text = "Falló la generación del nivel (ver la terminal)."
		return
	_cargar(ruta)
	_preparar_siguiente()


# Carga la serie guardada: etapa_0, etapa_1... hasta la primera que falte.
func _empezar_evolucion() -> void:
	etapas = PackedStringArray()
	while FileAccess.file_exists(dir_evolucion.path_join("etapa_%d/nivel.json" % etapas.size())):
		etapas.append(dir_evolucion.path_join("etapa_%d/nivel.json" % etapas.size()))
	if etapas.is_empty():
		aviso.text = "No hay ninguna evolución guardada en %s.
" % dir_evolucion \
			+ "Créala con: cargo run --release -- generar --evolucion\n\nEsc salir"
		return
	modo = "evolucion"
	etapa = 0
	_cargar(etapas[0])


func _configurar_controles() -> void:
	var mapa := {
		"izquierda": [KEY_LEFT, KEY_A],
		"derecha": [KEY_RIGHT, KEY_D],
		"arriba": [KEY_UP, KEY_W],
		"abajo": [KEY_DOWN, KEY_S],
		"saltar": [KEY_SPACE, KEY_Z, KEY_UP, KEY_W],
		"reiniciar": [KEY_R],
		"siguiente": [KEY_N, KEY_ENTER],
		"anterior": [KEY_B, KEY_BACKSPACE],
		"marcar": [KEY_F],
		"depurar": [KEY_F3, KEY_TAB],
		"salir": [KEY_ESCAPE],
		"menu": [KEY_M],
		"modo_infinito": [KEY_1, KEY_KP_1],
		"modo_evolucion": [KEY_2, KEY_KP_2],
	}
	for accion in mapa:
		if not InputMap.has_action(accion):
			InputMap.add_action(accion)
		for tecla in mapa[accion]:
			var ev := InputEventKey.new()
			ev.physical_keycode = tecla
			InputMap.action_add_event(accion, ev)


func _crear_nodos() -> void:
	capa_nivel = Node2D.new()
	add_child(capa_nivel)
	sprite_jugador = Sprite2D.new()
	sprite_jugador.centered = false
	sprite_jugador.z_index = 10
	add_child(sprite_jugador)
	capa_depuracion = Node2D.new()
	capa_depuracion.z_index = 20
	capa_depuracion.visible = false
	capa_depuracion.draw.connect(_dibujar_depuracion)
	add_child(capa_depuracion)
	camara = Camera2D.new()
	add_child(camara)
	camara.make_current()
	var capa_hud := CanvasLayer.new()
	add_child(capa_hud)
	hud = Label.new()
	hud.position = Vector2(12, 8)
	hud.add_theme_font_size_override("font_size", 18)
	hud.add_theme_color_override("font_outline_color", Color.BLACK)
	hud.add_theme_constant_override("outline_size", 6)
	# Panel oscuro detrás del texto para leerlo también sobre niveles cargados.
	var panel := StyleBoxFlat.new()
	panel.bg_color = Color(0, 0, 0, 0.5)
	panel.set_content_margin_all(6)
	hud.add_theme_stylebox_override("normal", panel)
	capa_hud.add_child(hud)
	aviso = Label.new()
	aviso.position = Vector2(12, 300)
	aviso.add_theme_font_size_override("font_size", 28)
	aviso.add_theme_color_override("font_outline_color", Color.BLACK)
	aviso.add_theme_constant_override("outline_size", 8)
	capa_hud.add_child(aviso)


# --- Generación de niveles ---------------------------------------------------

func _nuevo_destino() -> String:
	contador_generados += 1
	return dir_juego.path_join("nivel_%d_%d" % [Time.get_unix_time_from_system(), contador_generados])


# Ejecuta `nivelga generar` y devuelve la ruta del nivel creado; vacía si falla.
func _generar(destino: String) -> String:
	var args := ["generar", "--salida", destino, "--recursos", raiz.path_join("assets")]
	var modelo := raiz.path_join("modelos/naturalidad.nng")
	if FileAccess.file_exists(modelo):
		args.append_array(["--naturalidad", modelo])
	var salida := []
	if OS.execute(binario, args, salida, true) != 0:
		printerr("nivelga generar falló: ", salida)
		return ""
	return str(salida[0]).strip_edges()


func _preparar_siguiente() -> void:
	if nivel_fijo != "" or hilo != null:
		return
	siguiente = ""
	var destino := _nuevo_destino()
	hilo = Thread.new()
	hilo.start(_generar_en_hilo.bind(destino))


func _generar_en_hilo(destino: String) -> String:
	return _generar(destino)


func _siguiente_listo() -> bool:
	if hilo != null and not hilo.is_alive():
		siguiente = hilo.wait_to_finish()
		hilo = null
	return siguiente != ""


# --- Carga del nivel ---------------------------------------------------------

func _textura(img_path: String, indice: int, columnas: int, tam: int) -> Texture2D:
	var img := Image.load_from_file(img_path)
	var atlas := AtlasTexture.new()
	atlas.atlas = ImageTexture.create_from_image(img)
	atlas.region = Rect2((indice % columnas) * tam, (indice / columnas) * tam, tam, tam)
	return atlas


func _texturas(lista: Array) -> Array:
	var out := []
	for s in lista:
		out.append(_textura(s["hoja"], int(s["indice"]), int(s["columnas"]), int(s["tam"])))
	return out


func _cargar(ruta: String) -> void:
	var texto := FileAccess.get_file_as_string(ruta)
	var datos = JSON.parse_string(texto)
	if typeof(datos) != TYPE_DICTIONARY:
		aviso.text = "No se pudo leer %s" % ruta
		return
	nivel = datos
	# Las series guardadas usan rutas relativas: el fondo a la carpeta del nivel
	# y las hojas de sprites a la raíz del proyecto.
	if nivel["fondo"].is_relative_path():
		nivel["fondo"] = ruta.get_base_dir().path_join(nivel["fondo"])
	for lista in nivel["sprites"].values():
		for s in lista:
			if s["hoja"].is_relative_path():
				s["hoja"] = raiz.path_join(s["hoja"])
	filas = PackedStringArray(nivel["filas"])
	alto = filas.size()
	ancho = filas[0].length()
	numero_nivel += 1
	sprite_jugador.visible = true
	hud.visible = true
	capa_depuracion.visible = depuracion
	for hijo in capa_nivel.get_children():
		hijo.queue_free()
	var fondo := Sprite2D.new()
	fondo.centered = false
	fondo.texture = ImageTexture.create_from_image(Image.load_from_file(nivel["fondo"]))
	capa_nivel.add_child(fondo)
	var sprites: Dictionary = nivel["sprites"]
	texturas_jugador = _texturas(sprites["jugador"])
	texturas_enemigo = _texturas(sprites["enemigo"])
	texturas_bandera = _texturas(sprites["meta"])
	var texturas_moneda := _texturas(sprites["monedas"])
	enemigos.clear()
	monedas.clear()
	for y in alto:
		for x in ancho:
			match _tile(x, y):
				"@":
					aparicion = Vector2(x * T + (T - ANCHO_J) / 2.0, (y + 1) * T - ALTO_J)
				"o":
					var m := Sprite2D.new()
					m.centered = false
					m.texture = texturas_moneda[(x * 7 + y) % texturas_moneda.size()]
					m.position = Vector2(x * T, y * T)
					capa_nivel.add_child(m)
					monedas[Vector2i(x, y)] = m
				"E":
					_crear_enemigo(x, y)
				"!":
					var asta := Sprite2D.new()
					asta.centered = false
					asta.texture = texturas_bandera[0]
					asta.position = Vector2(x * T, y * T)
					capa_nivel.add_child(asta)
					sprite_bandera = Sprite2D.new()
					sprite_bandera.centered = false
					sprite_bandera.texture = texturas_bandera[1]
					sprite_bandera.position = Vector2(x * T, (y - 1) * T)
					capa_nivel.add_child(sprite_bandera)
	camara.limit_left = 0
	camara.limit_top = 0
	camara.limit_right = int(ancho * T)
	camara.limit_bottom = int(alto * T)
	muertes = 0
	ticks = 0
	ganado = false
	marcado = false
	_reaparecer()
	aviso.text = ""


func _crear_enemigo(x: int, y: int) -> void:
	var lo := x
	var hi := x
	if _libre_con_apoyo(x - 1, y):
		lo -= 1
	if _libre_con_apoyo(x + 1, y):
		hi += 1
	var s := Sprite2D.new()
	s.centered = false
	s.texture = texturas_enemigo[0]
	s.z_index = 8
	capa_nivel.add_child(s)
	enemigos.append({
		"x": x * T + 2.0, "y": y * T + 4.0, "dir": 1.0 if x % 2 == 0 else -1.0,
		"min": lo * T + 2.0, "max": hi * T + 2.0, "sprite": s,
	})


# --- Consultas sobre la rejilla ---------------------------------------------

func _tile(x: int, y: int) -> String:
	if x < 0 or y < 0 or x >= ancho or y >= alto:
		return "-"
	return filas[y][x]


# Solo el suelo bloquea en todas direcciones; las plataformas ("S") son de un
# solo sentido: se atraviesan al subir o al caminar y sostienen al caer encima.
func _solido(x: int, y: int) -> bool:
	if x < 0 or x >= ancho:
		return true # los lados del nivel son paredes
	return _tile(x, y) == "X"


func _apoyo(x: int, y: int) -> bool:
	return _solido(x, y) or _tile(x, y) == "S" or _tile(x, y) == "L"


func _libre_con_apoyo(x: int, y: int) -> bool:
	var t := _tile(x, y)
	var libre := not (t == "X" or t == "^" or t == "E") and x >= 0 and x < ancho
	return libre and _apoyo(x, y + 1)


func _celda(v: float) -> int:
	return int(floor(v / T))


func _caja_choca(x: float, y: float, w: float, h: float) -> bool:
	for cx in range(_celda(x), _celda(x + w - 0.01) + 1):
		for cy in range(_celda(y), _celda(y + h - 0.01) + 1):
			if cy >= 0 and _solido(cx, cy):
				return true
	return false


func _celdas_tocadas(margen: float) -> Array:
	var out := []
	for cx in range(_celda(px + margen), _celda(px + ANCHO_J - margen) + 1):
		for cy in range(_celda(py + margen), _celda(py + ALTO_J - margen) + 1):
			out.append(Vector2i(cx, cy))
	return out


# Apoyo de un solo sentido bajo los pies: una plataforma o el tope de una escalera.
func _apoyo_un_sentido_bajo(x: float, pies: float) -> bool:
	var cy := _celda(pies)
	for fx in [x + 1.0, x + ANCHO_J - 1.0]:
		var cx := _celda(fx)
		var t := _tile(cx, cy)
		if t == "S" or (t == "L" and _tile(cx, cy - 1) != "L"):
			return true
	return false


# --- Bucle del juego ---------------------------------------------------------

func _controles() -> Dictionary:
	return {
		"izq": Input.is_action_pressed("izquierda"),
		"der": Input.is_action_pressed("derecha"),
		"arriba": Input.is_action_pressed("arriba"),
		"abajo": Input.is_action_pressed("abajo"),
		"salto": Input.is_action_pressed("saltar"),
	}


func _physics_process(_delta: float) -> void:
	if nivel.is_empty():
		return
	anim += 1
	_mover_enemigos()
	if ganado:
		_actualizar_vista()
		return
	if muerto > 0:
		muerto -= 1
		if muerto == 0:
			_reaparecer()
		_actualizar_vista()
		return
	ticks += 1
	var c := _controles()
	var objetivo := VEL_CARRERA * (float(c["der"]) - float(c["izq"]))
	var acel := 0.5 if en_suelo else 0.3
	vx += clamp(objetivo - vx, -acel, acel)
	if c["izq"] != c["der"]:
		mira_izquierda = c["izq"]
	var en_celda_escalera := false
	for k in _celdas_tocadas(3.0):
		if _tile(k.x, k.y) == "L":
			en_celda_escalera = true
	if en_celda_escalera and (c["arriba"] or c["abajo"]):
		en_escalera = true
	if not en_celda_escalera:
		en_escalera = false
	var pulsado: bool = c["salto"] and not salto_pulsado
	salto_pulsado = c["salto"]
	bufer = BUFER_SALTO if pulsado else max(bufer - 1, 0)
	coyote = COYOTE if (en_suelo or en_escalera) else max(coyote - 1, 0)
	if bufer > 0 and coyote > 0 and not (en_escalera and c["arriba"] and not pulsado):
		vy = -VEL_SALTO
		bufer = 0
		coyote = 0
		en_escalera = false
	if not c["salto"] and vy < -3.0:
		vy = -3.0
	if en_escalera:
		vy = VEL_ESCALERA * (float(c["abajo"]) - float(c["arriba"]))
		vx *= 0.5
	else:
		vy = min(vy + GRAVEDAD, CAIDA_MAX)
	_mover_x(vx)
	_mover_y(vy, c["abajo"])
	_interactuar()
	_actualizar_vista()


func _mover_x(dx: float) -> void:
	var nx := px + dx
	if _caja_choca(nx, py, ANCHO_J, ALTO_J):
		if dx > 0.0:
			px = _celda(nx + ANCHO_J) * T - ANCHO_J - 0.01
		elif dx < 0.0:
			px = (_celda(nx) + 1) * T + 0.01
		vx = 0.0
	else:
		px = nx


func _mover_y(dy: float, abajo: bool) -> void:
	var ny := py + dy
	var pies := py + ALTO_J
	en_suelo = false
	# Aterrizaje sobre un apoyo de un solo sentido: solo al bajar y cruzando su
	# borde superior; con la flecha abajo se deja caer a través.
	var sobre_plataforma := dy > 0.0 and not abajo and not en_escalera \
		and _celda(pies - 0.01) < _celda(ny + ALTO_J) and _apoyo_un_sentido_bajo(px, ny + ALTO_J)
	if _caja_choca(px, ny, ANCHO_J, ALTO_J) or sobre_plataforma:
		if dy > 0.0:
			py = _celda(ny + ALTO_J) * T - ALTO_J
			en_suelo = true
		else:
			py = (_celda(ny) + 1) * T
		vy = 0.0
	else:
		py = ny


func _interactuar() -> void:
	if py > alto * T:
		_morir()
		return
	for k in _celdas_tocadas(1.0):
		var t := _tile(k.x, k.y)
		if t == "o" and monedas.has(k):
			monedas[k].queue_free()
			monedas.erase(k)
		elif t == "!":
			_ganar()
			return
	# Pinchos y enemigos usan una caja algo menor (más justo para quien juega).
	for k in _celdas_tocadas(3.0):
		if _tile(k.x, k.y) == "^":
			_morir()
			return
	for e in enemigos:
		if px + 2.0 < e["x"] + 14.0 and px + ANCHO_J - 2.0 > e["x"] \
				and py + 2.0 < e["y"] + 14.0 and py + ALTO_J > e["y"] + 2.0:
			_morir()
			return


func _mover_enemigos() -> void:
	for e in enemigos:
		e["x"] += e["dir"] * VEL_ENEMIGO
		if e["x"] <= e["min"] or e["x"] >= e["max"]:
			e["x"] = clamp(e["x"], e["min"], e["max"])
			e["dir"] = -e["dir"]
		var s: Sprite2D = e["sprite"]
		s.position = Vector2(e["x"] - 5.0, e["y"] - 10.0)
		s.flip_h = e["dir"] > 0.0
		s.texture = texturas_enemigo[(anim / 12) % texturas_enemigo.size()]


func _morir() -> void:
	muertes += 1
	muerto = TICKS_MUERTE


func _reaparecer() -> void:
	px = aparicion.x
	py = aparicion.y
	vx = 0.0
	vy = 0.0
	en_escalera = false


func _ganar() -> void:
	ganado = true
	_registrar("ganado")
	aviso.text = "¡Nivel superado en %.1f s con %d muertes!\nEnter o N: siguiente nivel" % [ticks / 60.0, muertes]


# --- Vista, marcador y registro ---------------------------------------------

func _actualizar_vista() -> void:
	var visible_j := muerto == 0 or (muerto / 4) % 2 == 1
	sprite_jugador.visible = visible_j
	if texturas_jugador.size() > 0:
		var andando: bool = abs(vx) > 0.3 and en_suelo
		sprite_jugador.texture = texturas_jugador[(anim / 8) % 2 if andando else 0]
	sprite_jugador.flip_h = mira_izquierda
	sprite_jugador.position = Vector2(px - 6.0, py - 8.0)
	if sprite_bandera != null and is_instance_valid(sprite_bandera):
		sprite_bandera.texture = texturas_bandera[1 + (anim / 15) % 2]
	var alto_vista := get_viewport_rect().size.y
	var zoom := alto_vista / (alto * T)
	camara.zoom = Vector2(zoom, zoom)
	camara.position = Vector2(px + get_viewport_rect().size.x / zoom * 0.15, alto * T / 2.0)
	var m: Dictionary = nivel["metricas"]
	var total_monedas := int(m["monedas"])
	var cabecera := "Nivel %d · " % numero_nivel
	var teclas := "N siguiente"
	if nivel.has("etapa"):
		var e: Dictionary = nivel["etapa"]
		cabecera = "EVOLUCIÓN, etapa %d de %d: %s\n" % [e["numero"], e["total"], e["titulo"]] \
			+ "Este nivel: aptitud %.3f · %s · calidad %.2f · dificultad %.2f · variedad %.2f · coherencia %.2f\n" % [
				m["aptitud"], "jugable" if m["jugable"] else "NO jugable", m["calidad"],
				m["dificultad"], m["variedad"], m["coherencia"]
			] + "Población en la generación %d: aptitud media %.3f · %.0f%% jugables · diversidad %.3f\n" % [
				e["generacion"], e["media"], 100.0 * e["jugables"], e["diversidad"]
			]
		teclas = "N siguiente etapa · B anterior"
	hud.text = "%s%s · %d columnas · dificultad medida %.2f\n" % [
		cabecera, nivel["estilo"], ancho, m["dificultad"]
	] + "Muertes %d · monedas %d/%d · %.1f s%s\n" % [
		muertes, total_monedas - monedas.size(), total_monedas, ticks / 60.0, " · MARCADO" if marcado else ""
	] + "←→ moverse · espacio saltar · ↑↓ escaleras · R reiniciar · %s · F marcar · F3 depurar · M menú · Esc salir" % teclas
	if depuracion:
		hud.text += "\nDepuración: blanco suelo · cian plataforma de un sentido · amarillo escalera\n" \
			+ "rojo mata · naranja patrulla de enemigo · dorado moneda · verde meta · gris decoración · azul jugador"
		capa_depuracion.queue_redraw()
	if ganado and modo == "infinito" and not _siguiente_listo() and hilo != null:
		aviso.text = "¡Nivel superado! Generando el siguiente..."


func _unhandled_input(ev: InputEvent) -> void:
	if ev.is_action_pressed("salir"):
		if not nivel.is_empty() and not ganado:
			_registrar("abandonado")
		_cerrar()
	elif modo == "" and nivel_fijo == "":
		if ev.is_action_pressed("modo_infinito"):
			_empezar_infinito()
		elif ev.is_action_pressed("modo_evolucion"):
			_empezar_evolucion()
	elif nivel.is_empty():
		return
	elif ev.is_action_pressed("menu") and nivel_fijo == "":
		if not ganado:
			_registrar("abandonado")
		_mostrar_menu()
	elif ev.is_action_pressed("reiniciar"):
		_reaparecer()
	elif ev.is_action_pressed("marcar"):
		_marcar()
	elif ev.is_action_pressed("depurar"):
		depuracion = not depuracion
		capa_depuracion.visible = depuracion
	elif ev.is_action_pressed("siguiente"):
		_pasar_al_siguiente()
	elif ev.is_action_pressed("anterior") and modo == "evolucion" and etapa > 0:
		etapa -= 1
		_cargar(etapas[etapa])


func _pasar_al_siguiente() -> void:
	if nivel_fijo != "":
		return
	if not ganado:
		_registrar("saltado")
	if modo == "evolucion":
		etapa += 1
		if etapa < etapas.size():
			_cargar(etapas[etapa])
		else:
			_mostrar_menu()
			aviso.text = "Fin de la evolución.\n\n" + aviso.text
		return
	if not _siguiente_listo():
		aviso.text = "Generando el siguiente nivel..."
		if hilo == null:
			_preparar_siguiente()
		await get_tree().create_timer(0.2).timeout
		_pasar_al_siguiente_esperando()
		return
	_cargar(siguiente)
	siguiente = ""
	_preparar_siguiente()


func _pasar_al_siguiente_esperando() -> void:
	while not _siguiente_listo():
		await get_tree().create_timer(0.2).timeout
	_cargar(siguiente)
	siguiente = ""
	_preparar_siguiente()


# Contorno de colores por tipo de celda con las cajas de colisión reales: así
# se ve qué es suelo, qué mata y qué es solo decoración.
func _dibujar_depuracion() -> void:
	var c := capa_depuracion
	var colores := {
		"X": Color(1, 1, 1, 0.9), "L": Color(1, 0.9, 0.2, 0.9),
		"o": Color(1, 0.75, 0.1, 0.9), "!": Color(0.2, 1, 0.3, 0.9), "d": Color(0.6, 0.6, 0.6, 0.7),
	}
	for y in alto:
		for x in ancho:
			var t := _tile(x, y)
			var r := Rect2(x * T, y * T, T, T)
			if t == "S":
				c.draw_rect(r, Color(0.2, 0.9, 1, 0.5), false, 1.0)
				c.draw_line(r.position, r.position + Vector2(T, 0), Color(0.2, 0.9, 1), 3.0)
			elif t == "^":
				c.draw_rect(r.grow(-3.0), Color(1, 0.1, 0.1, 0.45), true)
			elif t == "o" and not monedas.has(Vector2i(x, y)):
				continue
			elif colores.has(t):
				c.draw_rect(r.grow(-0.5), colores[t], false, 1.0)
	for e in enemigos:
		var zona := Rect2(e["min"], e["y"], e["max"] - e["min"] + 14.0, 14.0)
		c.draw_rect(zona, Color(1, 0.55, 0.1, 0.3), true)
		c.draw_rect(Rect2(e["x"], e["y"] + 2.0, 14.0, 12.0), Color(1, 0.1, 0.1, 0.9), false, 1.5)
	c.draw_rect(Rect2(px, py, ANCHO_J, ALTO_J), Color(0.3, 0.5, 1, 0.9), false, 1.5)


# Guarda una copia del nivel para revisarlo después (algo que no está en orden).
func _marcar() -> void:
	marcado = true
	var dir_marcados := raiz.path_join("resultados/marcados").path_join(str(nivel["semilla"]))
	DirAccess.make_dir_recursive_absolute(dir_marcados)
	var f := FileAccess.open(dir_marcados.path_join("nivel.json"), FileAccess.WRITE)
	f.store_string(JSON.stringify(nivel))
	f.close()
	DirAccess.copy_absolute(nivel["fondo"], dir_marcados.path_join("fondo.png"))
	aviso.text = "Nivel marcado en resultados/marcados/%s" % nivel["semilla"]


# Solo se registran las partidas del modo infinito: son niveles finales del GA
# y es con ellos con los que se compara la dificultad medida.
func _registrar(resultado: String) -> void:
	if modo == "evolucion":
		return
	var ruta := raiz.path_join("resultados/partidas.csv")
	var nuevo := not FileAccess.file_exists(ruta)
	var f := FileAccess.open(ruta, FileAccess.WRITE if nuevo else FileAccess.READ_WRITE)
	if f == null:
		return
	if nuevo:
		f.store_line("fecha,semilla,estilo,ancho,dificultad_objetivo,dificultad,variedad,coherencia,saltos,enemigos,peligros,huecos,resultado,muertes,segundos,monedas,monedas_total,marcado")
	f.seek_end()
	var m: Dictionary = nivel["metricas"]
	var total := int(m["monedas"])
	f.store_line("%s,%s,%s,%d,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d,%s,%d,%.2f,%d,%d,%s" % [
		Time.get_datetime_string_from_system(), nivel["semilla"], nivel["estilo"], ancho,
		nivel["dificultad_objetivo"], m["dificultad"], m["variedad"], m["coherencia"],
		int(m["saltos"]), int(m["enemigos"]), int(m["peligros"]), int(m["huecos"]),
		resultado, muertes, ticks / 60.0, total - monedas.size(), total, "si" if marcado else "no",
	])
	f.close()


func _cerrar() -> void:
	if hilo != null:
		hilo.wait_to_finish()
		hilo = null
	get_tree().quit()


func _exit_tree() -> void:
	if hilo != null:
		hilo.wait_to_finish()
