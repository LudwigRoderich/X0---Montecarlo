from anytree import Node
import tkinter as tk
from tkinter import ttk
from random import choice, seed, randint
from math import sqrt, log
from typing import List, Tuple, Optional, Dict, Any

# ===========================================================================================
# CONFIGURACIONES GLOBALES
# ===========================================================================================

ruta_donde_estan_las_imagenes: str = "Iconos/"
coef_balance: float = 1.414 

# ===========================================================================================
# MCTS: Monte Carlo Tree Search (SISTEMA NEGAMAX)
# ===========================================================================================

def check_winner(state: List[List[str]]) -> Optional[str]:
    """
    Evalúa el estado actual del tablero para determinar si existe un ganador.

    Args:
        state (List[List[str]]): Matriz de 3x3 que representa el tablero de juego.

    Returns:
        Optional[str]: 'x' si gana el jugador, 'o' si gana la IA, 'draw' si es empate,
                      o None si el juego aún sigue en curso.
    """
    for i in range(3):
        if state[i][0] == state[i][1] == state[i][2] and state[i][0] != "-":
            return state[i][0]
        if state[0][i] == state[1][i] == state[2][i] and state[0][i] != "-":
            return state[0][i]
    if state[0][0] == state[1][1] == state[2][2] and state[0][0] != "-":
        return state[0][0]
    if state[0][2] == state[1][1] == state[2][0] and state[0][2] != "-":
        return state[0][2]
    if not any("-" in row for row in state):
        return "draw"
    return None


class MCTSNode(Node):
    """
    Representa un nodo dentro del árbol de búsqueda de Montecarlo (MCTS).
    Hereda de anytree.Node para la gestión estructural del árbol.
    """
    def __init__(self, name: str, parent: Optional['MCTSNode'] = None, state: Optional[List[List[str]]] = None, **kwargs: Any) -> None:
        """
        Inicializa un nuevo nodo del árbol MCTS.

        Args:
            name (str): Identificador visual del nodo (coordenadas del movimiento).
            parent (Optional[MCTSNode]): Nodo padre en el árbol.
            state (Optional[List[List[str]]]): Estado del tablero en este nodo.
        """
        super().__init__(name, parent, **kwargs)
        self.state: List[List[str]] = state if state is not None else [["-" for _ in range(3)] for _ in range(3)]
        self.visits: int = 0
        self.value: float = 0.0
        
        # Determinación automática del turno basado en el conteo de piezas
        num_x: int = sum(row.count("x") for row in self.state)
        num_o: int = sum(row.count("o") for row in self.state)
        self.player_to_move: str = "o" if num_o < num_x else "x"
        self.player_just_moved: str = "x" if self.player_to_move == "o" else "o"

    @property
    def uct(self) -> float:
        """
        Calcula el valor UCT (Upper Confidence Bound para Árboles) del nodo.

        Returns:
            float: Valor de prioridad para la selección. Retorna infinito si no ha sido visitado.
        """
        if self.visits == 0:
            return float('inf')
        parent_visits: int = self.parent.visits if self.parent is not None else 1
        return (self.value / self.visits) + coef_balance * sqrt(log(parent_visits) / self.visits)


def selection(actual: MCTSNode) -> MCTSNode:
    """
    Fase de Selección de MCTS. Desciende por el árbol utilizando la métrica UCT 
    hasta encontrar un nodo que no esté completamente expandido o sea terminal.

    Args:
        actual (MCTSNode): Nodo raíz o nodo actual de la evaluación.

    Returns:
        MCTSNode: Nodo seleccionado para la expansión o simulación.
    """
    while check_winner(actual.state) is None:
        empty_spots: int = sum(row.count("-") for row in actual.state)
        if len(actual.children) < empty_spots:
            return actual
        actual = max(actual.children, key=lambda c: c.uct)
    return actual


def expand(actual: MCTSNode) -> MCTSNode:
    """
    Fase de Expansión de MCTS. Añade un nuevo nodo hijo correspondiente a un
    movimiento legal no explorado previamente desde el nodo actual.

    Args:
        actual (MCTSNode): Nodo a expandir.

    Returns:
        MCTSNode: El nuevo nodo hijo creado.
    """
    empty_moves: List[Tuple[int, int]] = [(i, j) for i in range(3) for j in range(3) if actual.state[i][j] == "-"]
    
    existing_moves: List[Tuple[int, int]] = []
    for child in actual.children:
        for i in range(3):
            for j in range(3):
                if child.state[i][j] != actual.state[i][j]:
                    existing_moves.append((i, j))
                    
    x, y = 0, 0
    for move in empty_moves:
        if move not in existing_moves:
            x, y = move
            break

    new_state: List[List[str]] = [row[:] for row in actual.state]
    new_state[x][y] = actual.player_to_move
    
    return MCTSNode(f"M({x},{y})", parent=actual, state=new_state)


def simulate(nodo: MCTSNode) -> float:
    """
    Fase de Simulación (Rollout) de MCTS. Realiza una partida aleatoria (Playout)
    desde el estado del nodo hasta alcanzar un estado terminal.

    Args:
        nodo (MCTSNode): Nodo desde el cual arranca la simulación simulada.

    Returns:
        float: 1.0 si gana la máquina ('o'), -1.0 si gana el jugador ('x'), 0.0 si es empate.
    """
    state: List[List[str]] = [row[:] for row in nodo.state]
    player: str = nodo.player_to_move
    
    while True:
        winner: Optional[str] = check_winner(state)
        if winner == 'o': return 1.0   
        if winner == 'x': return -1.0  
        if winner == 'draw': return 0.0 
        
        empty: List[Tuple[int, int]] = [(i, j) for i in range(3) for j in range(3) if state[i][j] == "-"]
        i, j = choice(empty)
        state[i][j] = player
        player = "x" if player == "o" else "o"


def backpropagation(actual: Optional[MCTSNode], result: float) -> None:
    """
    Fase de Retropropagación de MCTS. Sube por el árbol actualizando los valores
    de visitas y recompensas acumuladas desde el nodo simulación hasta la raíz.

    Args:
        actual (Optional[MCTSNode]): Nodo inicial de retropropagación.
        result (float): Resultado obtenido en la simulación.
    """
    while actual is not None:
        actual.visits += 1
        if actual.player_just_moved == "o":
            actual.value += result
        else:
            actual.value -= result
        actual = actual.parent


def MONTECARLO(S0_state: List[List[str]], num_moves: int) -> Tuple[MCTSNode, MCTSNode]:
    """
    Ejecuta el ciclo completo del algoritmo de Búsqueda de Árboles de Montecarlo.

    Args:
        S0_state (List[List[str]]): Estado inicial del tablero.
        num_moves (int): Número de iteraciones/simulaciones a ejecutar.

    Returns:
        Tuple[MCTSNode, MCTSNode]: Una tupla conteniendo (Mejor Nodo Elegido, Nodo Raíz del Árbol).
    """
    root = MCTSNode("Raíz", parent=None, state=S0_state)
    
    for _ in range(num_moves):
        leaf: MCTSNode = selection(root)
        if check_winner(leaf.state) is None and sum(row.count("-") for row in leaf.state) > 0:
            leaf = expand(leaf)
        result: float = simulate(leaf)
        backpropagation(leaf, result)
        
    best_node: MCTSNode = max(root.children, key=lambda c: c.visits)
    return best_node, root

# ===========================================================================================
# MANEJO SEGURO DE RECURSOS GRÁFICOS (FALLBACK SYSTEM)
# ===========================================================================================

def get_safe_image(path: str, fallback_text: str = "", color: str = "black") -> Dict[str, Any]:
    """
    Carga una imagen GIF/PNG de manera segura. Si el archivo no existe, genera
    un diccionario con metadatos de texto alternativo para evitar interrupciones en la App.

    Args:
        path (str): Ruta del archivo de imagen.
        fallback_text (str): Texto a mostrar si la imagen falla.
        color (str): Color del texto alternativo.

    Returns:
        Dict[str, Any]: Diccionario con el tipo de recurso ('img' o 'txt') y sus propiedades.
    """
    try:
        return {"type": "img", "img": tk.PhotoImage(file=path)}
    except tk.TclError:
        return {"type": "txt", "text": fallback_text, "color": color}


def apply_graphic(widget: tk.Label, graphic_dict: Dict[str, Any]) -> None:
    """
    Aplica de forma polimórfica un recurso multimedia (imagen o texto de respaldo) 
    sobre un widget Label de Tkinter.

    Args:
        widget (tk.Label): Widget contenedor sobre el cual aplicar el cambio visual.
        graphic_dict (Dict[str, Any]): Estructura generada por get_safe_image.
    """
    if graphic_dict["type"] == "img":
        widget.config(image=graphic_dict["img"], text="", bg="#ffffff")
    else:
        widget.config(image="", text=graphic_dict["text"], fg=graphic_dict["color"], 
                      font=("Arial", 30, "bold"), bg="#ffffff")

# ===========================================================================================
# CLASE PRINCIPAL: INTERFAZ GRÁFICA Y CONTROLADOR DE PARTIDA
# ===========================================================================================

class X0:
    """
    Clase controladora del juego Tic Tac Toe. Gestiona el ciclo de vida de la interfaz,
    los eventos del usuario y los hilos de juego de la Inteligencia Artificial.
    """
    def __init__(self) -> None:
        """Inicializa la ventana de Tkinter, variables de control e inicia el loop del sistema."""
        self.root: tk.Tk = tk.Tk()
        self.root.title("Tic Tac Toe - MonteCarlo AI")
        self.root.geometry("1100x700")
        self.root.config(bg="#d3d3d3")
        
        # Inicialización de componentes gráficos principales
        self.define_elements()
        self.set_elements()
        self.comment(time_seconds=10) 
        self.root.bind("<Alt-Return>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.end_fullscreen)
        
        self.restart() 
        self.root.mainloop()

    def define_elements(self) -> None:
        """Instancia todos los widgets, variables Reactivas de Tkinter y recursos gráficos."""
        # Marcos organizadores principales (Layout)
        self.game_options: tk.Frame = tk.Frame(self.root, bg="#ffffff")
        self.game_space: tk.Frame = tk.Frame(self.root, bg="#000000")
        self.icons_space: tk.Frame = tk.Frame(self.root, bg="#ffffff")
        self.feedback: tk.Label = tk.Label(self.root, bg="#ffffff", text="¡Que la estadística te acompañe!", justify="center", font=("Orbitron", 15))
        
        # Construcción del Tablero de Celdas (Inyección de propiedades dinámicas controladas)
        self.celdas: List[Any] = [tk.Frame(self.game_space, bg="#ffffff") for _ in range(9)]
        for i, cell in enumerate(self.celdas):
            cell.value = "-"
            cell.active = False
            cell.btn = tk.Label(cell, bg="#ffffff")
            cell.btn.place(relwidth=1, relheight=1, relx=0, rely=0)
            cell.btn.bind("<Button-1>", lambda event, index=i: self.select_cell(index))
        
        # Variables de Control de Estado de Interfaz
        self.game_mode: tk.StringVar = tk.StringVar(value="Montecarlo")
        self.previous_game_mode: str = "Montecarlo"
        self.num_iter_int: tk.IntVar = tk.IntVar(value=1000)
        self.show_tree_var: tk.BooleanVar = tk.BooleanVar(value=True) 
        
        # Elementos del Panel Lateral de Opciones
        self.tictactoe_label: tk.Label = tk.Label(self.game_options, text="Tic Tac Toe\nMontecarlo", justify="center", bg="#ffffff", font=("Arial", 25, "bold"))
        self.Modo_juego_label: tk.Label = tk.Label(self.game_options, text="Modo de juego", bg="#ffffff", font=("Arial", 16, "bold"))
        
        self.random_move_button: tk.Radiobutton = tk.Radiobutton(self.game_options, text="Aleatorio", variable=self.game_mode, value="Random", bg="#ffffff", font=("Arial", 12), command=self.comando)
        self.montecarlo_move_button: tk.Radiobutton = tk.Radiobutton(self.game_options, text="Montecarlo", variable=self.game_mode, value="Montecarlo", bg="#ffffff", font=("Arial", 12), command=self.comando)
        self.two_players_button: tk.Radiobutton = tk.Radiobutton(self.game_options, text="Dos jugadores", variable=self.game_mode, value="2Players", bg="#ffffff", font=("Arial", 12), command=self.comando)
        
        self.numero_iteraciones_label: tk.Label = tk.Label(self.game_options, text="Iteraciones de MCTS", bg="#ffffff", font=("Arial", 12, "bold"))
        self.num_iter_scale: tk.Scale = tk.Scale(self.game_options, from_=100, to=7500, orient="horizontal", bg="#ffffff", tickinterval=1000, resolution=100, variable=self.num_iter_int)
        
        self.show_tree_check: tk.Checkbutton = tk.Checkbutton(self.game_options, text="Ver árbol MCTS tras jugar", variable=self.show_tree_var, bg="#ffffff", font=("Arial", 11))
        self.restart_button: tk.Button = tk.Button(self.game_options, text="Reinicio", command=self.restart, font=("Arial", 14, "bold"), bg="#4CAF50", fg="white")

        # Carga de Colecciones de Sprites Básicas
        self.gfx_x: Dict[str, Any] = get_safe_image(ruta_donde_estan_las_imagenes + "X_pro.png", fallback_text="X", color="#1E90FF")
        self.gfx_o_random: Dict[str, Any] = get_safe_image(ruta_donde_estan_las_imagenes + "zero_random.png", fallback_text="O", color="#32CD32")
        self.gfx_o_mcts: Dict[str, Any] = get_safe_image(ruta_donde_estan_las_imagenes + "zero_montecarlo.png", fallback_text="O", color="#FF4500")
        self.gfx_o: Dict[str, Any] = get_safe_image(ruta_donde_estan_las_imagenes + "zero.png", fallback_text="O", color="#FF8C00")
        
        # Carga de Estados de Iconos Reactivos Superiores
        self.machine_playing_icon: Dict[str, Any] = get_safe_image(ruta_donde_estan_las_imagenes + "machine_playing.png", fallback_text="🤖", color="black")
        self.machine_win_icon: List[Dict[str, Any]] = [get_safe_image(ruta_donde_estan_las_imagenes + f"machine_win{i}.png", fallback_text=f"🤖 W{i}") for i in range(1, 5)]
        self.machine_lose_icon: List[Dict[str, Any]] = [get_safe_image(ruta_donde_estan_las_imagenes + f"machine_lose{i}.png", fallback_text=f"🤖 L{i}") for i in range(1, 3)]
        self.machine_draw_icon: Dict[str, Any] = get_safe_image(ruta_donde_estan_las_imagenes + "machine_draw.png", fallback_text="🤖 D")
        
        self.player_playing_icon: Dict[str, Any] = get_safe_image(ruta_donde_estan_las_imagenes + "player_playing.png", fallback_text="🧑", color="black")
        self.player_win_icon: List[Dict[str, Any]] = [get_safe_image(ruta_donde_estan_las_imagenes + f"player_win{i}.png", fallback_text=f"🧑 W{i}") for i in range(1, 5)]
        self.player_lose_icon: List[Dict[str, Any]] = [get_safe_image(ruta_donde_estan_las_imagenes + f"player_lose{i}.png", fallback_text=f"🧑 L{i}") for i in range(1, 5)]
        self.player_draw_icon: Dict[str, Any] = get_safe_image(ruta_donde_estan_las_imagenes + "player_draw.png", fallback_text="🧑 D")
        
        self.result_playing_icon: List[Dict[str, Any]] = [get_safe_image(ruta_donde_estan_las_imagenes + f"result_playing{i}.png", fallback_text="VS") for i in range(1, 11)]
        self.result_machine_icon: Dict[str, Any] = get_safe_image(ruta_donde_estan_las_imagenes + "result_machine.png", fallback_text="🏆 O")
        self.result_player_icon: Dict[str, Any] = get_safe_image(ruta_donde_estan_las_imagenes + "result_player.png", fallback_text="🏆 X")
        self.result_draw_icon: Dict[str, Any] = get_safe_image(ruta_donde_estan_las_imagenes + "result_draw.png", fallback_text="⚖️")

        # Asignación de Contenedores de Iconos Superiores
        self.machine_icon: tk.Label = tk.Label(self.icons_space, bg="#ffffff")
        self.result_icon: tk.Label = tk.Label(self.icons_space, bg="#ffffff")
        self.player1_icon: tk.Label = tk.Label(self.icons_space, bg="#ffffff")

        apply_graphic(self.machine_icon, self.machine_playing_icon)
        apply_graphic(self.player1_icon, self.player_playing_icon)
        apply_graphic(self.result_icon, choice(self.result_playing_icon))

        # Flags lógicas de flujo de control de turnos
        self.player1_turn: bool = True
        self.machine_turn: bool = False
        self.fullscreen: bool = False

    def set_elements(self) -> None:
        """Organiza espacialmente (Renderizado y Posicionamiento) los widgets en la Grid/Canvas."""
        self.game_options.place(relwidth=0.3, relheight=1, relx=0, rely=0)
        self.icons_space.place(relwidth=0.7, relheight=0.15, relx=0.3, rely=0)
        self.feedback.place(relwidth=0.7, relheight=0.1, relx=0.3, rely=0.15)
        self.game_space.place(relwidth=0.7, relheight=0.75, relx=0.3, rely=0.25)

        self.tictactoe_label.place(relwidth=1, relheight=0.15, relx=0, rely=0.02)
        self.Modo_juego_label.place(relwidth=1, relheight=0.05, relx=0, rely=0.2)
        self.random_move_button.place(relwidth=0.8, relheight=0.05, relx=0.1, rely=0.28)
        self.montecarlo_move_button.place(relwidth=0.8, relheight=0.05, relx=0.1, rely=0.35)
        self.two_players_button.place(relwidth=0.8, relheight=0.05, relx=0.1, rely=0.42)
        
        self.numero_iteraciones_label.place(relwidth=1, relheight=0.05, relx=0, rely=0.55)
        self.num_iter_scale.place(relwidth=0.8, relheight=0.1, relx=0.1, rely=0.62)
        self.show_tree_check.place(relwidth=0.9, relheight=0.05, relx=0.05, rely=0.75)
        self.restart_button.place(relwidth=0.6, relheight=0.08, relx=0.2, rely=0.85)

        self.machine_icon.place(relwidth=0.3, relheight=1, relx=0.05, rely=0)
        self.result_icon.place(relwidth=0.3, relheight=1, relx=0.35, rely=0)
        self.player1_icon.place(relwidth=0.3, relheight=1, relx=0.65, rely=0)

        # Enlace de eventos interactivos para alternar los iconos superiores mediante clicks
        self.result_icon.bind("<Button-1>", lambda event: self.set_image(Player=None))
        self.machine_icon.bind("<Button-1>", lambda event: self.set_image(Player=False))
        self.player1_icon.bind("<Button-1>", lambda event: self.set_image(Player=True))

        for i in range(9):
            x_pos, y_pos = (i % 3) * 0.334, (i // 3) * 0.334
            self.celdas[i].place(relwidth=0.33, relheight=0.33, relx=x_pos, rely=y_pos)

    def get_state(self) -> List[List[str]]:
        """
        Serializa el estado visual actual del tablero de Tkinter a una estructura nativa de matriz.

        Returns:
            List[List[str]]: Representación bidimensional del tablero.
        """
        return [[self.celdas[i * 3 + j].value for j in range(3)] for i in range(3)]

    def select_cell(self, index: int) -> None:
        """
        Disparador principal ante la interacción del usuario con una celda del tablero.
        Controla los turnos y la respuesta lógica del adversario (Humano o Máquina).

        Args:
            index (int): Índice plano (0-8) de la celda pulsada.
        """
        if self.game_is_over():
            self.feedback.config(text="Reinicia el juego...", justify="center")
            return
            
        if self.celdas[index].active:
            return

        game_mode: str = self.game_mode.get()

        if game_mode in ["Random", "Montecarlo"]:
            if self.player1_turn:
                self.make_move(index, "x", self.gfx_x)
                if not self.game_is_over():
                    self.root.update()
                    self.machine_turn = True
                    self.player1_turn = False
                    
                    if game_mode == "Random":
                        self.random_move(index)
                    else:
                        self.montecarlo_move(index)
        else:
            # Modo Dos Jugadores locales alternos
            if self.player1_turn:
                self.make_move(index, "x", self.gfx_x)
            else:
                self.make_move(index, "o", self.gfx_o)
            self.player1_turn = not self.player1_turn
            self.game_is_over()

    def make_move(self, index: int, value: str, gfx: Dict[str, Any]) -> None:
        """
        Dibuja físicamente una pieza y bloquea lógicamente la celda especificada.

        Args:
            index (int): Celda de destino.
            value (str): Marcador de estado ('x' u 'o').
            gfx (Dict[str, Any]): Recurso visual a aplicar en la celda.
        """
        self.celdas[index].value = value
        self.celdas[index].active = True
        apply_graphic(self.celdas[index].btn, gfx)
        
    def random_move(self, index1: int = -1) -> None:
        """
        Ejecuta la respuesta lógica de la IA en modo Aleatorio.

        Args:
            index1 (int): Último movimiento realizado por el jugador para construir el log de comentarios.
        """
        celdas_libres: List[int] = [i for i in range(9) if not self.celdas[i].active]
        if celdas_libres:
            index2: int = choice(celdas_libres)
            self.make_move(index2, "o", self.gfx_o_random)
            self.comment(called_by="Random", index1=index1, index2=index2)
            self.machine_turn = False
            self.player1_turn = True
            self.game_is_over()

    def montecarlo_move(self, index1: int = -1) -> None:
        """
        Ejecuta el pipeline de procesamiento MCTS para encontrar el movimiento óptimo.

        Args:
            index1 (int): Último movimiento del jugador humano.
        """
        state: List[List[str]] = self.get_state()
        move_index: int = 0
        
        # Ejecución del motor de toma de decisiones
        best_node, root_node = MONTECARLO(state, self.num_iter_int.get())
        
        for i in range(3):
            for j in range(3):
                if state[i][j] != best_node.state[i][j]:
                    move_index = i * 3 + j
                    break

        if self.show_tree_var.get():
            self.visualize_tree(root_node, best_node)
        
        self.make_move(move_index, "o", self.gfx_o_mcts)
        
        # Heurística para deducir si un movimiento fue crítico/defensivo
        obligated_bool: bool = (best_node.visits > self.num_iter_int.get() * 0.8)
        self.comment(called_by="Montecarlo", index1=index1, index2=move_index, obligated=obligated_bool)
        
        self.machine_turn = False
        self.player1_turn = True
        self.game_is_over()

    def visualize_tree(self, root_node: MCTSNode, best_node: MCTSNode) -> None:
        """
        Despliega de forma jerárquica las estadísticas y ramas analizadas por MCTS
        en una nueva ventana secundaria (TreeView).

        Args:
            root_node (MCTSNode): Nodo raíz evaluado.
            best_node (MCTSNode): Nodo dictaminado como la mejor jugada encontrada.
        """
        top: tk.Toplevel = tk.Toplevel(self.root)
        top.title("Explorador de Árbol MCTS")
        top.geometry("750x400")
        
        columns: Tuple[str, ...] = ("win_rate", "visits", "value", "uct")
        tree: ttk.Treeview = ttk.Treeview(top, columns=columns, selectmode="none")
        tree.heading("#0", text="Nodo / Movimiento")
        tree.heading("win_rate", text="Tasa Éxito O")
        tree.heading("visits", text="Visitas")
        tree.heading("value", text="Valor Acum.")
        tree.heading("uct", text="UCT Máximo")

        tree.column("#0", width=180, anchor="w")
        tree.column("win_rate", width=100, anchor="center")
        tree.column("visits", width=80, anchor="center")
        tree.column("value", width=100, anchor="center")
        tree.column("uct", width=120, anchor="center")
        
        vsb: ttk.Scrollbar = ttk.Scrollbar(top, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        def add_node_to_tree(parent_id: str, node: MCTSNode, is_root: bool = False) -> None:
            if node.visits > 0:
                val_relativo: float = node.value if node.player_just_moved == 'o' else -node.value
                win_rate_str: str = f"{(val_relativo / node.visits * 100):.1f}%"
            else:
                win_rate_str = "0%"
                
            uct_str: str = f"{node.uct:.3f}" if node.uct != float('inf') else "INF"
            tag: str = "best" if node == best_node else ""
            
            node_id: str = tree.insert(parent_id, "end", text=node.name, values=(win_rate_str, node.visits, node.value, uct_str), tags=(tag,))
            
            children: List[MCTSNode] = sorted([c for c in node.children if c.visits > 0], key=lambda c: c.visits, reverse=True)
            for child in children:
                add_node_to_tree(node_id, child)
                
            if is_root: 
                tree.item(node_id, open=True)

        tree.tag_configure("best", background="#90EE90") 
        add_node_to_tree("", root_node, is_root=True)

    def set_image(self, Player: Optional[bool] = None, called_by: Optional[str] = None, winner: Optional[int] = None) -> None:
        """
        Controla los gráficos interactivos de los avatares e indicadores superiores,
        reaccionando dinámicamente tanto a clicks manuales como al final de partida.

        Args:
            Player (Optional[bool]): True si interactúa jugador1, False máquina, None resultado.
            called_by (Optional[str]): Canalizador de contexto ("game_is_over" o None).
            winner (Optional[int]): ID de estado final (1 Ganador, -1 Perdedor, 0 Empate).
        """
        seed()

        if called_by is None:
            robot_images: List[Dict[str, Any]] = [self.machine_draw_icon, self.machine_playing_icon]
            robot_images.extend(self.machine_lose_icon)
            robot_images.extend(self.machine_win_icon)
            index_machine: int = randint(0, len(robot_images) - 1)

            player_images: List[Dict[str, Any]] = [self.player_playing_icon, self.player_draw_icon]
            player_images.extend(self.player_win_icon)
            player_images.extend(self.player_lose_icon)
            index_player: int = randint(0, len(player_images) - 1)

            if Player is None:
                apply_graphic(self.result_icon, choice(self.result_playing_icon))
            else:
                if Player:
                    apply_graphic(self.player1_icon, player_images[index_player])
                else:
                    apply_graphic(self.machine_icon, robot_images[index_machine])
            return
            
        if called_by == "game_is_over":
            if winner == 1:
                apply_graphic(self.player1_icon, choice(self.player_win_icon))
                apply_graphic(self.machine_icon, choice(self.machine_lose_icon))
                apply_graphic(self.result_icon, self.result_player_icon)
            elif winner == -1:
                apply_graphic(self.machine_icon, choice(self.machine_win_icon))
                apply_graphic(self.player1_icon, choice(self.player_lose_icon))
                apply_graphic(self.result_icon, self.result_machine_icon)
            else:
                apply_graphic(self.result_icon, self.result_draw_icon)
                apply_graphic(self.machine_icon, self.machine_draw_icon)
                apply_graphic(self.player1_icon, self.player_draw_icon)

    def comment(self, called_by: Optional[str] = None, time_seconds: int = 2, texto1: Optional[str] = None, 
                texto2: Optional[str] = None, winner: Optional[str] = None, index1: Optional[int] = None, 
                index2: Optional[int] = None, obligated: Optional[bool] = None) -> None:
        """
        Generador dinámico de logs de juego y barra informativa con datos teóricos curiosos.

        Args:
            called_by (Optional[str]): Origen del evento desencadenante.
            time_seconds (int): Delay en segundos para el bucle recursivo de datos.
            texto1 (Optional[str]): Modo previo para el callback de comando.
            texto2 (Optional[str]): Nuevo modo seleccionado.
            winner (Optional[str]): Entidad vencedora ("Player", "Machine", o "Draw").
            index1 (int): Casilla de origen del jugador.
            index2 (int): Casilla respuesta de la IA.
            obligated (Optional[bool]): True si MCTS determinó defensa mandatoria.
        """
        random_comments: List[str] = [
            "¿Sabías que hay 255,168 posibles partidas\n en el tic-tac-toe?",
            "En el tic-tac-toe, el jugador que empieza siempre\n tiene una ventaja si juega correctamente.",
            "El tic-tac-toe es uno de los primeros juegos en los que muchos\n niños aprenden a jugar.",
            "¿Has oído hablar de la estrategia ganadora en el tic-tac-toe\n llamada 'centro, esquinas, lados'?",
            "A pesar de su aparente simplicidad, el tic-tac-toe es un juego\n que implica estrategia y táctica.",
            "¿Sabías que el juego tic-tac-toe se remonta a la época\n del Antiguo Egipto?",
            "Sabías que el tic-tac-toe también se conoce por otros\n nombres como 'gato', 'X0', 'Michi', 'Tres en línea'.",
            "El tic-tac-toe es un juego de suma cero, lo que significa\n que el resultado es siempre un empate o una victoria para uno de los jugadores.",
            "¿Has oído hablar de la variante 3D del tic-tac-toe, donde\n se juega en un cubo, en lugar de en un tablero plano?",
            "En el tic-tac-toe, si ambos jugadores juegan perfectamente,\n el juego terminará siempre en empate.",
            "¿Sabías que si olvidamos las reglas del juego, el tablero\n puede ser rellenado de 512 formas distintas?",
            "¿Sabías que si ignoramos las condiciones de victoria,\n hay 362,880 formas distintas de llenar el tablero?",
            "Sabías que si ambos jugadores juegan al azar, la probabilidad de\n ganar en el tic-tac-toe es de aproximadamente\n 0.58 para el jugador que empieza y 0.42 para el otro jugador?"
        ]
        seed()
        index: int = randint(0, len(random_comments) - 1)
    
        if called_by is None:
            # Implementación de loop asíncrono no bloqueante mediante Tcl/Tk clock
            self.feedback.after(1000 * time_seconds, lambda: self.feedback.config(text=random_comments[index], justify="center"))
            self.feedback.after(1000 * time_seconds, lambda: self.comment(time_seconds=10))
        
        elif called_by == "comando":
            if texto1 == texto2:
                self.feedback.config(text="Buen intento, pero es el mismo modo de juego", justify="center")
                return
            
            t1: str = "Aleatorio" if texto1 == "Random" else ("Dos jugadores" if texto1 == "2Players" else str(texto1))
            t2: str = "Aleatorio" if texto2 == "Random" else ("Dos jugadores" if texto2 == "2Players" else str(texto2))
            
            self.feedback.config(text=f"Modo de juego: {t1} -> {t2} ", justify="center")
       
        elif called_by == "game_is_over":
            if winner == "Player":
                msg = "El ganador es: JUGADOR 1" if self.game_mode.get() == "2Players" else "El ganador es: LA HUMANIDAD"
            elif winner == "Machine":
                if self.game_mode.get() == "2Players": msg = "El ganador es: JUGADOR 2"
                elif self.game_mode.get() == "Montecarlo": msg = "El ganador es: El RACIOCINIO ARTIFICIAL"
                else: msg = "El ganador es: LA ALEATORIEDAD"
            else:
                msg = "Fue una partida reñida... Supongo"
            self.feedback.config(text=msg, justify="center")
                
        elif called_by == "restart":
            self.feedback.config(text="Juego Reiniciado", justify="center")

        elif called_by == "Random":
            if not self.game_is_over() and index1 is not None and index2 is not None:
                value: int = randint(1, 1000)
                if value % 5 == 0:
                    self.feedback.config(text=f"El jugador selecciona la casilla {index1+1},\n la aleatoriedad opta por la casilla {index2+1}", justify="center")
                    self.comment(time_seconds=3)
                elif value % 3 == 0:
                    self.feedback.config(text=f"El jugador racionalmente escoje la casilla {index1+1},\n el universo se decanta por la casilla {index2+1}... nomas porque sí", justify="center")
                    self.comment(time_seconds=3)
                    
        elif called_by == "Montecarlo":
            if not self.game_is_over() and index1 is not None and index2 is not None:
                if index1 == -1: return
                value = randint(1, 1000)
                if obligated:
                    if value % 5 == 0:
                        self.feedback.config(text=f"El jugador intenta ganar con casilla {index1+1},\n la máquina contrarresta con casilla {index2+1}", justify="center")
                    elif value % 3 == 0:
                        self.feedback.config(text=f"El jugador elegantemente escoje la casilla {index1+1},\n la máquina se ve obligada a escoger la casilla {index2+1}", justify="center")
                else:
                    if value % 5 == 0:
                        self.feedback.config(text=f"Después de {self.num_iter_int.get()} simulaciones (no siempre perfectas),\n la máquina decide jugar casilla {index2+1}", justify="center")
                    elif value % 3 == 0:
                        self.feedback.config(text=f"Puede que detrás de la elección de la casilla {index2+1}, en la mayoría de \nlas {self.num_iter_int.get()} simulaciones, la máquina sea la que termina ganando...", justify="center")        

    def game_is_over(self) -> bool:
        """
        Analiza las reglas del juego para finalizar la partida y coordinar las animaciones gráficas.

        Returns:
            bool: True si la partida culminó por victoria o tablas, False en caso contrario.
        """
        state: List[List[str]] = self.get_state()
        winner: Optional[str] = check_winner(state)
        
        if winner == "x":
            self.comment(called_by="game_is_over", winner="Player")
            self.set_image(called_by="game_is_over", winner=1)
            return True
        elif winner == "o":
            self.comment(called_by="game_is_over", winner="Machine")
            self.set_image(called_by="game_is_over", winner=-1)
            return True
        elif winner == "draw":
            self.comment(called_by="game_is_over", winner="Draw")
            self.set_image(called_by="game_is_over", winner=0)
            return True
        return False

    def comando(self) -> None:
        """Sincroniza y valida las variaciones en los botones de selección del Modo de juego."""
        game_mode: str = self.game_mode.get()
        self.comment(called_by="comando", texto1=self.previous_game_mode, texto2=game_mode)
        
        if game_mode in ["Random", "Montecarlo"]:
            if not self.player1_turn: 
                if game_mode == "Random": self.random_move()
                else: self.montecarlo_move()
                
        self.previous_game_mode = game_mode

    def restart(self) -> None:
        """Limpia el estado interno de las variables de juego y reinicia los canvas visuales."""
        self.comment(called_by="restart")
        for celda in self.celdas:
            celda.value = "-"
            celda.active = False
            celda.btn.config(image="", text="", bg="#ffffff")
            
        self.player1_turn = True
        self.machine_turn = False
        apply_graphic(self.machine_icon, self.machine_playing_icon)
        apply_graphic(self.player1_icon, self.player_playing_icon)
        apply_graphic(self.result_icon, choice(self.result_playing_icon))
        
    def toggle_fullscreen(self, event: Optional[tk.Event] = None) -> None:
        """Activa el modo de visualización de pantalla completa."""
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def end_fullscreen(self, event: Optional[tk.Event] = None) -> None:
        """Desactiva el modo de visualización de pantalla completa."""
        self.fullscreen = False
        self.root.attributes("-fullscreen", self.fullscreen)


if __name__ == "__main__":
    app = X0()