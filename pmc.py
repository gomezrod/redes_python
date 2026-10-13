import numpy as np
import numpy.typing as npt

def funcion_escalon(x:float)->float:
    return np.where(x >= 0, 1, -1)

def sigmoid(x:float)->float:
    """Función sigmoide."""
    return 1 / (1 + np.exp(-x))

def tanh(x)->float:
    """Función Tangente Hiperbólica."""
    return np.tanh(x)

def activacion(h, tipo="escalon"):
    if tipo == "escalon":
        return np.where(h >= 0, 1, -1)
    elif tipo == "lineal":
        return h
    elif tipo == "sigmoide":
        return 1 / (1 + np.exp(-h))
    elif tipo == "tanh":
        return np.tanh(h)
    else:
      raise Exception(f'Nombre de funcion no valida: {tipo}')

def derivada(O:npt.NDArray[np.float64], tipo:str="escalon") -> float:
    if tipo == "sigmoide":
        return O * (1 - O)
    elif tipo == "tanh":
        return 1 - O**2
    else:
      return 1

def entrenamiento_pmc2(X:npt.NDArray[np.float64], Y:npt.NDArray[np.float64], etha: float, COTA: int, H1: int, H2: int, K: int, f_act: str = 'tanh')->npt.NDArray:
    p = len(Y)
    N_entradas = X.shape[1]  # Dimensiones de entrada (ya incluye el bias)

    # 1. INICIALIZACIÓN: En el PMC los pesos DEBEN empezar con valores aleatorios pequeños
    w_oculta1 = np.random.uniform(-0.5, 0.5, (N_entradas, H1))
    w_oculta2 = np.random.uniform(-0.5, 0.5, (H1 + 1, H2))
    w_salida = np.random.uniform(-0.5, 0.5, (H2 + 1, K))
    error = float('inf')
    i = 0

    while error > 0.05 and i < COTA:
        rnd = np.random.randint(0, p)
        x_i = X[rnd:rnd+1]  # Lo mantenemos como matriz 1xN para facilitar np.dot
        y_i = Y[rnd:rnd+1]

        # --- PROPAGACIÓN ---
        A1 = np.dot(x_i, w_oculta1) # X . W1 [p x N] . [N x H1]
        V1 = activacion(A1, f_act) #V1 = [p x H1]
        # Bias para la capa oculta 1
        V1_con_bias = np.concatenate((np.ones((1, 1)), V1), axis=1) # [p x H1+1]
        A2 = np.dot(V1_con_bias, w_oculta2) #[p x H1+1].[H1+1 x H2]
        V2 = activacion(w_oculta2, f_act) #[p x H2]
        # Bias para la capa oculta 2
        V2_con_bias = np.concatenate((np.ones((1, 1)), V2), axis=1) #[p x H2+1]
        A3 = np.dot(V2_con_bias, w_salida) #[pxH2+1] x [H2+1 x K]
        O = activacion(w_salida, f_act) # [p x K]

        # --- RETROPROPAGACIÓN ---
        # Delta de la salida
        delta3 = (y_i - O) * derivada(O, f_act) #[p x K]
        # Delta de la capa oculta 2 (recortamos la fila del bias de w_salida con [1:, :])
        w_salida_sin_bias = w_salida[1:, :] # [H2 x K]
        delta2 = derivada(V2, f_act) * np.dot(delta3, w_salida_sin_bias.T) # [p x K] . [K x H2] = [p x H2]
        # Delta de la capa oculta 1 (recortamos la fila del bias de w_oculta2 con [1:, :])
        w_oculta2_sin_bias = w_oculta2[1:,:] # [H1 x H2]
        delta1 = derivada(V1, f_act) * np.dot(delta2, w_oculta2_sin_bias.T) #[p x H2] . [H2 x H1]

        # --- ACTUALIZACIÓN DE PESOS ---
        w_salida += etha * np.dot(V2_con_bias.T, delta3)
        w_oculta2 += etha * np.dot(V1_con_bias.T, delta2) 
        w_oculta1 += etha * np.dot(x_i.T, delta1)

        # --- CÁLCULO DE ERROR TOTAL ---
        error = calcular_error_pmc2(X, Y, w_oculta1, w_oculta2, w_salida, f_act)
        i += 1

    return w_oculta1, w_oculta2, w_salida, error, i

def calcular_error_pmc2(X:npt.NDArray[np.float64], Y:npt.NDArray[np.float64], w_oculta1:npt.NDArray[np.float64], w_oculta2:npt.NDArray[np.float64], w_salida:npt.NDArray[np.float64], f_act='tanh') -> float:
    """Calcula el error pasando la información por ambas capas."""

    # 1. Viaje de Entrada -> Capa Oculta
    A1 = np.dot(X, w_oculta1)
    V1 = activacion(A1, f_act)

    # 2. Agregar el Bias (una columna de unos) a la capa oculta 1
    bias1 = np.ones((len(V1), 1))
    V1_con_bias = np.concatenate((bias1, V1), axis=1)

    # 3. Viaje de Capa Oculta 1 -> Capa Oculta 2
    A2 = np.dot(V1_con_bias, w_oculta2)
    V2 = activacion(A2, f_act)

    # 4. Agregar el Bias (una columna de unos) a la capa oculta 2
    bias2 = np.ones((len(V2), 1))
    V2_con_bias = np.concatenate((bias2, V2), axis=1)

    # 5. Viaje a la capa de salida
    A3 = np.dot(V2_con_bias, w_salida)
    O = activacion(A3, f_act)

    # 4. Error Cuadrático Medio
    # Usamos Y.reshape(-1, 1) para asegurar que Y y O tengan la misma forma (columna)
    return 0.5 * np.sum((Y - O)**2)