import numpy as np
import perceptron as p

#TEST PARA XOR
x_original = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
y_salida = np.array([[1], [1], [-1], [-1]])

print(y_salida.shape)
print(y_salida)

p = y_salida.shape[0]
bias = np.ones((p, 1))
X_con_bias = np.concatenate((x_original, bias), axis=1)


w_oculta_fin, W_salida_fin, error_final, iteraciones = p.entrenamiento_pmc(X_con_bias, y_salida, etha=0.1, COTA=5000, neuronas_ocultas=2, neuronas_salida=1, f_act="tanh")
print(f"Vector de Pesos: {W_salida_fin} | Error Promedio: {error_final} |  Iteraciones: {iteraciones}")

print('Test de clasificación XOR:')

p.test_pmc(X_con_bias, y_salida, w_oculta_fin, W_salida_fin, f_act='tanh')

datos = np.loadtxt('digitos_tp.txt')
x_original = datos.reshape(10, 35)
y_salida = np.array([[1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1]])

#TEST PARA NÚMEROS PARES

#Entrenamiento con dataset completo
p = len(y_salida)
bias = np.ones((p, 1))
X_con_bias = np.concatenate((x_original, bias), axis=1)

#32 entradas, 16 perceptrones en capa oculta, 1 perceptron de salida
w_oculta_fin, W_salida_fin, error_final, iteraciones = p.entrenamiento_pmc(X_con_bias, y_salida, etha=0.1, COTA=5000, neuronas_ocultas=16, neuronas_salida=1, f_act="tanh")
print(f"Vector de Pesos: {W_salida_fin} | Error Promedio: {error_final} |  Iteraciones: {iteraciones}")

p.test_pmc(X_con_bias, y_salida, w_oculta_fin, W_salida_fin, f_act='tanh')

#Entrenamiento con subconjunto
corte = 8

x_train = x_original[:corte]
y_train = y_salida[:corte]

x_test = x_original[corte:]
y_test = y_salida[corte:]

bias_train = np.ones((len(y_train), 1))
X_train = np.concatenate((bias_train, x_train), axis=1)

bias_test = np.ones((len(y_test), 1))
X_test = np.concatenate((bias_test, x_test), axis=1)

w_oculta_fin, W_salida_fin, error_final, iteraciones = p.entrenamiento_pmc(X_train, y_train, etha=0.1, COTA=5000, neuronas_ocultas=16, neuronas_salida=1, f_act="tanh")
print(f"Vector de Pesos: {W_salida_fin} | Error Promedio: {error_final} |  Iteraciones: {iteraciones}")

p.test_pmc(X_test, y_test, w_oculta_fin, W_salida_fin, f_act='tanh')

#TEST PARA NÚMEROS INDIVIDUALES

#PUNTO 3

y_nuevo = np.identity(10)

#En caso de probar con tanh:
# for i in range(len(y_nuevo)):
#   for j in range(len(y_nuevo)):
#     if y_nuevo[i, j] == 0:
#       y_nuevo[i, j] = -1

print(y_nuevo)

x_train = x_original[:corte]
y_train = y_nuevo[:corte]

x_test = x_original[corte:]
y_test = y_nuevo[corte:]

bias_train = np.ones((len(y_train), 1))
X_train = np.concatenate((bias_train, x_train), axis=1)

bias_test = np.ones((len(y_test), 1))
X_test = np.concatenate((bias_test, x_test), axis=1)

w_oculta_fin, W_salida_fin, error_final, iteraciones = p.entrenamiento_pmc(X_con_bias, y_nuevo, etha=0.1, COTA=5000, neuronas_ocultas=32, neuronas_salida=10, f_act="sigmoide")
print(f"Vector de Pesos: {W_salida_fin} | Error Promedio: {error_final} |  Iteraciones: {iteraciones}")

p.test_pmc(X_con_bias, y_nuevo, w_oculta_fin, W_salida_fin, f_act='sigmoide')

#EVALUACIÓN CON RUIDO
def aplicar_ruido(X_original, prob=0.02):
    # Copiamos la matriz para no alterar los datos originales
    X_ruidoso = np.copy(X_original)
    
    # Generamos una matriz de las mismas dimensiones con valores [0, 1)
    aleatorios = np.random.rand(*X_original.shape)
    
    # Creamos una máscara booleana donde se cumple la probabilidad
    mascara = aleatorios < prob
    
    # Invertimos los bits solo donde la máscara es True
    # Si era 1, (1 - 1) = 0. Si era 0, (1 - 0) = 1.
    X_ruidoso[mascara] = 1 - X_ruidoso[mascara]
    
    return X_ruidoso

X_ruido = aplicar_ruido(x_original)
bias_ruido = np.ones((len(y_nuevo), 1))
X_ruido_bias = np.concatenate((bias_ruido, X_ruido), axis=1)

p.test_pmc(X_ruido_bias, y_nuevo, w_oculta_fin, W_salida_fin, f_act='sigmoide')

