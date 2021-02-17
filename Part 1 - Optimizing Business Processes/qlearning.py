#Artificial intelligence applied to business and enterprise
#Part 1 - Optimizacion de los flijos de trabajo en un almacen con Q-Learning
#Import libreries
import numpy as np

#Configuracion de los parametros gamma y alpha para el algoritmo de Q-Learning
#Factor de descuento, afecta mucho
gamma = 0.75
#Ratio de aprendizaje
alpha = 0.9

#PARTE 1 - DEFINICION DEL ENTORNO

#Definicion de los estados
location_to_state = {
        'A' : 0,
        'B' : 1,
        'C' : 2,
        'D' : 3,
        'E' : 4,
        'F' : 5,
        'G' : 6,
        'H' : 7,
        'I' : 8,
        'J' : 9,
        'K' : 10,
        'L' : 11 }

#Definicion de las acciones
actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

priority = ['G', 'K', 'L', 'J', 'A', 'I', 'H', 'C', 'B', 'D', 'F', 'E']

# Definicion de las recompensas

R = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # A
              [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # B
              [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # C
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # D
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # E
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # F
              [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # G
              [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],  # H
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # I
              [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],  # J
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  # K
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]]) # L



#PARTE 2 - CONSTRUCCION DE LA SOLUCION DE IA CON Q-LEARNING

#Inicializacion de los valores Q-Learning
Q = np.array(np.zeros([12,12]))

#Implementacion de proceso Q-Learning
for i in range(1000) :
    current_state = np.random.randint(0, 12)
    playable_actions = []
    for j in range(12):
        if R[current_state, j] > 0:
            playable_actions.append(j)
    next_state = np.random.choice(playable_actions)
    TD = R[current_state, next_state] + gamma*Q[next_state, np.argmax(Q[next_state,])]- Q[current_state, next_state]
    Q[current_state, next_state] += alpha*TD 



#Tranformar de estado a ubicacion
state_to_location = {state : location for location, state in location_to_state.items()}



#PARTE 2 - CONSTRUCCION DE LA SOLUCION DE IA CON Q-LEARNING
#Crear la funcion que devuelve la solucion optima
#def es funcion
def route(starting_location, ending_location):
    R_new = np.copy(R)
    ending_state = location_to_state[ending_location]
    R_new[ending_state, ending_state] = 10000
    Q = np.array(np.zeros([12,12]))
    
    #Implementacion de proceso Q-Learning
    for i in range(1000) :
        current_state = np.random.randint(0, 12)
        playable_actions = []
        for j in range(12):
            if R_new[current_state, j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions)
        TD = R_new[current_state, next_state] + gamma*Q[next_state, np.argmax(Q[next_state,])]- Q[current_state, next_state]
        Q[current_state, next_state] += alpha*TD 
    
    
    
    
    route = [starting_location]
    next_location = starting_location
    while(next_location != ending_location) :
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route
#Parte 3 - PONER UN MODELO DE PRODUCCION
def best_route(starting_location, intermediary_location, ending_location):
    return route(starting_location, intermediary_location) + route(intermediary_location, ending_location)[1:]

#Imprimir la ruta optima
print("Ruta Eleguda:")
print(best_route('E', 'G', 'A'))
