'''
Manual de instrucciones:
Este programa busca calcular problemas de criptoaritmetica generalizados, fue diseñado para resolver el problema SEND+MORE=MONEY, pero puede resolver cualquier problema de criptoaritmetica
con el suficiente tiempo y la suficiente cantidad de generaciones, el unico problema siendo que no podemos controlar el tiempo de ejecucion, ya que el algoritmo genetico es un algoritmo probabilistico
y puede que no llegue al resultado final, o que se quede en un maximo local, ya que estamos limitando la cantidad de generaciones, por lo demas, cumple con lo que promete.

Para utilizarlo, solo es necesario correr el programa y escribir el problema de criptoaritmetica en el formato ####+####=#####, donde los gatos solo significan la posicion de las letras, favor de no poner espacios
Tambien se pueden alterar los parametros de ga, dado que 1000 generaciones tal vez sea muy alto para algunas maquinas de bajo rendimiento, es bastante rapido, pero no quita el que una computadora normal pueda pasar
un mal rato.
'''


# future se utiliza para evitar problemas de interprete de python
from __future__ import annotations
# random se utiliza para generar muestras aleatorias y desordenar listas
from random import shuffle, sample, choices, random
# deepcopy se utiliza para copiar los genomas de los padres seleccionados y poder modificarlos sin afectar a los padres
from copy import deepcopy
# abc se utiliza para crear clases abstractas
from abc import ABC, abstractmethod
# typing se utiliza para definir tipos de variables genericas mas sencillo, son como los templates de c++
from typing import TypeVar, Generic, List, Callable, Type
# enum se utiliza para definir enumeraciones de tipos complejos generados por las funciones de typing
from enum import Enum
# heapq se utiliza para obtener los n elementos mas grandes de una lista, utilizando un maxheap que funciona como un heapsort, con nlargest obtenemos los n elementos mas grandes
from heapq import nlargest
# mean se utiliza para obtener el promedio de las diferencias entre las generaciones, lejos de eso es innecesario para el funcionamiento del algoritmo
from statistics import mean


T = TypeVar('T', bound='Chromosome') # for returning self
# Clase cromosoma, su unica funcion es ser una clase abstracta para crear los genotipos desde aqui
class Chromosome(ABC):

    @abstractmethod
    def fitness(self) -> float:
        ...

    @classmethod
    @abstractmethod
    def random_instance(cls: Type[T]) -> T:
        ...

    @abstractmethod
    def crossover(self: T, other: T) -> tuple[T, T]:
        ...

    @abstractmethod
    def mutate(self) -> None:
        ...



C = TypeVar('C', bound=Chromosome) # Tipo de cromosomas a utilizar
class GeneticAlgorithm(Generic[C]): # Clase algoritmo genetico, tiene un tipo generico C que es el tipo de cromosomas a utilizar
    SelectionType = Enum("SelectionType", "ROULETTE TOURNAMENT") # Enumeracion para el tipo de seleccion a utilizar, que se refiere al metodo por el que se compararan las generaciones

    # inicializacion de la clase, recibe una poblacion inicial, un umbral, un maximo de generaciones, una probabilidad de mutacion, una probabilidad de cruzamient y un tipo de seleccion
    def __init__(self, initial_population: List[C], threshold: float, max_generations: int = 100, mutation_chance: float = 0.01, crossover_chance: float = 0.7, selection_type: SelectionType = SelectionType.TOURNAMENT) -> None:
        self._population: List[C] = initial_population
        self._threshold: float = threshold
        self._max_generations: int = max_generations
        self._mutation_chance: float = mutation_chance
        self._crossover_chance: float = crossover_chance
        self._selection_type: GeneticAlgorithm.SelectionType = selection_type
        self._fitness_key: Callable = type(self._population[0]).fitness # fitness es la funcion que determina que tan cerca del resultado esperado estamos

    """
    Dos metodos para escoger los individuos de la poblacion
    Escoge 2 individuos de la poblacion de acuerdo a su fitness, utilizando una rueda de probabilidades
    simula el que giremos una rueda que contiene a todos los individuos, y el que se detenga en un individuo es el que se escoge
    No sirve con valores de fitness negativo
    """
    def _pick_roulette(self, wheel: List[float]) -> tuple[C, C]:
        return tuple(choices(self._population, weights=wheel, k=2))
    
    # Escoge el numero de participantes y toma los 2 mejores de ellos
    def _pick_tournament(self, num_participants: int) -> tuple[C, C]:
        participants: List[C] = choices(self._population, k=num_participants)
        return tuple(nlargest(2, participants, key=self._fitness_key))
    
    # Reproduce y reemplaza la poblacion actual con una nueva afectada por cruzamiento y mutacion
    def _reproduce_and_replace(self) -> None:
        new_population: List[C] = []
        # Hasta que hayamos reemplazado a la poblacion actual se repetira el metodo
        while len(new_population) < len(self._population):
            # Escogemos los genotipos padre
            if self._selection_type == GeneticAlgorithm.SelectionType.ROULETTE:
                parents: tuple[C, C] = self._pick_roulette([x.fitness() for x in self._population])
            else:
                parents = self._pick_tournament(len(self._population) // 2)
            # utilizamos random para ver si se cruzan o no
            if random() < self._crossover_chance:
                new_population.extend(parents[0].crossover(parents[1]))
            else:
                new_population.extend(parents)
        # en dado caso de tener un numero impar de individuos, eliminamos uno
        if len(new_population) > len(self._population):
            new_population.pop()
        self._population = new_population # replace reference

    # Mutamos a los individuos de la poblacion en dado caso de caer dentro de una probabilidad preestablecida
    def _mutate(self) -> None:
        for individual in self._population:
            if random() < self._mutation_chance:
                # llama al metodo mutate de la clase al cryptarithmetics
                individual.mutate()

    # Corremos el algoritmo por n generaciones maximas, o hasta que se llegue a un umbral de fitness
    # y regresamos el mejor individuo encontrado
    def run(self) -> C:
        best: C = max(self._population, key=self._fitness_key)
        for generation in range(self._max_generations):
            if best.fitness() >= self._threshold: # si lo encontramos antes lo devolvemos
                return best
            print(f"Generation {generation} Best {best.fitness()} Avg {mean(map(self._fitness_key, self._population))}")
            self._reproduce_and_replace()
            self._mutate()
            highest: C = max(self._population, key=self._fitness_key)
            if highest.fitness() > best.fitness():
                best = highest 
        return best # lo mejor que pudimos encontrar en n generaciones


# Clase criptarithmetics, hereda de la clase cromosoma
class cryptarithmetics(Chromosome):

    # tenemos como atributos la lista de letras y el problema
    def __init__(self, letters: List[str], inp: str) -> None:
        self.letters: List[str] = letters
        self.inp: str = inp

    # definimos a la funcion fitness, esta nos devolvera la diferencia entre el valor resultante de la suma de las palabras y el valor esperado
    # si la diferencia es 1 entonces paramos
    def fitness(self) -> float:
        # obtenemos las palabras y el resultado
        first_word = self.inp[:self.inp.index("+")]
        second_word = self.inp[self.inp.index("+") + 1:self.inp.index("=")]
        result = self.inp[self.inp.index("=") + 1:]

        # agregamos las letras unicas a un diccionario para hacer facil su acceso
        values_dict = {}
        for letter in self.letters:
            values_dict[letter] = self.letters.index(letter)

        # calculamos los valores de cada una de las palabras segun su indice y su posicion
        first_word_value: int = 0
        second_word_value: int = 0
        result_value: int = 0

        for i in range(len(first_word)):
            # por cada ciclo, se agrega el valor de la letra multiplicado por 10 elevado a la posicion de la letra, empezando en 10^(n-1) y terminando en 10^0
            first_word_value += values_dict[first_word[i]] * (10 ** (len(first_word) - 1 - i))

        for i in range(len(second_word)):
            # lo mismo que en first_word_value
            second_word_value += values_dict[second_word[i]] * (10 ** (len(second_word) - 1 - i))

        for i in range(len(result)):
            # lo mismo que en first_word_value
            result_value += values_dict[result[i]] * (10 ** (len(result) - 1 - i)) 

        # calculamos el valor de la dferencia que habiamos comentado al principio del metodo y lo devolvemos
        difference: int = abs(result_value - (first_word_value + second_word_value))
        print([first_word_value, second_word_value, result_value])
        return 1 / (difference + 1)
    
    """
    random_instance es un metodo que sirve para inicializar el problema, a lo que se dedica es a obtener las palabras unicas y a rellenar el resto de la lista con espacios
    despues se desordena la lista y se devuelve un objeto de la clase cryptarithmetics, lo que le da un indice unico diferente de su posicion original a cada letra
    lo que nos ayuda a aplicar el algoritmo genetico
    """
    @classmethod
    def random_instance(cls, inp) -> cryptarithmetics:
        aux = inp
        aux = aux.upper()
        aux = aux.replace("+", " ")
        aux = aux.replace("=", " ")
        aux = aux.replace(" ", "")

        letters: List[str] = []
        letters.append(aux[0])
        for i in range(1, len(aux)):
            if aux[i] not in aux[:i]:
                letters.append(aux[i])

        for i in range(len(letters), 11):
            letters.append(" ")

        shuffle(letters)
        return cryptarithmetics(letters, inp.upper())
    
    # cruzamos dos individuos, intercambiando dos letras de sus listas
    def crossover(self, other: cryptarithmetics) -> tuple[cryptarithmetics, cryptarithmetics]:
        child1: cryptarithmetics = deepcopy(self)
        child2: cryptarithmetics = deepcopy(other)
        idx1, idx2 = sample(range(len(self.letters)), k=2)
        l1, l2 = child1.letters[idx1], child2.letters[idx2]
        child1.letters[child1.letters.index(l2)], child1.letters[idx2] = child1.letters[idx2], l2
        child2.letters[child2.letters.index(l1)], child2.letters[idx1] = child2.letters[idx1], l1
        return child1, child2
    
    # mutamos un individuo, intercambiando dos letras de su lista por dos escogidas aleatoriamente con la funcion sample
    def mutate(self) -> None: # swap two letters' locations
        idx1, idx2 = sample(range(len(self.letters)), k=2)
        self.letters[idx1], self.letters[idx2] = self.letters[idx2], self.letters[idx1]
        

    # metodo auxiliar para imprimir el resultado
    def __str__(self) -> str:
        first_word = self.inp[:self.inp.index("+")]
        second_word = self.inp[self.inp.index("+") + 1:self.inp.index("=")]
        result = self.inp[self.inp.index("=") + 1:]

        # agregamos las letras unicas a un diccionario para hacer facil su acceso
        values_dict = {}
        for letter in self.letters:
            values_dict[letter] = self.letters.index(letter)

        # calculamos los valores de cada una de las palabras segun su indice y su posicion
        first_word_value: int = 0
        second_word_value: int = 0
        result_value: int = 0

        for i in range(len(first_word)):
            # por cada ciclo, se agrega el valor de la letra multiplicado por 10 elevado a la posicion de la letra, empezando en 10^(n-1) y terminando en 10^0
            first_word_value += values_dict[first_word[i]] * (10 ** (len(first_word) - 1 - i))

        for i in range(len(second_word)):
            # lo mismo que en first_word_value
            second_word_value += values_dict[second_word[i]] * (10 ** (len(second_word) - 1 - i))

        for i in range(len(result)):
            # lo mismo que en first_word_value
            result_value += values_dict[result[i]] * (10 ** (len(result) - 1 - i)) 

        difference: int = abs(result_value - (first_word_value + second_word_value))
        return f"{first_word_value} + {second_word_value} = {result_value}"
    

if __name__ == "__main__":
    # Aqui nos pide cualquier problema de criptoaritmetica, en el formato ####+####=#####, donde los gatos solo significan la posicion de las letras, favor de no poner espacios
    inp = input("Input a cryptarithmetics problem: ")

    # inicializamos la poblacion inicial con 1000 individuos con el metodo random_instance que ya habiamos analizado
    initial_population: List[cryptarithmetics] = [cryptarithmetics.random_instance(inp) for _ in range(1000)]

    # ga es nuestra instancia del algoritmo genetico, aqui le pasamos la poblacion inicial, el umbral, el maximo de generaciones, la probabilidad de mutacion, la probabilidad de cruzamiento y el tipo de seleccion
    ga: GeneticAlgorithm[cryptarithmetics] = GeneticAlgorithm(initial_population=initial_population, threshold=1.0, max_generations = 1000, mutation_chance=0.4, crossover_chance=0.7, selection_type=GeneticAlgorithm.SelectionType.TOURNAMENT)

    # corremos el algoritmo y guardamos el resultado en result
    result: cryptarithmetics = ga.run()

    # imprimimos el resultado, puede ser que llegue al resultado final o que se quede en un maximo local, esto debido a como funcionan los algoritmos geneticos
    print(result)