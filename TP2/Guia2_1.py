from Red import Red

# red[0] = entradas; red[1 y 2] = numero de neuronas por capa
red = Red([2, 2, 1])

red.entrenar("XOR_trn.csv", 1000, 0.01, 0.005)
red.testear("XOR_tst.csv")
