from Red import Red

# red[0] = entradas; red[1 y 2] = numero de neuronas por capa
red = Red([2, 2, 1])

red.entrenar("XOR_trn.csv", 100, 0.1, 0.05)

red.testear("XOR_tst.csv")
