from PerceptronSimple import PerceptronSimple

# Repitiendo los procedimientos anteriormente utilizados, se repite con los datos OR_50 y OR_90 respectivamente.

or50 = PerceptronSimple("Guia1/OR_50_trn.csv", "Guia1/OR_50_trn.csv", "Guia1/OR_50_tst.csv", gamma=0.01, max_iterations=100, success_rate=0.95)
or50.train()
or50.test_data()

or90 = PerceptronSimple("Guia1/OR_90_trn.csv", "Guia1/OR_90_trn.csv", "Guia1/OR_90_tst.csv", gamma=0.01, max_iterations=100, success_rate=0.95)
or90.train()
or90.test_data()