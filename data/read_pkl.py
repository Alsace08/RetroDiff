import pickle

f = open('data/USPTO50K/valid/rxn_data_1.pkl', 'rb')
inf = pickle.load(f)
doc = open('data/pkl_sample.txt', 'w')
print(inf, file=doc)
