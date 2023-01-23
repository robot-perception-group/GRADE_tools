import pickle as pkl
f = open('mapping.pkl','rb')
plan = pkl.load(f)
print(plan)