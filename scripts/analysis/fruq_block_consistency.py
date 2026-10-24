import numpy as np, json, sys
from collections import Counter
d=sys.argv[1]
folds=json.load(open(d+'/folds.json'))
z=np.load(d+'/dino_features.npz'); fr=z['fruits']; Y=z['labels']; Fe=z['features']
C=['F','M','R']
print("MobileNet: per (fruit,class) block -> share of most common prediction, and that prediction")
shares=[]
for f in folds:
    m=np.load(f"{d}/mnv3_f{f['fold']}_s0.npz"); p=m['logits'].argmax(1); y=m['labels']
    row=[]
    for c in range(3):
        pc=p[y==c]; mc,n=Counter(pc).most_common(1)[0]; shares.append(n/len(pc))
        row.append(f"{C[c]}->{C[mc]} {n/len(pc):.2f} (n={len(pc)})")
    print(f"  {f['test']:9s}", " | ".join(row))
print(f"median block share {np.median(shares):.2f}, blocks with >=90% same prediction: {np.mean(np.array(shares)>=0.9)*100:.0f}%")
# how similar are frames within a block vs across fruits (DINOv2 cosine)
Fn=Fe/np.linalg.norm(Fe,axis=1,keepdims=True)
rng=np.random.RandomState(0)
def msim(a,b):
    ia=rng.choice(a,min(300,len(a)),replace=False); ib=rng.choice(b,min(300,len(b)),replace=False)
    return float((Fn[ia]@Fn[ib].T).mean())
fruits=sorted(set(fr)); w,cf,ff=[],[],[]
for f_ in fruits:
    for c in range(3):
        a=np.where((fr==f_)&(Y==c))[0]; w.append(msim(a,a))
        b=np.where((fr!=f_)&(Y==c))[0]; ff.append(msim(a,b))
        o=np.where((fr==f_)&(Y!=c))[0]; cf.append(msim(a,o))
print(f"\nDINOv2 cosine similarity: same fruit+class {np.mean(w):.2f} | same fruit, other class {np.mean(cf):.2f} | same class, other fruit {np.mean(ff):.2f}")
