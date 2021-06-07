# Incremental-SVM
KAIST 2021 Spring Semester - CS504: Computational Geometry - Term Project

20213571 Sanghyun Jung, 20215204 Jeongseok Oh, 20209016 Sangmin Lee

## HMISVM (Hard Margin Incremental SVM)
### Pseudocode
```
n ← number of dataset
d ← dimension of data

Choose d points from positive class, set as SVs of class 1
Choose d points from negative class, set as SVs of class 0

W, b ← decision boundary using 2*d support vectors
X, y ← Randomly permuted dataset

For i=1 to n:
    If X[i] is in SV region:
        Do nothing
    Else:
        For sv in SVs of class y[i]:
            Replace sv to X[i] and make hyperplane b
            Check new SV region by b contains old SV
            Check new SV region by b does not contain data in other class

            If b satisfies conditions:
                X[i] is new SV and sv is removed from SVs
                b is new SV boundary
        
        W, b ← decision boundary using new 2*d SVs
```

### Expriment Result
Expreriment Result for Linearly Separable Dataset

| Train Dataset Size | Test Dataset Size | Accuracy |
|:---:|:---:|:---:|
| 90     | 10     | 1.0000 |
| 900    | 100    | 0.9900 |
| 9000   | 1000   | 0.9990 |
| 90000  | 10000  | 0.9997 |
