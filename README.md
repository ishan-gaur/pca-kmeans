Just a quick foray into PCA, k-means, and writing python bindings for C++

Goals:
1. Operational PCA and Kmeans algorithms in Pytorch
2. Write C++ code for basic matrix ops: element-wise arithmetic, matrix
   multiply, norm (forbenius), transpose, and broadcasting. Also need 
   initializations with random, zeros, and ones
3. Implement PCA and Kmeans with those bindings
4. Optimize C++ with processor specific ASM
5. Add both in pure python

