# kNNTutorial
Python notebooks for kNN Tutorial paper available here https://arxiv.org/abs/2004.04523:
1. `kNN-Basic`: Code for a basic *k*-NN classifier in `scikit-learn`.
2. `kNN-Correlation`: How to use correlation as the *k*-NN metric `scikit-learn`.
3. `kNN-Cosine`: How to use Cosine as the *k*-NN metric in `scikit-learn`. Using Cosine similarity for text classification. 
4. `kNN-DTW`: Using the `tslearn` library for time-series classification using DTW.
5. `kNN-MetricLearn`: Using the `metric-learn` library to *learn* a similarity metric. 
6. `kNN-Speedup`: Testing the `scikit-learn` speedup mechanisms (`kd_tree` and `ball_tree`) on four datasets. Requires the four datasets and a `py` file `kNNDataLoader.py` to run (all available in this repo). 
7. `kNN-Annoy`: Testing the impact of using `annoy` for speedup. `annoy` provides code for Approximate Nearest Neighbour that may not be as accurate as full *k*-NN. Requires `kNNAnnoy.py` that contains some wrapper code for `annoy`. Also requires the four datasets and a `py` file `kNNDataLoader.py` to run (all available in this repo). 
8. `kNN-PCA`: Some code to use PCA to estimate the intrinsic dimension of the four datasets. Requires `kNNDataLoader.py` and the data files. 
9. `kNN-InstSel`: An assessment of two instance selection algorithms (CNN and CRR) on three datasets. Requires `kNNEdit.py` that containst basic implementations of the two algorithms. Requires `kNNDataLoader.py` and the data files. 
10. `kNN-Model-Selection`: Using `grid-search` for model selection (hyper-parameter tuning). 
