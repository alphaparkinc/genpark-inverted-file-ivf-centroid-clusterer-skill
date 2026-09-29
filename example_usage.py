from client import IVFClusterer

ivf = IVFClusterer(n_centroids=2)
ivf.fit_centroids([[1.0, 0.0], [0.0, 1.0]])
ivf.insert("doc_x", [0.95, 0.05])
ivf.insert("doc_y", [0.05, 0.95])
matches = ivf.query([1.0, 0.0], nprobe=1, top_k=1)
print("IVF Top Match:", matches)
