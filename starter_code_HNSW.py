import os
import urllib.request
import faiss
import h5py
import numpy as np

def maybe_download(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, filename)
        print("Download finished.")

def evaluate_hnsw():
    # Download SIFT1M dataset if not already present
    dataset_url = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
    dataset_file = "sift-128-euclidean.hdf5"
    maybe_download(dataset_url, dataset_file)

    # Load data from HDF5 file
    with h5py.File(dataset_file, 'r') as f:
        base = f['train'][:].astype(np.float32)   # training vectors
        query = f['test'][:].astype(np.float32)   # test/query vectors

    # Build HNSW index with dimension 128 and M=16
    dim = 128
    index = faiss.IndexHNSWFlat(dim, 16)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 200

    # Add base vectors to the index
    index.add(base)

    # Perform a search for 10 nearest neighbors of the first query vector
    D, I = index.search(query[0:1], 10)

    # Write indices of the 10 approximate nearest neighbors to 'output.txt'
    with open('output.txt', 'w') as f:
        for idx in I[0]:
            f.write(f"{idx}\n")

if __name__ == "__main__":
    evaluate_hnsw()
