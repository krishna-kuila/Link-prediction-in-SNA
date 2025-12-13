from node2vec import Node2Vec
import pickle
import os

# CONFIG
GRAPH_PATH = 'models/twitter_graph.gpickle'
MODEL_PATH = 'models/twitter_node2vec.model'

def train_node2vec():
    print("1. Loading Graph...")
    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError(f"Run Linkprediction.ipynb first to generate {GRAPH_PATH}")
        
    with open(GRAPH_PATH, 'rb') as f:
        G = pickle.load(f)
    print(f"   Loaded {G.number_of_nodes()} nodes.")

    print("2. Initializing Node2Vec (Generating Random Walks)...")
    # p=1, q=0.5 -> We prefer DFS (exploration) to find structural equivalents (shared interests)
    # workers=4 -> Uses 4 CPU cores for speed
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=100, p=1, q=0.5, workers=4, quiet=False)

    print("3. Training the Neural Network (Word2Vec)...")
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    print("4. Saving Model...")
    model.save(MODEL_PATH)
    print(f"   Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_node2vec()