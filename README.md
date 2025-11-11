# spotify-recommendation

## 1. Project Summary

**Problem:** 

Traditional shuffle feels rigid: it cycles genres too hard and surfaces songs you’ve heard way too recently. It ignores temporal context and treats all “liked” songs as equally good right now.

**Core idea:**

- Assume the graph consists only of songs the user likes.
- Build a song–song graph using audio & metadata similarity.
- When choosing the next track, don’t just pick a random neighbor:
- Start from a similarity-based candidate set.
- Adjust each candidate’s score using user-specific context features:
  how recently the song was played, whether it’s the same artist, whether keys / musical attributes align, optionally: cluster/mood diversity.
- Turn these scores into probabilities with a softmax and sample the next song → a controlled random walk over the graph.

**Learning the “weights”:**

- Use synthetic users and interaction logs (since you don’t have real user data) to estimate how important recency, same artist, key match, etc. should be.
- Fit a simple softmax choice model (cross-entropy) to learn global feature weights.

**Positioning:**
It’s a time-aware, graph-based shuffle that sits between: pure content-based recsys (song similarity), and full-blown user-log–driven RL/bandits, with a clear path to production (global model + user embeddings, online learning from skips/replays).

## 2. Roadmap for Implementation

### Phase 1 – Data & feature pipeline
**Goal:** Clean track data and extract a consistent feature matrix for each song.  
**Steps:**  
1. Load your 1M+ song dataset.  
2. Select base features:  
   - audio: tempo, energy, valence, danceability, loudness, etc.  
   - metadata: key, mode, time signature, year, maybe genre tags if available.  
3. Clean:  
   - drop rows with missing critical features,  
   - standardize numerical columns (z-score),  
   - encode categorical features (e.g. key, mode) if needed.  
4. Save:  
   - X_raw (pandas DataFrame) with track_id as index and cleaned features as columns.  

**Deliverable:** X_raw.npy or similar + a small EDA notebook.  

---

### Phase 2 – Representation learning (PCA + autoencoder)
**Goal:** Get a compact, meaningful embedding for each song.  
**Steps:**  
1. PCA baseline:  
   - Fit PCA on X_raw.  
   - Choose enough components to explain, say, ~90–95% variance.  
   - Save X_pca (orthogonal representation).  

2. Autoencoder on PCA output:  
   - Define a simple fully-connected autoencoder.  
   - Train to reconstruct X_pca.  
   - Extract bottleneck representation as X_embed.  

3. Evaluate:  
   - reconstruction loss vs PCA-only,  
   - optional: visualize a 2D projection with t-SNE/UMAP for sanity.  

**Deliverable:** X_embed as your main song embedding for graph building.  

---

### Phase 3 – Graph construction
**Goal:** Build a k-NN graph over liked songs using the embedding.  
**Steps:**  
- Choose a similarity metric (cosine is a good default on X_embed).  
- For each song:  
  - find its k nearest neighbors (e.g. k = 50).  
  - store edge weight = similarity score.  
- Represent the graph:  
  - adjacency list or sparse matrix format,  
  - ensure you can quickly retrieve neighbors for a given song.  

**Deliverable:** graph.pkl (neighbors + weights for each track_id).  

---

### Phase 4 – Base shuffle recommender (no external features yet)
**Goal:** Implement a simple graph-based shuffle using only similarity.  
**Steps:**  
- Given a current song:  
  - Retrieve its neighbors from the graph.  
  - Compute a softmax over similarity scores (optionally with a temperature parameter).  
  - Sample the next song according to this distribution.  
- Implement a random-walk function:  
  - Input: starting song, number of steps.  
  - Output: generated playlist.  

At this point you have a graph-based shuffle baseline to compare later.  

---

### Phase 5 – Add external features and scoring function
**Goal:** Turn this into a time-aware, context-aware shuffle.  
**Define per-song, per-user state features:**  
 - For each candidate song j given current song i:  
   - sim(i, j) → from graph edge weight
   - recency_score(j) → e.g. negative function of “time since last played”
   - same_artist(i, j) → 1 or 0
   - key_match(i, j) → 1 or 0 (or a small distance measure)
   - optional: cluster_diversity → whether j’s cluster differs from recent history

**Define a scoring function:**
score(j) = w*sim * sim(i, j) + w*recency * recency*score(j) + w_artist * same*artist(i, j) + w_key * key_match(i, j) + …
Convert scores to probabilities via softmax and update your random-walk sampler.

**Deliverable:** a function recommend_next(current_song, user_state, weights) that returns a distribution over candidates.


---

### Phase 6 – Synthetic users + parameter estimation (global model)
**Goal:** Learn good global weights for recency, same-artist penalty, etc., instead of hand-tuning.  

**6.1 Define “true” behavior for simulation**  
Choose some “true” weights (call them w_true) that reflect what you think is good listening behavior.  
Use these to define a utility function that a simulated user uses internally to pick songs.  

**6.2 Simulate user sessions**  
For each synthetic user:  
- Sample a taste profile over clusters (from your clustering on X_embed).  
- Pick an initial song.  
- For T steps:  
  - Get candidate neighbors.  
  - Compute utilities with w_true and the current state (recency, artist, etc.).  
  - Sample the next song (softmax over utilities).  
  - Log:  
    - feature matrix for candidates at this step,  
    - which candidate was chosen.  
  - Update recency state.  

You now have interaction data: many “choice sets” with one chosen song per set.  

**6.3 Estimate parameters from choices**  
Build a dataset where each row is:  
- candidate features for step t,  
- label = index of chosen song in the candidate set.  

Train a simple linear softmax model (multinomial logistic regression or PyTorch nn.Linear + CrossEntropyLoss) to predict the chosen song.  
The learned weights are your estimated global parameters w_hat.  

**6.4 Evaluate**  
Compare w_hat to w_true (do they recover signs and relative magnitudes?).  
Show that using w_hat in your recommender produces “reasonable” playlists (low repeat rate, decent genre diversity, etc.).  

**Deliverable:** a `train_choice_model.py` that:  
- loads simulated logs,  
- trains global weights,  
- saves them to disk,  
- and prints evaluation metrics.  

---

### Phase 7 – Evaluation & diagnostics
**Goal:** Show that your shuffle is “better” in interpretable ways.  
**Ideas:**  
- Repeat-avoidance metrics:  
  - fraction of songs that appear within the last N tracks,  
  - comparison vs a purely similarity-based shuffle.  
- Diversity / coherence:  
  - distribution over clusters in a long session,  
  - average similarity between consecutive songs (tradeoff between variety and coherence).  
- Qualitative examples:  
  - Show side-by-side “vanilla shuffle” vs “your shuffle” from the same starting track.  
  - Explain why your algorithm chose each next song (recency penalty, artist diversity, etc.).  

**Deliverables:** a few plots + tables + 2–3 narrative examples for your report/presentation.  

---

### Phase 8 – Small interactive demo
**Goal:** Make it feel like a product, not just a model.  
**Options:**  
- Simple:  
  - A Jupyter notebook with widgets:  
    - select seed song by name,  
    - generate N recommendations,  
    - print song metadata.  
- Nicer:  
  - A Streamlit app:  
    - search for a song,  
    - click “Smart Shuffle”,  
    - see the next track(s) and some “why this song” explanations (recency, similarity, etc.).  

This is huge for portfolio and interviews.  

---

### Phase 9 – Documentation & “Spotify pitch”
**Goal:** Tie everything together as a case study.  
**In your README / slide deck:**  
- Problem: why current shuffle feels bad.  
- Approach:  
  - graph over liked tracks,  
  - embeddings (PCA + autoencoder),  
  - time-aware softmax over neighbors.  
- Learning and evaluation:  
  - synthetic user simulation,  
  - softmax choice model to estimate weights,  
  - metrics: repeat-avoidance, diversity, qualitative examples.  
- Scalability & business discussion:  
  - use a global model of weights in this project,  
  - outline how production could add:  
    - user embeddings that modulate these weights,  
    - online updates from skips/replays via contextual bandits or RL.  
- Future work:  
  - incorporate cross-user data for cold start,  
  - model sessions more explicitly (time of day, device, activity),  
  - A/B testing strategy. 
