import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is on sys.path for imports
PROJECT_BASE = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_BASE not in sys.path:
	sys.path.insert(0, PROJECT_BASE)
try:
	from algorithms.symmetric_nmf import symnmf, SymNMFOptions
except ModuleNotFoundError:
	if PROJECT_BASE not in sys.path:
		sys.path.insert(0, PROJECT_BASE)
	from algorithms.symmetric_nmf import symnmf, SymNMFOptions


def load_karate_mat(path_mat: str) -> np.ndarray:
	mat = sio.loadmat(path_mat, squeeze_me=True, struct_as_record=False)
	# Case 1: edges list present (MATLAB style: 1-based indices)
	if "edges" in mat:
		edges = np.array(mat["edges"], dtype=np.int64)
		if edges.ndim != 2 or edges.shape[1] != 2:
			raise ValueError("'edges' must be an E x 2 array")
		# Convert to 0-based indices
		u = edges[:, 0].astype(int) - 1
		v = edges[:, 1].astype(int) - 1
		n = int(max(u.max(), v.max()) + 1)
		A = np.zeros((n, n), dtype=np.float64)
		A[u, v] = 1.0
		A[v, u] = 1.0
		# Zero diagonal for adjacency visualization
		np.fill_diagonal(A, 0.0)
		return A
	# Case 2: search for square numeric matrices
	candidates = []
	for key, val in mat.items():
		if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[0] == val.shape[1] and val.shape[0] > 1:
			candidates.append((key, np.array(val, dtype=np.float64)))
	for preferred in ("A", "karate", "W", "Adj", "adj"):
		for key, arr in candidates:
			if key == preferred:
				A = arr
				break
		else:
			continue
		break
	else:
		if not candidates:
			raise KeyError(f"No adjacency found in {path_mat} (no 'edges' and no square matrix)")
		A = max(candidates, key=lambda kv: kv[1].shape[0])[1]
	A = np.maximum(A, 0.0)
	A = 0.5 * (A + A.T)
	np.fill_diagonal(A, 0.0)
	return A


def plot_rank1_terms(H: np.ndarray, out_dir: str):
	os.makedirs(out_dir, exist_ok=True)
	n, r = H.shape
	for k in range(r):
		hk = H[:, [k]]  # column k
		rank1 = hk @ hk.T  # outer product
		plt.figure(figsize=(4, 4))
		plt.imshow(rank1, cmap="viridis")
		plt.colorbar()
		plt.title(f"Rank-1 term k={k+1}")
		plt.tight_layout()
		out = os.path.join(out_dir, f"karate_rank1_k{k+1}.pdf")
		plt.savefig(out, bbox_inches="tight")
		plt.close()


def plot_raw_adjacency(A: np.ndarray, out_dir: str):
	os.makedirs(out_dir, exist_ok=True)
	plt.figure(figsize=(5, 5))
	plt.imshow(A, cmap="gray_r")
	plt.title("Adjacency (raw order)")
	plt.tight_layout()
	out = os.path.join(out_dir, "karate_adjacency_raw.pdf")
	plt.savefig(out, bbox_inches="tight")
	plt.close()


def plot_communities(A: np.ndarray, H: np.ndarray, out_dir: str):
	os.makedirs(out_dir, exist_ok=True)
	labels = np.argmax(H, axis=1)
	r = H.shape[1]
	# Build ordering: for each community k, sort nodes by descending H[:,k]
	order = []
	bounds = [0]
	for k in range(r):
		idx = np.where(labels == k)[0]
		if idx.size:
			idx_sorted = idx[np.argsort(-H[idx, k])]
			order.extend(idx_sorted.tolist())
			bounds.append(bounds[-1] + idx_sorted.size)
	order = np.array(order, dtype=int)
	A_ord = A[np.ix_(order, order)]
	plt.figure(figsize=(5, 5))
	plt.imshow(A_ord, cmap="gray_r")
	# Draw block boundaries
	for b in bounds:
		plt.axhline(b - 0.5, color="red", linewidth=0.8)
		plt.axvline(b - 0.5, color="red", linewidth=0.8)
	plt.title("Adjacency reordered by communities (blocks)")
	plt.tight_layout()
	out = os.path.join(out_dir, "karate_communities_reordered.pdf")
	plt.savefig(out, bbox_inches="tight")
	plt.close()


def plot_hht(H: np.ndarray, out_dir: str):
	os.makedirs(out_dir, exist_ok=True)
	HHt = H @ H.T
	plt.figure(figsize=(5, 5))
	plt.imshow(HHt, cmap="viridis")
	plt.colorbar()
	plt.title("H H^T (heatmap)")
	plt.tight_layout()
	out = os.path.join(out_dir, "karate_HHt.pdf")
	plt.savefig(out, bbox_inches="tight")
	plt.close()


def plot_graph(A: np.ndarray, H: np.ndarray, out_dir: str):
	os.makedirs(out_dir, exist_ok=True)
	try:
		import networkx as nx
	except Exception as exc:
		print("networkx not installed; skipping graph plot:", exc)
		return
	G = nx.from_numpy_array((A > 0).astype(int))
	labels = np.argmax(H, axis=1)
	# Color map per community
	colors = plt.cm.tab10(labels % 10)
	pos = nx.spring_layout(G, seed=0)
	plt.figure(figsize=(6, 5))
	nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=200, linewidths=0.5, edgecolors='k')
	nx.draw_networkx_edges(G, pos, alpha=0.6)
	nx.draw_networkx_labels(G, pos, font_size=8, font_color='white')
	plt.axis('off')
	plt.tight_layout()
	out = os.path.join(out_dir, "karate_graph_by_community.pdf")
	plt.savefig(out, bbox_inches="tight")
	plt.close()


def main():
	base = PROJECT_BASE
	data_dir = os.path.join(base, "data sets")
	cand = ["karate.mat", "Karate.mat"]
	for name in cand:
		path = os.path.join(data_dir, name)
		if os.path.exists(path):
			karate_path = path
			break
	else:
		raise FileNotFoundError("karate .mat file not found in 'data sets' directory")

	A = load_karate_mat(karate_path)

	# Use r=2 for communities
	r = 2
	options = SymNMFOptions(
		maxiter=1000,
		timelimit=10.0,
		initmatrix="dense01",
		seed=0,
		shuffle_columns=1,
		display="on",
	)

	H, e, t = symnmf(A, r, options)

	fig_dir = os.path.join(base, "figs")
	plot_rank1_terms(H, fig_dir)
	plot_raw_adjacency(A, fig_dir)
	plot_communities(A, H, fig_dir)
	plot_hht(H, fig_dir)
	plot_graph(A, H, fig_dir)
	plt.figure()
	plt.plot(t, e)
	plt.xlabel("Time (s.)")
	plt.ylabel("Error  -  1/2 * ||A - HH^T||_F^2")
	plt.tight_layout()
	plt.savefig(os.path.join(fig_dir, "karate_error_vs_time.pdf"), bbox_inches="tight")
	plt.close()
	print("Saved karate demo figures to", fig_dir)


if __name__ == "__main__":
	main()
