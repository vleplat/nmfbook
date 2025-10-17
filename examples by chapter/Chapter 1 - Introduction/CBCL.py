from __future__ import annotations
import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE not in sys.path:
	sys.path.insert(0, BASE)
try:
	from algorithms.nmf import fro_nmf, FroNMFOptions
except ModuleNotFoundError:
	# Fallback: ensure realpath is on sys.path then retry
	if BASE not in sys.path:
		sys.path.insert(0, BASE)
	from algorithms.nmf import fro_nmf, FroNMFOptions

DATA = os.path.join(BASE, "data sets", "CBCL.mat")
FIGS = os.path.join(BASE, "figs")


def load_cbcl(path_mat: str) -> np.ndarray:
	mat = sio.loadmat(path_mat)
	if "X" not in mat:
		raise KeyError("CBCL.mat must contain matrix 'X' with vectorized images in columns")
	X = np.array(mat["X"], dtype=np.float64)
	# Ensure nonnegative and clip
	X = np.maximum(X, 0.0)
	return X


def infer_image_shape(num_pixels: int) -> tuple[int, int]:
	# Prefer square if possible
	s = int(np.sqrt(num_pixels))
	if s * s == num_pixels:
		return s, s
	# Otherwise keep as (num_pixels, 1)
	return num_pixels, 1


def grid_show_columns(X: np.ndarray, Li: int, Co: int, num: int = 64, perrow: int = 8, title: str | None = None, path: str | None = None, transpose_images: bool = False):
	m, n = X.shape
	num = min(num, n)
	rows = int(np.ceil(num / perrow))
	# Normalize columns to max 1 for visualization
	Xn = X[:, :num].copy()
	col_max = Xn.max(axis=0)
	col_max[col_max == 0] = 1.0
	Xn = Xn / col_max
	# Build canvas with 1px separators
	sep = 1
	H = rows * Li + (rows - 1) * sep
	W = perrow * Co + (perrow - 1) * sep
	canvas = np.ones((H, W), dtype=float)
	idx = 0
	for r in range(rows):
		for c in range(perrow):
			if idx >= num:
				break
			img = Xn[:, idx].reshape(Li, Co)
			if transpose_images:
				img = img.T
			r0 = r * (Li + sep)
			c0 = c * (Co + sep)
			canvas[r0 : r0 + Li, c0 : c0 + Co] = img
			idx += 1
	plt.figure(figsize=(min(12, W / 10), min(12, H / 10)))
	plt.imshow(canvas, cmap="gray_r", vmin=0.0, vmax=1.0)
	plt.axis("off")
	if title:
		plt.title(title)
	plt.tight_layout()
	if path:
		plt.savefig(path, bbox_inches="tight")
	plt.close()


def main():
	os.makedirs(FIGS, exist_ok=True)
	X = load_cbcl(DATA)
	m, n = X.shape
	Li, Co = infer_image_shape(m)
	# Grid of sample faces
	grid_path = os.path.join(FIGS, "cbcl_samples_grid.pdf")
	grid_show_columns(X, Li, Co, num=64, perrow=8, title="CBCL sample faces", path=grid_path, transpose_images=True)
	# Average face
	avg = X.mean(axis=1)
	avg_max = max(avg.max(), 1e-12)
	avg_img = (avg / avg_max).reshape(Li, Co).T
	plt.figure(figsize=(4, 4))
	plt.imshow(avg_img, cmap="gray_r", vmin=0.0, vmax=1.0)
	plt.axis("off")
	plt.title("CBCL average face")
	plt.tight_layout()
	avg_path = os.path.join(FIGS, "cbcl_average_face.pdf")
	plt.savefig(avg_path, bbox_inches="tight")
	plt.close()
	# Run Frobenius-NMF to extract basis faces
	r = 49
	opts = FroNMFOptions(maxiter=300, timemax=15.0, algo="HALS", inneriter=200, delta=1e-4, display=1)
	W, H, e, t, _ = fro_nmf(X, r, opts)
	# Visualize basis (columns of W)
	basis_path = os.path.join(FIGS, "cbcl_basis_faces.pdf")
	grid_show_columns(W, Li, Co, num=min(49, r), perrow=7, title="CBCL basis faces (NMF)", path=basis_path, transpose_images=True)
	# Reconstruction error curve
	plt.figure()
	plt.plot(t, e)
	plt.xlabel("Time (s.)")
	plt.ylabel("Relative error ||X-WH||_F / ||X||_F")
	plt.tight_layout()
	err_path = os.path.join(FIGS, "cbcl_fronmf_error_vs_time.pdf")
	plt.savefig(err_path, bbox_inches="tight")
	plt.close()
	print("Saved:", grid_path)
	print("Saved:", avg_path)
	print("Saved:", basis_path)
	print("Saved:", err_path)


if __name__ == "__main__":
	main()
