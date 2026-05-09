"""Verify a few Experiment6_PyG .pt samples load under brep_mfr_pyg."""
import pathlib
import sys

import torch

def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else r"Z:\Experiment6_PyG")
    pts = list(root.rglob("*.pt"))
    if not pts:
        print("No .pt files found")
        sys.exit(1)
    errs = []
    for p in pts[:50]:
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
            assert getattr(d, "edge_index", None) is not None and d.edge_index.dim() == 2
            assert getattr(d, "node_data", None) is not None
            assert getattr(d, "label_feature", None) is not None
        except Exception as e:
            errs.append((str(p), repr(e)))
    mid = pts[len(pts) // 2]
    d = torch.load(mid, map_location="cpu", weights_only=False)
    print("n_pt_files", len(pts))
    print("spot_ok_first50", len(errs) == 0, "errs", len(errs))
    if errs:
        print("first_errors", errs[:3])
        sys.exit(2)
    print("mid_sample", mid)
    print("edge_index_shape", tuple(d.edge_index.shape), "node_data", tuple(d.node_data.shape))
    print("extras", getattr(d, "edge_path", None) is not None, getattr(d, "spatial_pos", None) is not None)
    sys.exit(0)


if __name__ == "__main__":
    main()
