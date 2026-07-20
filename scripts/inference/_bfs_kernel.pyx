# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Compiled all-pairs BFS kernel with edge-path reconstruction.

Drop-in replacement for the serial BFS in ``json_to_brepmfr_pyg_optimized.py``.
Uses parent-backtrack instead of per-edge prefix-row copies and runs the
entire inner loop in compiled C via Cython typed memoryviews.

Build::

    conda activate brep_mfr_pyg
    python scripts/inference/build_bfs.py
"""

import numpy as np
cimport numpy as cnp

cnp.import_array()

ctypedef cnp.int32_t I32


def all_pairs_bfs(
    const I32[::1] offsets,
    const I32[::1] targets,
    const I32[::1] edge_ids,
    int num_nodes,
    int max_dist,
):
    """
    All-pairs BFS returning ``(spatial_pos, edges_path)`` as numpy int32 arrays.

    Parameters
    ----------
    offsets : int32[num_nodes + 1]
        CSR row offsets.  Neighbours of node *u* live at
        ``targets[offsets[u] : offsets[u+1]]``.
    targets : int32[nnz]
        Target node for each directed arc.
    edge_ids : int32[nnz]
        Edge ID (original arc index) for each directed arc.
    num_nodes : int
        Number of nodes.
    max_dist : int
        Maximum number of edge IDs stored per (source, target) path.

    Returns
    -------
    spatial_pos : int32[num_nodes, num_nodes]
        BFS distances; unreachable pairs have value 10**9.
    edges_path : int32[num_nodes, num_nodes, max_dist]
        Edge-ID sequences for shortest paths; unused slots are -1.
    """
    if num_nodes == 0:
        return (
            np.empty((0, 0), dtype=np.int32),
            np.empty((0, 0, max_dist), dtype=np.int32),
        )

    cdef:
        I32[:, ::1] sp
        I32[:, :, ::1] ep
        I32[::1] dist
        I32[::1] par_n
        I32[::1] par_e
        I32[::1] que
        I32[::1] pbuf
        int s, u, v, du, t, d, k
        int qh, qt, a0, a1, j
        int cur, plen
        int BIG = 1000000000

    # ---- allocate output + working arrays --------------------------------
    sp_np = np.full((num_nodes, num_nodes), BIG, dtype=np.int32)
    ep_np = np.full((num_nodes, num_nodes, max_dist), -1, dtype=np.int32)
    sp = sp_np
    ep = ep_np

    dist_np  = np.empty(num_nodes, dtype=np.int32)
    par_n_np = np.empty(num_nodes, dtype=np.int32)
    par_e_np = np.empty(num_nodes, dtype=np.int32)
    que_np   = np.empty(num_nodes, dtype=np.int32)
    pbuf_np  = np.empty(num_nodes, dtype=np.int32)
    dist  = dist_np
    par_n = par_n_np
    par_e = par_e_np
    que   = que_np
    pbuf  = pbuf_np

    # ---- main loop -------------------------------------------------------
    for s in range(num_nodes):
        # reset distances
        for j in range(num_nodes):
            dist[j] = -1
        dist[s] = 0
        sp[s, s] = 0

        # ---- BFS from source s -------------------------------------------
        qh = 0
        qt = 0
        que[qt] = s
        qt += 1

        while qh < qt:
            u = que[qh]
            qh += 1
            du = dist[u]
            a0 = offsets[u]
            a1 = offsets[u + 1]
            for j in range(a0, a1):
                v = targets[j]
                if dist[v] != -1:
                    continue
                dist[v] = du + 1
                par_n[v] = u
                par_e[v] = edge_ids[j]
                que[qt] = v
                qt += 1

        # ---- backtrack to fill edge paths --------------------------------
        for t in range(num_nodes):
            d = dist[t]
            if d <= 0:
                continue
            sp[s, t] = d

            cur = t
            plen = 0
            while cur != s:
                pbuf[plen] = par_e[cur]
                cur = par_n[cur]
                plen += 1

            k = plen if plen < max_dist else max_dist
            for j in range(k):
                ep[s, t, j] = pbuf[plen - 1 - j]

    return sp_np, ep_np
