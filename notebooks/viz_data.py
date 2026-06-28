import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import json

    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    def repo_root():
        """Find the repo root (the dir containing cached_data/) walking up from cwd."""
        for d in [Path.cwd(), *Path.cwd().parents]:
            if (d / "cached_data").is_dir():
                return d
        return Path.cwd()

    ROOT = repo_root()
    return ROOT, json, mo, np, plt


@app.cell
def _(mo):
    mo.md("""
    # Planning-Model data viewer

    Decode and render raw cached problems for both domains. Each cached
    sequence is laid out as **`START | start-state | goal-state | moves [| states] | END`**.
    For *world-model* (`wm`) data the resulting state is interleaved after every move;
    for `baseline` data only the moves are stored and we replay them to recover the states.
    """)
    return


@app.cell
def _(mo):
    # ---- controls ----
    domain = mo.ui.dropdown(
        ["blocks_world", "eight_puzzle"], value="blocks_world", label="domain"
    )
    arch = mo.ui.dropdown(["baseline", "wm"], value="baseline", label="architecture")
    split = mo.ui.dropdown(["std", "productivity"], value="std", label="split")
    fold = mo.ui.dropdown(["train", "test"], value="train", label="fold")
    mo.hstack([domain, arch, split, fold], justify="start")
    return arch, domain, fold, split


@app.cell
def _(ROOT, arch, domain, fold, json, mo, split):
    # ---- load the matching cached file ----
    suffix = "" if split.value == "std" else "_productivity"
    fname = ROOT / "cached_data" / f"{domain.value}_{fold.value}_{arch.value}{suffix}.json"
    data = json.load(open(fname))
    mo.md(f"**Loaded** `{fname.name}` — {len(data):,} problems.")
    return (data,)


@app.cell
def _(data, mo):
    idx = mo.ui.slider(0, len(data) - 1, value=0, label="problem index", full_width=True)
    idx
    return (idx,)


@app.cell
def _(plt):
    BW_VOCAB = {
        0: "START", 1: "END", 2: "A", 3: "B", 4: "C", 5: "D",
        6: "POS_0", 7: "POS_1", 8: "POS_2", 9: "POS_3", 10: "PAD",
    }
    BW_COLORS = {"A": "#e74c3c", "B": "#3498db", "C": "#2ecc71", "D": "#f4d03f"}

    def bw_decode_state(names):
        """8 token-names -> 4 towers (bottom-to-top). Mirrors _decode_state."""
        state = [[] for _ in range(4)]
        buf = []
        for t in names:
            if t.startswith("POS_"):
                state[int(t[-1])] = list(reversed(buf))
                buf = []
            else:
                buf.append(t)
        return state

    def bw_apply(state, block, pos):
        """Move `block` (top of its tower) to position `pos`. Mirrors apply_action."""
        new = [list(c) for c in state]
        for c in new:
            if c and c[-1] == block:
                c.pop()
                break
        new[pos].append(block)
        return new

    def bw_parse(seq, is_wm):
        names = [BW_VOCAB[t] for t in seq if BW_VOCAB[t] != "PAD"]
        start = bw_decode_state(names[1:9])      # after START
        goal = bw_decode_state(names[9:17])
        body = names[17:]
        if body and body[-1] == "END":
            body = body[:-1]
        steps, cur, i = [], start, 0
        while i + 1 < len(body):
            block, pos = body[i], int(body[i + 1][-1])
            i += 2
            if is_wm:
                after = bw_decode_state(body[i : i + 8])
                i += 8
            else:
                after = bw_apply(cur, block, pos)
            steps.append((f"{block}→p{pos}", after))
            cur = after
        return start, goal, steps

    def bw_draw(ax, state, title):
        for p, tower in enumerate(state):
            for h, b in enumerate(tower):
                ax.add_patch(
                    plt.Rectangle((p, h), 0.9, 0.9, facecolor=BW_COLORS[b], edgecolor="black")
                )
                ax.text(p + 0.45, h + 0.45, b, ha="center", va="center", fontweight="bold")
        ax.set_xlim(-0.15, 4.05)
        ax.set_ylim(-0.15, 4.6)
        ax.set_xticks([p + 0.45 for p in range(4)])
        ax.set_xticklabels([f"p{p}" for p in range(4)], fontsize=7)
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)

    return bw_draw, bw_parse


@app.cell
def _(np):
    EP_MOVES = {10: "up", 11: "down", 12: "left", 13: "right"}

    def ep_apply(state, move):
        """Slide the blank (0) one cell in `move` direction. Mirrors apply_move."""
        s = state.copy()
        (r, c), = np.argwhere(s == 0)
        dr, dc = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[move]
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            s[r, c], s[nr, nc] = s[nr, nc], s[r, c]
        return s

    def ep_parse(seq, is_wm):
        """Layout (see eight_puzzle.encode_sequence):
        [dummy move][9 start][PAD=15][9 goal][moves (+9-grid per move for wm)][SEP=14][PAD...]
        """
        body = seq[1:]  # drop the leading dummy 'right' move
        start = np.array(body[0:9]).reshape(3, 3)
        goal = np.array(body[10:19]).reshape(3, 3)  # body[9] is the PAD separator
        rest = body[19:]
        steps, cur, i = [], start, 0
        while i < len(rest):
            t = rest[i]
            if t in (14, 15):  # SEP / trailing PAD -> done
                break
            if t in EP_MOVES:
                mv = EP_MOVES[t]
                i += 1
                if is_wm:
                    after = np.array(rest[i : i + 9]).reshape(3, 3)
                    i += 9
                else:
                    after = ep_apply(cur, mv)
                steps.append((mv, after))
                cur = after
            else:
                i += 1
        return start, goal, steps

    def ep_draw(ax, state, title):
        ax.imshow(np.where(state == 0, 1.0, 0.0), cmap="Greys", vmin=0, vmax=1)
        for r in range(3):
            for c in range(3):
                v = int(state[r, c])
                if v != 0:
                    ax.text(c, r, str(v), ha="center", va="center", fontsize=13, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9)

    return ep_draw, ep_parse


@app.cell
def _(arch, bw_draw, bw_parse, data, domain, ep_draw, ep_parse, idx, mo, plt):
    # ---- render the selected problem: start, goal, and the rollout ----
    rec = data[idx.value]
    seq = rec["sequence"]
    is_wm = arch.value == "wm"

    if domain.value == "blocks_world":
        start, goal, steps = bw_parse(seq, is_wm)
        draw = bw_draw
    else:
        start, goal, steps = ep_parse(seq, is_wm)
        draw = ep_draw

    # layout: row 1 = start + goal; following rows = rollout (6 per row)
    per_row = 6
    n_steps = len(steps)
    roll_rows = (n_steps + per_row - 1) // per_row if n_steps else 0
    fig, axes = plt.subplots(
        1 + roll_rows, per_row, figsize=(2.0 * per_row, 2.3 * (1 + roll_rows)), squeeze=False
    )
    for ax in axes.ravel():
        ax.axis("off")

    draw(axes[0][0], start, "START")
    axes[0][0].axis("on")
    draw(axes[0][1], goal, "GOAL")
    axes[0][1].axis("on")
    for k, (label, st) in enumerate(steps):
        r, c = 1 + k // per_row, k % per_row
        draw(axes[r][c], st, f"{k + 1}. {label}")
        axes[r][c].axis("on")
    fig.suptitle(
        f"{domain.value} / {arch.value} — problem {idx.value} "
        f"({rec.get('num_moves', '?')} moves, {rec.get('length', '?')} tokens)",
        fontsize=11,
    )
    fig.tight_layout()

    src = "stored in data" if is_wm else "replayed from moves"
    mo.vstack([mo.md(f"Intermediate states are **{src}**."), fig])
    return


@app.cell
def _(data, domain, idx, mo):
    # ---- raw + decoded token view ----
    rec2 = data[idx.value]
    raw = rec2["sequence"]
    if domain.value == "blocks_world":
        names = {0: "START", 1: "END", 2: "A", 3: "B", 4: "C", 5: "D",
                 6: "P0", 7: "P1", 8: "P2", 9: "P3", 10: "PAD"}
        dec = " ".join(names[t] for t in raw if t != 10)
    else:
        names = {10: "up", 11: "down", 12: "left", 13: "right", 14: "|", 15: "PAD"}
        dec = " ".join(names.get(t, str(t)) for t in raw if t != 15)
    mo.md(
        f"""
        **Raw tokens** (`{len(raw)}`): `{raw}`

        **Decoded:** `{dec}`
        """
    )
    return


if __name__ == "__main__":
    app.run()
