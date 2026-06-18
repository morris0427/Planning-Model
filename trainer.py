"""
Training and evaluation for planning models.

Extracted from train_blocks_world.py (user's working code) and adapted
to work with the experiment framework.

CRITICAL: Uses learning_rate=0.0001 (not 0.001) to prevent divergence.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import time
import numpy as np
from typing import Dict, Any

from config import ExperimentConfig
from data.base import DatasetFactory


class BlocksWorldDataset(Dataset):
    """
    PyTorch Dataset for sequence prediction.
    
    Extracted from user's working train_blocks_world.py.
    Handles both Blocks World and 8-Puzzle encoded sequences.
    """
    
    def __init__(self, data, max_seq_length, pad_token=9):
        """
        Args:
            data: List of problem dicts with 'sequence' key
            max_seq_length: Maximum sequence length
            pad_token: Token ID for padding (default 9 for Blocks World)
        """
        self.data = data
        self.max_seq_length = max_seq_length
        self.pad_token = pad_token
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns (input, target) pair for next-token prediction.
        
        Input:  [tok0, tok1, tok2, ..., tokN-1]
        Target: [tok1, tok2, tok3, ..., tokN]
        """
        sequence = self.data[idx]['sequence']
        
        # Pad sequence to max_seq_length
        padded = sequence + [self.pad_token] * (self.max_seq_length - len(sequence))
        padded = padded[:self.max_seq_length]
        
        # Create input/target pairs (shifted by 1)
        x = torch.tensor(padded[:-1], dtype=torch.long)
        y = torch.tensor(padded[1:], dtype=torch.long)
        
        return x, y


class PlanningTransformer(nn.Module):
    """
    Transformer model for planning tasks.
    Based on user's PuzzleTransformer from cot_comparison_experiment.py.
    """
    
    def __init__(self, vocab_size=15, d_model=128, nhead=4, num_layers=4, 
                 dim_feedforward=512, max_seq_length=100):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Embedding(max_seq_length, d_model)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Move mapping (for 8-puzzle compatibility)
        self.move_to_token = {'up': 9, 'down': 10, 'left': 11, 'right': 12}
        self.token_to_move = {9: 'up', 10: 'down', 11: 'left', 12: 'right'}
        self.PAD_TOKEN = 13
        self.SEP_TOKEN = 14
    
    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape
        
        # Token embeddings
        token_emb = self.embedding(x)
        
        # Positional embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_encoder(positions)
        
        # Combine embeddings
        emb = token_emb + pos_emb
        
        # Create causal mask
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Transformer
        output = self.transformer_decoder(emb, emb, tgt_mask=mask, memory_mask=mask)
        
        # Project to vocabulary
        logits = self.output_layer(output)
        
        return logits


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    vocab_size: int
) -> float:
    """
    Train for one epoch.
    
    Extracted from user's working train_blocks_world.py.
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        logits = model(batch_x)  # (batch, seq_len, vocab_size)
        
        # Reshape for loss computation
        logits = logits.reshape(-1, vocab_size)
        batch_y = batch_y.reshape(-1)
        
        # Compute loss
        loss = criterion(logits, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (from user's code)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def generate_solution(
    model: nn.Module,
    problem: Dict[str, Any],
    dataset_generator,
    device: torch.device,
    max_length: int = 50,
    return_info: bool = False,
    state_source: str = "oracle",
):
    """
    Generate a solution for a problem using the model.

    For baseline models: generates action tokens autoregressively. The
    state_source parameter is ignored (baseline sequences contain no
    per-step state tokens).

    For WM models:
      - state_source="oracle": after each action, the oracle computes
        the true post-action state and appends its encoded form to the
        context. The model's own state-token predictions are bypassed.
        Keeps the context grounded throughout the trajectory.
      - state_source="model": the model emits action AND state tokens
        autoregressively. The model's state predictions become its own
        input context for subsequent predictions.

    Args:
        model: Trained model
        problem: Problem dictionary with 'sequence' and 'num_moves'
        dataset_generator: Dataset instance for vocab/oracle access
        device: Device
        max_length: Maximum number of moves to generate
        return_info: If True, also return diagnostics dict
        state_source: "oracle" or "model" (WM only; ignored for baselines)

    Returns:
        Generated token list, or (tokens, info) if return_info=True.
        info_dict termination values:
          'solved_sep'        - reached goal, emitted SEP/END
          'sep'               - emitted SEP/END without solving
          'truncation_seqlen' - hit positional embedding capacity
          'invalid_move'      - generated a move illegal in current state
          'max_steps'         - WM wasted-prediction safety cap
          'max_length'        - hit max_length move cap / loop end
    """
    if state_source not in ("oracle", "model"):
        raise ValueError(
            f"state_source must be 'oracle' or 'model', got {state_source!r}"
        )

    model.eval()
    model_max_seq_length = model.max_seq_length

    sequence = problem['sequence']
    num_moves = problem['num_moves']
    domain_name = dataset_generator.__class__.__name__
    use_world_model = dataset_generator.use_world_model

    termination = 'max_length'
    moves_generated = 0

    is_eight_puzzle = 'EightPuzzle' in domain_name

    # ----- Domain-specific setup -----
    if is_eight_puzzle:
        state_length = 20  # 1 + 9 + 1 + 9
        end_token = 14     # SEP
        per_state_tokens = 9
        start_state = np.array(sequence[1:10]).reshape(3, 3)
        goal_state = np.array(sequence[11:20]).reshape(3, 3)
        move_tokens = {10: 'up', 11: 'down', 12: 'left', 13: 'right'}

        def oracle_step(cur_state, action_pieces):
            return apply_move_8puzzle(cur_state, move_tokens[action_pieces[0]])

        def encode_state(cur_state):
            return cur_state.flatten().tolist()

        def states_equal(a, b):
            return np.array_equal(a, b)

        def is_action_start_tok(tok):
            return tok in move_tokens

        action_tok_size = 1  # one token per move
    else:
        # Blocks World
        state_length = 17  # START(1) + start_state(8) + goal_state(8)
        end_token = 1      # END
        per_state_tokens = 8
        start_state = dataset_generator._decode_state(sequence[1:9])
        goal_state = dataset_generator._decode_state(sequence[9:17])

        BW_BLOCK_TOKENS = {2: 'A', 3: 'B', 4: 'C', 5: 'D'}
        BW_POS_TOKENS = {6: 0, 7: 1, 8: 2, 9: 3}

        def oracle_step(cur_state, action_pieces):
            block_tok, pos_tok = action_pieces
            if block_tok not in BW_BLOCK_TOKENS or pos_tok not in BW_POS_TOKENS:
                return None
            return dataset_generator.apply_action(
                cur_state, (BW_BLOCK_TOKENS[block_tok], BW_POS_TOKENS[pos_tok])
            )

        def encode_state(cur_state):
            return dataset_generator._encode_state(cur_state)

        def states_equal(a, b):
            return a == b

        def is_action_start_tok(tok):
            return tok in BW_BLOCK_TOKENS

        action_tok_size = 2  # block + pos

    # Initial context (start + goal)
    generated = sequence[:state_length].copy()

    # ----- WM + oracle path -----
    if use_world_model and state_source == "oracle":
        if is_eight_puzzle:
            current_state = start_state.copy()
        else:
            current_state = [tower[:] for tower in start_state]

        steps = 0
        max_steps = max_length * 20
        termination = 'max_length'

        with torch.no_grad():
            while moves_generated < max_length and steps < max_steps:
                steps += 1

                headroom = action_tok_size + per_state_tokens + 1
                if len(generated) >= model_max_seq_length - headroom:
                    termination = 'truncation_seqlen'
                    break

                # Collect a complete action (1 token for 8-puzzle, 2 for BW)
                action_pieces = []
                premature_sep = False
                while len(action_pieces) < action_tok_size:
                    input_seq = torch.tensor([generated], dtype=torch.long).to(device)
                    logits = model(input_seq)
                    next_tok = int(torch.argmax(logits[0, -1, :]).item())

                    if not action_pieces and next_tok == end_token:
                        # Model decided to stop right before producing an action
                        premature_sep = True
                        break

                    if not action_pieces and not is_action_start_tok(next_tok):
                        # Model emitted neither END nor a valid action-start
                        # token. Treat as garbage termination; do not append.
                        premature_sep = True
                        break

                    generated.append(next_tok)
                    action_pieces.append(next_tok)

                if premature_sep:
                    termination = 'sep'
                    break
                if len(action_pieces) < action_tok_size:
                    # Didn't finish a full action this pass; iterate again
                    continue

                # Oracle step
                next_state = oracle_step(current_state, action_pieces)
                if next_state is None:
                    termination = 'invalid_move'
                    break

                current_state = next_state
                moves_generated += 1

                # Inject the oracle-encoded post-action state into context.
                # This replaces whatever state tokens the model would have
                # produced.
                generated.extend(encode_state(current_state))

                if states_equal(current_state, goal_state):
                    generated.append(end_token)
                    termination = 'solved_sep'
                    break
            else:
                termination = 'max_steps' if steps >= max_steps else 'max_length'

    # ----- Pure-model path (baseline OR WM with state_source="model") -----
    else:
        termination = 'max_length'
        stride = action_tok_size + (per_state_tokens if use_world_model else 0)
        with torch.no_grad():
            # Bound the per-token loop so a WM emitting nothing useful still terminates
            # Same headroom as the oracle path so the two state_source modes
            # truncate at the same effective boundary
            headroom = action_tok_size + (per_state_tokens if use_world_model else 0) + 1
            for _ in range(max_length * max(1, stride + 1)):
                if len(generated) >= model_max_seq_length - headroom:
                    termination = 'truncation_seqlen'
                    break

                input_seq = torch.tensor([generated], dtype=torch.long).to(device)
                logits = model(input_seq)
                next_tok = int(torch.argmax(logits[0, -1, :]).item())
                generated.append(next_tok)

                if next_tok == end_token:
                    termination = 'sep'
                    break

                approx_moves = max(0, (len(generated) - state_length)) // max(1, stride)
                if approx_moves >= max_length:
                    moves_generated = approx_moves
                    termination = 'max_length'
                    break
                moves_generated = approx_moves

    if return_info:
        info = {
            'termination': termination,
            'moves_generated': moves_generated,
            'final_len': len(generated),
            'model_max_seq_length': model_max_seq_length,
            'state_source': state_source if use_world_model else 'n/a',
        }
        return generated, info
    return generated


def apply_move_8puzzle(state: np.ndarray, move: str) -> np.ndarray:
    """
    Apply a move to an 8-puzzle state.
    
    Args:
        state: 3x3 numpy array
        move: 'up', 'down', 'left', or 'right'
    
    Returns:
        New state after move, or None if invalid
    """
    state = state.copy()
    
    # Find blank (0) position
    blank_pos = np.argwhere(state == 0)
    if len(blank_pos) == 0:
        return None
    
    row, col = blank_pos[0]
    
    # Apply move
    if move == 'up' and row > 0:
        state[row, col], state[row-1, col] = state[row-1, col], state[row, col]
        return state
    elif move == 'down' and row < 2:
        state[row, col], state[row+1, col] = state[row+1, col], state[row, col]
        return state
    elif move == 'left' and col > 0:
        state[row, col], state[row, col-1] = state[row, col-1], state[row, col]
        return state
    elif move == 'right' and col < 2:
        state[row, col], state[row, col+1] = state[row, col+1], state[row, col]
        return state
    
    return None  # Invalid move


def check_solution_correctness(
    generated_tokens: list[int],
    problem: Dict[str, Any],
    dataset_generator
) -> bool:
    """
    Check if generated solution actually solves the problem.

    For 8-Puzzle: applies moves to the start state and checks goal reach.
    For Blocks World: applies actions to the start state and checks goal reach
                      (semantic correctness, not byte-equality with reference).

    A generated trajectory is correct iff its action sequence is legal at
    every step and the final state equals the goal. State predictions in
    the generated tokens (under WM) are NOT required to match the reference;
    only the actions must lead to the goal.

    Args:
        generated_tokens: Generated token sequence
        problem: Original problem with ground truth sequence
        dataset_generator: Dataset for decoding/applying actions

    Returns:
        True if solution is correct
    """
    try:
        gt_sequence = problem['sequence']
        num_moves = problem['num_moves']
        domain_name = dataset_generator.__class__.__name__

        if 'EightPuzzle' in domain_name:
            # 8-puzzle layout: [dummy(1), start_state(9), PAD(1), goal_state(9), moves, SEP]
            start_state = np.array(gt_sequence[1:10]).reshape(3, 3)
            goal_state = np.array(gt_sequence[11:20]).reshape(3, 3)

            move_tokens = {10: 'up', 11: 'down', 12: 'left', 13: 'right'}
            gen_moves = []
            for token in generated_tokens[20:]:
                if token == 14:  # SEP
                    break
                if token in move_tokens:
                    gen_moves.append(move_tokens[token])

            current_state = start_state.copy()
            for move in gen_moves:
                blank_pos = np.argwhere(current_state == 0)
                if len(blank_pos) == 0:
                    return False
                row, col = blank_pos[0]
                if move == 'up' and row > 0:
                    current_state[row, col], current_state[row-1, col] = \
                        current_state[row-1, col], current_state[row, col]
                elif move == 'down' and row < 2:
                    current_state[row, col], current_state[row+1, col] = \
                        current_state[row+1, col], current_state[row, col]
                elif move == 'left' and col > 0:
                    current_state[row, col], current_state[row, col-1] = \
                        current_state[row, col-1], current_state[row, col]
                elif move == 'right' and col < 2:
                    current_state[row, col], current_state[row, col+1] = \
                        current_state[row, col+1], current_state[row, col]
                else:
                    return False
            return np.array_equal(current_state, goal_state)

        else:
            # Blocks World layout (uniform encoding):
            #   [START(1), start_state(8), goal_state(8), then per-step blocks, END(1)]
            #   Baseline per-step: action(2 tokens)
            #   WM per-step:       action(2 tokens) + state(8 tokens)
            CONTEXT_LEN = 17     # START + start_state + goal_state
            STATE_LEN = 8
            ACTION_LEN = 2
            END_TOKEN = 1
            BLOCK_TOKS = {2: 'A', 3: 'B', 4: 'C', 5: 'D'}
            POS_TOKS = {6: 0, 7: 1, 8: 2, 9: 3}

            if len(generated_tokens) < CONTEXT_LEN + ACTION_LEN:
                # Didn't generate even one action
                return False

            # Decode start and goal from the reference sequence's lossless
            # state blocks. Note: we use the reference (ground truth)
            # sequence's context, not the generated tokens', since the
            # context tokens are always set from the problem.
            try:
                start_state = dataset_generator._decode_state(
                    gt_sequence[1:1 + STATE_LEN]
                )
                goal_state = dataset_generator._decode_state(
                    gt_sequence[1 + STATE_LEN:CONTEXT_LEN]
                )
            except Exception:
                # If decoding fails, fall back to declaring incorrect
                return False

            use_wm = dataset_generator.use_world_model
            stride = ACTION_LEN + (STATE_LEN if use_wm else 0)

            current_state = [tower[:] for tower in start_state]
            pos = CONTEXT_LEN

            while pos < len(generated_tokens):
                if generated_tokens[pos] == END_TOKEN:
                    break
                # Need at least an action pair from here
                if pos + ACTION_LEN > len(generated_tokens):
                    return False
                block_tok = generated_tokens[pos]
                pos_tok = generated_tokens[pos + 1]
                if block_tok not in BLOCK_TOKS or pos_tok not in POS_TOKS:
                    # Malformed action (state tokens in an action slot, etc.)
                    return False
                action = (BLOCK_TOKS[block_tok], POS_TOKS[pos_tok])
                next_state = dataset_generator.apply_action(current_state, action)
                if next_state is None:
                    # Illegal action (block not on top of any tower)
                    return False
                current_state = next_state
                pos += stride

            return current_state == goal_state

    except Exception:
        # If any error occurs in checking, declare incorrect rather than
        # raising. This matches the previous function's behavior.
        return False


def evaluate_solve_rate(
    model: nn.Module,
    test_problems: list[Dict],
    dataset_generator,
    device: torch.device,
    max_samples: int = 100
) -> Dict[str, float]:
    """
    Evaluate actual solve rate on test problems.
    
    Args:
        model: Trained model
        test_problems: List of test problems
        dataset_generator: Dataset for encoding/decoding
        device: Device
        max_samples: Maximum problems to evaluate (for speed)
    
    Returns:
        Dictionary with solve rate metrics
    """
    model.eval()
    
    num_to_test = min(len(test_problems), max_samples)
    solved = 0
    
    # Per-problem records for diagnostics
    records = []  # each: (num_moves, moves_generated, termination, solved, final_len, model_max)
    
    print(f"  Evaluating solve rate on {num_to_test} test problems...")
    
    with torch.no_grad():
        for i, problem in enumerate(test_problems[:num_to_test]):
            if (i + 1) % 20 == 0:
                print(f"    Progress: {i+1}/{num_to_test} ({solved}/{i+1} solved)")
            
            # Generate solution (with diagnostics)
            generated, info = generate_solution(
                model, problem, dataset_generator, device,
                max_length=100, return_info=True
            )
            
            # Check if correct
            is_solved = check_solution_correctness(generated, problem, dataset_generator)
            if is_solved:
                solved += 1
            
            records.append((
                problem.get('num_moves', -1),
                info['moves_generated'],
                info['termination'],
                is_solved,
                info['final_len'],
                info['model_max_seq_length'],
            ))
    
    solve_rate = solved / num_to_test
    
    print(f"  ✓ Solve rate: {solved}/{num_to_test} = {solve_rate:.1%}")
    
    # --- Diagnostic 1: solve rate stratified by reference solution length ---
    from collections import defaultdict
    by_len_total = defaultdict(int)
    by_len_solved = defaultdict(int)
    by_len_gen = defaultdict(list)
    for num_moves, moves_gen, term, is_solved, final_len, mmax in records:
        by_len_total[num_moves] += 1
        by_len_solved[num_moves] += int(is_solved)
        by_len_gen[num_moves].append(moves_gen)
    
    print("  ── Solve rate by reference solution length ──")
    print(f"    {'len':>4} {'n':>4} {'solved':>7} {'rate':>6} {'avg_gen_moves':>14}")
    for L in sorted(by_len_total):
        n = by_len_total[L]
        s = by_len_solved[L]
        avg_gen = sum(by_len_gen[L]) / len(by_len_gen[L]) if by_len_gen[L] else 0
        print(f"    {L:>4} {n:>4} {s:>7} {s/n:>6.0%} {avg_gen:>14.1f}")
    
    # --- Diagnostic 2: termination reason counts ---
    term_counts = defaultdict(int)
    term_solved = defaultdict(int)
    for num_moves, moves_gen, term, is_solved, final_len, mmax in records:
        term_counts[term] += 1
        term_solved[term] += int(is_solved)
    
    print("  ── Termination reasons ──")
    for term in sorted(term_counts):
        c = term_counts[term]
        s = term_solved[term]
        print(f"    {term:<20} {c:>4}  (solved: {s})")
    
    # Flag truncation prominently — this is the artifact we are testing for
    trunc = term_counts.get('truncation_seqlen', 0)
    if trunc > 0:
        model_max = records[0][5] if records else 0
        print(f"  ⚠️  {trunc}/{num_to_test} generations hit the seq-length truncation cap "
              f"(model_max_seq_length={model_max}).")
        print(f"      If these cluster at the longest reference lengths, low solve rate "
              f"is partly a truncation artifact, not a planning result.")
    
    return {
        'solve_rate': solve_rate,
        'problems_solved': solved,
        'problems_tested': num_to_test,
        'solve_by_length': {int(L): {'n': by_len_total[L], 'solved': by_len_solved[L]}
                            for L in by_len_total},
        'termination_counts': dict(term_counts),
        'truncation_count': trunc,
    }


def evaluate(
    model: nn.Module,
    test_dataset: BlocksWorldDataset,
    device: torch.device,
    vocab_size: int,
    pad_token: int
) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Returns:
        Dictionary with metrics:
        - test_loss: Cross-entropy loss
    """
    model.eval()
    
    # Create test dataloader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
    criterion_per_token = nn.CrossEntropyLoss(ignore_index=pad_token, reduction='none')
    
    total_loss = 0
    num_batches = 0
    
    # For separating action vs state losses (both domains)
    action_losses = []
    state_losses = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits = model(batch_x)
            
            # Overall loss
            logits_flat = logits.reshape(-1, vocab_size)
            batch_y_flat = batch_y.reshape(-1)
            loss = criterion(logits_flat, batch_y_flat)
            total_loss += loss.item()
            num_batches += 1
            
            # Per-token loss for action/state separation
            per_token_loss = criterion_per_token(logits_flat, batch_y_flat)
            
            # Categorize losses by token type (domain-specific)
            for i, (token_id, token_loss) in enumerate(zip(batch_y_flat, per_token_loss)):
                token_id = token_id.item()
                token_loss_val = token_loss.item()
                
                # 8-Puzzle: actions (10-13), states (0-8)
                if token_id in [10, 11, 12, 13]:
                    action_losses.append(token_loss_val)
                elif token_id in range(0, 9):
                    state_losses.append(token_loss_val)
                # Blocks World: actions (2-7), states (would need similar categorization)
                # For now, this primarily works for 8-Puzzle
    
    avg_loss = total_loss / num_batches
    
    # Calculate separate losses if we have data
    result = {'test_loss': avg_loss}
    
    if action_losses and state_losses:
        result['action_loss'] = sum(action_losses) / len(action_losses)
        result['state_loss'] = sum(state_losses) / len(state_losses)
        result['num_action_tokens'] = len(action_losses)
        result['num_state_tokens'] = len(state_losses)
        result['state_action_ratio'] = len(state_losses) / len(action_losses) if action_losses else 0
    
    return result


# =============================================================================
# DATA CACHING FUNCTIONS (to avoid regenerating data for each experiment)
# =============================================================================

def save_generated_data(train_problems, test_problems, config, use_wm):
    """Save generated data to disk for reuse across experiments.

    Also writes a {filename}.meta.json sidecar so the next load can show
    the user exactly what is in the cache.
    """
    cache_dir = Path("cached_data")
    cache_dir.mkdir(exist_ok=True)
    
    # Create filename based on config
    domain = config.data.domain
    wm_suffix = "_wm" if use_wm else "_baseline"
    # Include split type so productivity runs do not collide with in-distribution
    split_suffix = "_productivity" if config.data.split_type.value == "productivity" else ""
    
    train_file = cache_dir / f"{domain}_train{wm_suffix}{split_suffix}.json"
    test_file = cache_dir / f"{domain}_test{wm_suffix}{split_suffix}.json"
    
    # Save
    with open(train_file, 'w') as f:
        json.dump(train_problems, f)
    
    with open(test_file, 'w') as f:
        json.dump(test_problems, f)
    
    # Metadata sidecar (purely informational; trainer never reads this
    # to decide whether to use the cache)
    import time as _time
    train_ranges, test_ranges = config.data.get_split_ranges()
    meta = {
        "created_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        "domain": domain,
        "use_world_model": use_wm,
        "split_type": config.data.split_type.value,
        "train": {
            "difficulty_range": list(train_ranges),
            "num_samples": len(train_problems),
            "num_moves_min": min(p["num_moves"] for p in train_problems),
            "num_moves_max": max(p["num_moves"] for p in train_problems),
        },
        "test": {
            "difficulty_range": list(test_ranges),
            "num_samples": len(test_problems),
            "num_moves_min": min(p["num_moves"] for p in test_problems),
            "num_moves_max": max(p["num_moves"] for p in test_problems),
        },
    }
    for path in (train_file, test_file):
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"💾 SAVED DATA FOR REUSE")
    print(f"{'='*70}")
    print(f"Train: {train_file}")
    print(f"Test:  {test_file}")
    print(f"Metadata sidecars written alongside.")
    print(f"This data will be reused for remaining experiments!")
    print(f"{'='*70}\n")
    
    return train_file, test_file


def load_cached_data(config, use_wm, force_regenerate=False):
    """Try to load cached data if it exists.

    If force_regenerate is True, skip the lookup and return (None, None)
    so the caller regenerates from scratch.

    On a cache hit, reads the {filename}.meta.json sidecar (if present)
    and prints its contents. Warns if recorded split_type or difficulty
    range disagrees with the current config.
    """
    if force_regenerate:
        print(f"\n{'='*70}")
        print(f"🔁 FORCE REGENERATE: skipping any cached data")
        print(f"{'='*70}\n")
        return None, None
    
    cache_dir = Path("cached_data")
    domain = config.data.domain
    
    wm_suffix = "_wm" if use_wm else "_baseline"
    # Include split type so productivity runs do not collide with in-distribution
    split_suffix = "_productivity" if config.data.split_type.value == "productivity" else ""
    
    train_file = cache_dir / f"{domain}_train{wm_suffix}{split_suffix}.json"
    test_file = cache_dir / f"{domain}_test{wm_suffix}{split_suffix}.json"
    
    if train_file.exists() and test_file.exists():
        print(f"\n{'='*70}")
        print(f"📂 LOADING CACHED DATA (Skipping generation!)")
        print(f"{'='*70}")
        print(f"Train: {train_file}")
        print(f"Test:  {test_file}")
        meta_path = train_file.with_suffix(".meta.json")
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                print(f"Cache metadata:")
                print(f"  created:       {meta.get('created_utc', '?')}")
                print(f"  domain:        {meta.get('domain', '?')}")
                print(f"  split_type:    {meta.get('split_type', '?')}")
                print(f"  world_model:   {meta.get('use_world_model', '?')}")
                tr = meta.get("train", {})
                te = meta.get("test", {})
                print(f"  train: range={tr.get('difficulty_range','?')} "
                      f"samples={tr.get('num_samples','?')} "
                      f"num_moves=[{tr.get('num_moves_min','?')}..{tr.get('num_moves_max','?')}]")
                print(f"  test:  range={te.get('difficulty_range','?')} "
                      f"samples={te.get('num_samples','?')} "
                      f"num_moves=[{te.get('num_moves_min','?')}..{te.get('num_moves_max','?')}]")
                cur_split = config.data.split_type.value
                if meta.get("split_type") and meta["split_type"] != cur_split:
                    print(f"  ⚠️  cache split_type ({meta['split_type']}) "
                          f"differs from current config ({cur_split})")
                req_train_range, req_test_range = config.data.get_split_ranges()
                if tr.get("difficulty_range") and list(tr["difficulty_range"]) != list(req_train_range):
                    print(f"  ⚠️  cached train range {tr['difficulty_range']} "
                          f"differs from current {list(req_train_range)}")
                if te.get("difficulty_range") and list(te["difficulty_range"]) != list(req_test_range):
                    print(f"  ⚠️  cached test range {te['difficulty_range']} "
                          f"differs from current {list(req_test_range)}")
            except Exception as e:
                print(f"  (could not read metadata sidecar: {e})")
        else:
            print(f"  (no metadata sidecar; cache predates the metadata feature)")
        print(f"{'='*70}\n")
        
        with open(train_file, 'r') as f:
            train_problems = json.load(f)
        
        with open(test_file, 'r') as f:
            test_problems = json.load(f)
        
        return train_problems, test_problems
    
    return None, None


def train(config: ExperimentConfig, force_regenerate: bool = False) -> Dict[str, Any]:
    """
    Main training function.
    
    Extracted from user's train_blocks_world.py and adapted
    to use the experiment framework configuration.
    
    CRITICAL: Uses learning_rate=0.0001 (not 0.001) to prevent divergence.
    
    Args:
        config: Experiment configuration
        force_regenerate: If True, skip any cached data and regenerate.
            Default False preserves the cache-when-present behavior.
    
    Returns:
        Dictionary with results
    """
    print("=" * 70)
    print(f"TRAINING: {config.experiment_name}")
    print("=" * 70)
    print(config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Generate datasets using factory
    print("\nLoading/Generating data...")
    train_ranges, test_ranges = config.data.get_split_ranges()
    
    # Try to load cached data first (saves hours for 8-puzzle!)
    use_wm = config.model.use_world_model
    train_problems, test_problems = load_cached_data(config, use_wm, force_regenerate=force_regenerate)
    
    if train_problems is None:
        # Cache miss - generate fresh data
        print("Cache miss - generating fresh data...")
        
        train_dataset_generator = DatasetFactory.create(
            domain=config.data.domain,
            difficulty_range=train_ranges,
            num_samples=config.data.num_train_samples,
            use_world_model=config.model.use_world_model,
            seed=config.seed
        )
        train_problems = train_dataset_generator.generate_dataset()
        
        test_dataset_generator = DatasetFactory.create(
            domain=config.data.domain,
            difficulty_range=test_ranges,
            num_samples=config.data.num_test_samples,
            use_world_model=config.model.use_world_model,
            seed=config.seed + 1  # Different seed for test
        )
        test_problems = test_dataset_generator.generate_dataset()
        
        # Save for next experiments (saves HOURS for 8-puzzle!)
        save_generated_data(train_problems, test_problems, config, use_wm)
    else:
        # Cache hit - just create generators for vocab info
        print("Cache hit - data loaded successfully!")
        train_dataset_generator = DatasetFactory.create(
            domain=config.data.domain,
            difficulty_range=train_ranges,
            num_samples=1,  # Just need one for vocab
            use_world_model=config.model.use_world_model,
            seed=config.seed
        )
        test_dataset_generator = DatasetFactory.create(
            domain=config.data.domain,
            difficulty_range=test_ranges,
            num_samples=1,  # Just need one for vocab
            use_world_model=config.model.use_world_model,
            seed=config.seed + 1
        )
    
    print(f"✓ Train: {len(train_problems)} problems")
    print(f"✓ Test:  {len(test_problems)} problems")

    
    # Get vocabulary and sequence length info
    vocab_size = train_dataset_generator.get_vocab_size()
    
    # Use MAXIMUM of train and test seq lengths (for productivity splits)
    train_max_seq = train_dataset_generator.get_max_sequence_length()
    test_max_seq = test_dataset_generator.get_max_sequence_length()
    max_seq_length = max(train_max_seq, test_max_seq)
    
    if train_max_seq != test_max_seq:
        print(f"  Note: Train max_seq={train_max_seq}, Test max_seq={test_max_seq}")
        print(f"        Using max={max_seq_length} for model capacity")
    
    # Pad token (domain-specific)
    # Blocks World: PAD = 9 (based on vocab in blocks_world.py: PAD=10, but ignore_index uses 9)
    # Note: This should match your actual PAD token
    pad_token = 10 if config.data.domain == "blocks_world" else 0
    
    print(f"✓ Vocabulary size: {vocab_size}")
    print(f"✓ Max sequence length: {max_seq_length}")
    print(f"✓ Pad token: {pad_token}")
    
    # Create PyTorch datasets using user's BlocksWorldDataset class
    train_dataset = BlocksWorldDataset(train_problems, max_seq_length, pad_token)
    test_dataset = BlocksWorldDataset(test_problems, max_seq_length, pad_token)
    
    # Create dataloader (batch_size=32 from user's code)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.model.batch_size,  # Default is 32
        shuffle=True
    )
    
    # Create model (matching user's architecture from train_blocks_world.py)
    print("\nInitializing model...")
    model = PlanningTransformer(
        vocab_size=vocab_size,
        d_model=config.model.d_model,        # 128 for small
        nhead=config.model.n_heads,          # 4
        num_layers=config.model.n_layers,    # 4
        dim_feedforward=config.model.d_ff,   # 512
        max_seq_length=max_seq_length
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {config.model.name}")
    print(f"✓ Parameters: {num_params:,}")
    
    # Setup optimizer and loss (from user's code)
    # CRITICAL: learning_rate=0.0001 (not 0.001!)
    optimizer = optim.Adam(model.parameters(), lr=config.model.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {config.model.max_epochs}")
    print(f"  Learning rate: {config.model.learning_rate} ✓")
    print(f"  Batch size: {config.model.batch_size}")
    print(f"  World model: {config.model.use_world_model}")
    if config.model.learning_rate != 0.0001:
        print(f"  ⚠️  WARNING: lr != 0.0001 may cause divergence!")
    
    # Training loop (from user's code)
    print(f"\nTraining for {config.model.max_epochs} epochs...")
    print("Expected: Loss should DECREASE consistently\n")
    
    train_losses = []
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(config.model.max_epochs):
        # Train
        epoch_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, vocab_size
        )
        train_losses.append(epoch_loss)
        
        # Track best
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            
            # Save checkpoint if enabled
            if config.save_checkpoint:
                save_dir = Path(config.save_dir) / config.experiment_name
                save_dir.mkdir(parents=True, exist_ok=True)
                
                checkpoint_path = save_dir / "best_model.pth"
                torch.save(model.state_dict(), checkpoint_path)
        
        # Print progress (matching user's format)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            
            if epoch == 0:
                trend = ""
            elif epoch_loss < train_losses[max(0, epoch - 10)]:
                trend = "✓ decreasing"
            else:
                trend = "✗ increasing (check learning rate!)"
            
            print(f"  Epoch [{epoch+1:>3}/{config.model.max_epochs}], "
                  f"Loss: {epoch_loss:.4f} {trend}, "
                  f"Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    
    # Final evaluation
    print("\nEvaluating on test set...")
    test_results = evaluate(model, test_dataset, device, vocab_size, pad_token)
    
    # Evaluate solve rate (actual problem-solving ability)
    print("\nEvaluating solve rate...")
    solve_rate_results = evaluate_solve_rate(
        model, test_problems, test_dataset_generator, device, max_samples=100
    )
    test_results.update(solve_rate_results)
    
    # Print summary (matching user's format)
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Initial loss: {train_losses[0]:.4f}")
    print(f"Final loss:   {train_losses[-1]:.4f}")
    print(f"Best loss:    {best_loss:.4f}")
    print(f"Test loss:    {test_results['test_loss']:.4f}")
    
    # Print action/state breakdown if available
    if 'action_loss' in test_results and 'state_loss' in test_results:
        print(f"  ↳ Action loss: {test_results['action_loss']:.4f} (planning)")
        print(f"  ↳ State loss:  {test_results['state_loss']:.4f} (dynamics complexity)")
        print(f"  ↳ State/Action ratio: {test_results['state_action_ratio']:.1f}:1 tokens")
    
    print(f"Solve rate:   {test_results['solve_rate']:.1%} ({test_results['problems_solved']}/{test_results['problems_tested']})")
    
    if train_losses[-1] < train_losses[0]:
        improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
        print(f"✓ Loss decreased by {improvement:.1f}%")
    else:
        print(f"✗ Loss INCREASED - check configuration!")
    
    print(f"\nTotal time: {total_time:.1f}s ({total_time/config.model.max_epochs:.2f}s per epoch)")
    
    # Save training history
    save_dir = Path(config.save_dir) / config.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    history = {
        'train_losses': train_losses,
        'best_loss': best_loss,
        'total_time': total_time,
        'config': {
            'learning_rate': config.model.learning_rate,
            'epochs': config.model.max_epochs,
            'vocab_size': vocab_size,
            'model': config.model.name,
        }
    }
    
    history_path = save_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nSaved:")
    if config.save_checkpoint:
        print(f"  Model: {save_dir / 'best_model.pth'}")
    print(f"  History: {history_path}")
    
    # Return results for experiment tracking
    results = {
        'experiment_name': config.experiment_name,
        'domain': config.data.domain,
        'model': config.model.name,
        'split_type': config.data.split_type.value,
        'train_loss_initial': train_losses[0],
        'train_loss_final': train_losses[-1],
        'train_loss_best': best_loss,
        'test_loss': test_results['test_loss'],
        'solve_rate': test_results.get('solve_rate', 0.0),
        'problems_solved': test_results.get('problems_solved', 0),
        'problems_tested': test_results.get('problems_tested', 0),
        'training_time': total_time,
        'num_parameters': num_params,
        'converged': train_losses[-1] < train_losses[0],
    }
    
    # Add action/state breakdown if available
    if 'action_loss' in test_results:
        results['action_loss'] = test_results['action_loss']
        results['state_loss'] = test_results['state_loss']
        results['num_action_tokens'] = test_results['num_action_tokens']
        results['num_state_tokens'] = test_results['num_state_tokens']
        results['state_action_ratio'] = test_results['state_action_ratio']
    
    return results


if __name__ == "__main__":
    # Test the trainer
    print("Testing trainer...")
    
    from config import ExperimentConfig, ModelPresets, DataPresets
    
    # Create test config
    config = ExperimentConfig(
        model=ModelPresets.small(use_world_model=False),
        data=DataPresets.blocks_world_standard(),
        experiment_name="trainer_test"
    )
    
    # Override to run quick test
    config.data.num_train_samples = 100
    config.data.num_test_samples = 20
    config.model.max_epochs = 5
    
    print("\nRunning quick test with 100 samples, 5 epochs...")
    results = train(config)
    
    print("\n✓ Trainer test complete!")
    print(f"Final results: {results}")
