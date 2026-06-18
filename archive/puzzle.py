"""
Generate Varied-Length 8-Puzzle Test Set (FIXED)

Creates random 8-puzzle problems, solves them, then groups by optimal length.

This is more efficient than trying to generate specific lengths.

Usage:
    python3 generate_varied_8puzzle_FIXED.py
"""

import numpy as np
import json
from collections import deque, defaultdict
import random


def get_blank_pos(board):
    """Find position of blank (0) tile"""
    pos = np.argwhere(board == 0)
    return tuple(pos[0]) if len(pos) > 0 else None


def apply_move(board, move):
    """Apply a move and return new board (or None if invalid)"""
    board = board.copy()
    blank = get_blank_pos(board)
    if blank is None:
        return None
    
    row, col = blank
    
    if move == 'up' and row > 0:
        board[row, col], board[row-1, col] = board[row-1, col], board[row, col]
        return board
    elif move == 'down' and row < 2:
        board[row, col], board[row+1, col] = board[row+1, col], board[row, col]
        return board
    elif move == 'left' and col > 0:
        board[row, col], board[row, col-1] = board[row, col-1], board[row, col]
        return board
    elif move == 'right' and col < 2:
        board[row, col], board[row, col+1] = board[row, col+1], board[row, col]
        return board
    return None


def get_valid_moves(board):
    """Get all valid moves from current position"""
    blank = get_blank_pos(board)
    if blank is None:
        return []
    
    row, col = blank
    moves = []
    
    if row > 0: moves.append('up')
    if row < 2: moves.append('down')
    if col > 0: moves.append('left')
    if col < 2: moves.append('right')
    
    return moves


def count_inversions(board):
    """Count inversions in the board (for parity check)"""
    flat = board.flatten()
    tiles = [x for x in flat if x != 0]
    
    inversions = 0
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            if tiles[i] > tiles[j]:
                inversions += 1
    
    return inversions


def is_solvable(board):
    """Check if board is solvable to standard goal"""
    inversions = count_inversions(board)
    return inversions % 2 == 0


def solve_puzzle_bfs(start, goal, max_length=20):
    """
    Solve puzzle with BFS to find optimal solution
    
    Returns:
        moves: list of moves
        length: solution length
        intermediate_states: states after each move
    """
    if np.array_equal(start, goal):
        return [], 0, []
    
    # Check if solvable
    if not is_solvable(start):
        return None, None, None
    
    queue = deque([(start, [])])
    visited = {tuple(start.flatten())}
    
    while queue:
        current, moves = queue.popleft()
        
        if len(moves) >= max_length:
            continue
        
        for move in get_valid_moves(current):
            next_state = apply_move(current, move)
            if next_state is None:
                continue
            
            state_tuple = tuple(next_state.flatten())
            if state_tuple in visited:
                continue
            
            visited.add(state_tuple)
            new_moves = moves + [move]
            
            if np.array_equal(next_state, goal):
                # Reconstruct intermediate states
                intermediate_states = []
                state = start.copy()
                for m in new_moves:
                    state = apply_move(state, m)
                    intermediate_states.append(state.tolist())
                
                return new_moves, len(new_moves), intermediate_states
            
            queue.append((next_state, new_moves))
    
    return None, None, None


def generate_random_state(goal, num_moves=50):
    """
    Generate a random solvable state by applying random moves from goal
    
    Args:
        goal: Goal state
        num_moves: Number of random moves to apply (high enough to randomize)
    
    Returns:
        Random state (guaranteed solvable if goal is solvable)
    """
    current = goal.copy()
    
    for _ in range(num_moves):
        valid_moves = get_valid_moves(current)
        if not valid_moves:
            break
        
        move = random.choice(valid_moves)
        next_state = apply_move(current, move)
        if next_state is not None:
            current = next_state
    
    return current


def generate_varied_test_set(
    total_problems=1200,
    problems_per_length=200,
    length_range=(10, 15),
    max_solve_length=20,
    output_file='./outputs/15/8puzzle_varied_test_FIXED.json'
):
    """
    Generate test set with varied problem lengths
    
    Strategy:
    1. Generate many random problems
    2. Solve each to find optimal length
    3. Group by length
    4. Sample to get desired distribution
    
    Args:
        total_problems: Total problems to generate
        problems_per_length: Target number per length
        length_range: (min, max) optimal solution lengths
        max_solve_length: Maximum length to solve (20 is good for 8-puzzle)
        output_file: Where to save
    """
    
    print("="*70)
    print("GENERATE VARIED 8-PUZZLE TEST SET (FIXED)")
    print("="*70)
    
    goal = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    
    min_len, max_len = length_range
    
    print(f"\nTarget: {problems_per_length} problems per length")
    print(f"Lengths: {min_len} to {max_len} moves")
    print(f"Generating {total_problems} random problems...")
    print()
    
    # Generate random problems and solve them
    problems_by_length = defaultdict(list)
    
    generated = 0
    attempts = 0
    max_attempts = total_problems * 10  # Safety limit
    
    while generated < total_problems and attempts < max_attempts:
        attempts += 1
        
        # Generate random state
        start = generate_random_state(goal, num_moves=50)
        
        # Skip if too similar to goal
        if np.array_equal(start, goal):
            continue
        
        # Solve it
        solution, length, intermediate_states = solve_puzzle_bfs(
            start, goal, max_length=max_solve_length
        )
        
        if solution is None or length is None:
            continue  # Couldn't solve in max_length moves
        
        # Only keep problems in target range
        if min_len <= length <= max_len:
            problem = {
                'start_state': start.tolist(),
                'goal_state': goal.tolist(),
                'moves': solution,
                'intermediate_states': intermediate_states,
                'solution_length': length
            }
            
            problems_by_length[length].append(problem)
            generated += 1
            
            if generated % 100 == 0:
                print(f"  Generated {generated}/{total_problems}...")
    
    print(f"\n✓ Generated {generated} total problems")
    
    # Show distribution
    print("\n" + "="*70)
    print("GENERATED DISTRIBUTION")
    print("="*70)
    
    print(f"\n{'Length':<10} {'Count':<10} {'Status':<20}")
    print("-" * 50)
    
    for length in range(min_len, max_len + 1):
        count = len(problems_by_length[length])
        
        if count >= problems_per_length:
            status = "✓ Enough"
        elif count > 0:
            status = f"⚠️ Only {count}"
        else:
            status = "✗ None generated"
        
        print(f"{length:<10} {count:<10} {status:<20}")
    
    # Sample to get balanced distribution
    print("\n" + "="*70)
    print("BALANCING DISTRIBUTION")
    print("="*70)
    
    final_problems = []
    
    for length in range(min_len, max_len + 1):
        available = problems_by_length[length]
        
        if len(available) >= problems_per_length:
            # Sample randomly
            sampled = random.sample(available, problems_per_length)
            final_problems.extend(sampled)
            print(f"  {length} moves: sampled {problems_per_length}/{len(available)}")
        elif len(available) > 0:
            # Use all available
            final_problems.extend(available)
            print(f"  {length} moves: using all {len(available)} (wanted {problems_per_length})")
        else:
            print(f"  {length} moves: ✗ none available")
    
    # Shuffle
    random.shuffle(final_problems)
    
    # Save
    print(f"\n" + "="*70)
    print("SAVING")
    print(f"="*70)
    
    with open(output_file, 'w') as f:
        json.dump(final_problems, f, indent=2)
    
    print(f"\n✓ Saved {len(final_problems)} problems to: {output_file}")
    
    # Final distribution
    from collections import Counter
    final_lengths = Counter(p['solution_length'] for p in final_problems)
    
    print(f"\nFinal distribution:")
    for length in range(min_len, max_len + 1):
        count = final_lengths.get(length, 0)
        print(f"  {length} moves: {count} problems")
    
    print(f"\n" + "="*70)
    print("COMPLETE")
    print(f"="*70)
    
    print(f"\nYou can now run decay analysis with:")
    print(f"  python3 run_8puzzle_decay_analysis.py --test-problems {output_file}")
    
    return final_problems


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate varied-length 8-puzzle test set')
    parser.add_argument('--total', type=int, default=1500,
                       help='Total problems to generate (default: 1500)')
    parser.add_argument('--per-length', type=int, default=200,
                       help='Target problems per length (default: 200)')
    parser.add_argument('--min-length', type=int, default=10,
                       help='Minimum optimal length (default: 10)')
    parser.add_argument('--max-length', type=int, default=15,
                       help='Maximum optimal length (default: 15)')
    parser.add_argument('--output', default='8puzzle_varied_test_FIXED.json',
                       help='Output file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    problems = generate_varied_test_set(
        total_problems=args.total,
        problems_per_length=args.per_length,
        length_range=(args.min_length, args.max_length),
        output_file=args.output
    )
    
    print(f"\n✅ Test set ready!")
    print(f"   {len(problems)} total problems")
    print(f"   Lengths {args.min_length}-{args.max_length} moves")
