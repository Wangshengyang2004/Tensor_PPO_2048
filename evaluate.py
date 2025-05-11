# evaluate.py
import sys
import torch
import pygame
import argparse
import random
from vector2048_env import Vector2048Env
from ppo2048 import PPOAgent
from dqn2048 import DQN
from mpc2048 import mpc_policy

# Configuration
grid_size = 4
TILE_SIZE = 120  # pixels per tile
WINDOW_SIZE = grid_size * TILE_SIZE
FPS = 2

# Color mapping for tiles
TILE_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218), 4: (237, 224, 200), 8: (242, 177, 121), 16: (245, 149, 99),
    32: (246, 124, 95), 64: (246, 94, 59), 128: (237, 207, 114), 256: (237, 204, 97),
    512: (237, 200, 80), 1024: (237, 197, 63), 2048: (237, 194, 46)
}

# Text colors
TEXT_COLOR_LIGHT = (249, 246, 242)
TEXT_COLOR_DARK = (119, 110, 101)

# Action name mapping for display
ACTION_NAMES = {0: 'Up', 1: 'Right', 2: 'Down', 3: 'Left'}


def draw_board(screen, grid, font):
    for i in range(grid_size):
        for j in range(grid_size):
            val = grid[i, j]
            color = TILE_COLORS.get(val, TILE_COLORS[0])
            rect = pygame.Rect(j * TILE_SIZE, i * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (187, 173, 160), rect, 2)
            if val != 0:
                text_surf = font.render(str(val), True,
                                        TEXT_COLOR_DARK if val <= 4 else TEXT_COLOR_LIGHT)
                text_rect = text_surf.get_rect(center=rect.center)
                screen.blit(text_surf, text_rect)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', choices=['ppo','random','mpc','dqn'], default='ppo')
    parser.add_argument('--ppo-model', type=str, default='models/ppo2048_final.pth', help='Path to trained PPO model file')
    parser.add_argument('--dqn-model', type=str, help='Path to trained DQN .pth file')
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--samples', type=int, default=512)
    parser.add_argument('--device', choices=['cpu','cuda','mps','auto'], default='auto')
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--render-mode', choices=['terminal','pygame'], default='terminal')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for action sampling (0.0 for argmax)')
    parser.add_argument('--max-tile', type=int, default=2048, help='Tile value that counts as a win')
    args = parser.parse_args()

    # Setup device (only CUDA or CPU; ignore MPS)
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    elif args.device == 'mps':
        print("Warning: MPS not supported, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
        
    env = Vector2048Env(1, grid_size, device)

    # Prepare policies
    try:
        if args.policy == 'ppo':
            from ppo2048 import PPOAgent
            # Create the PPO agent (compatible with both old and new architecture)
            try:
                # Try the new architecture first (default parameters)
                agent = PPOAgent(grid_size=grid_size, num_envs=1).to(device)
            except TypeError:
                # Fall back to the old architecture if needed
                agent = PPOAgent(grid_size, 1).to(device)
                
            # Load model, handling DDP-trained models by removing 'module.' prefix
            try:
                state_dict = torch.load(args.ppo_model, map_location=device)
            except FileNotFoundError:
                print(f"Error: Model file '{args.ppo_model}' not found.")
                return
            except Exception as e:
                print(f"Error loading model: {e}")
                return
                
            # Check if this is a DDP-trained model (keys have 'module.' prefix)
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            try:
                agent.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading state dict: {e}")
                return
            agent.eval()
        elif args.policy == 'dqn':
            try:
                net = DQN(grid_size).to(device)
                net.load_state_dict(torch.load(args.dqn_model, map_location=device))
                net.eval()
            except FileNotFoundError:
                print(f"Error: Model file '{args.dqn_model}' not found.")
                return
            except Exception as e:
                print(f"Error loading DQN model: {e}")
                return
    except Exception as e:
        print(f"Error initializing policy: {e}")
        return

    # unified eval loop with selectable render mode
    state, done = env.reset()
    score = 0.0; step = 0
    rewards_hist = []
    max_tiles_hist = []
    
    # Variables to track stagnation
    no_reward_steps = 0
    stagnation_threshold = 300
    last_max_tile = 0
    max_tile_unchanged_steps = 0
    max_tile_threshold = 1000
    last_state = None
    repeated_states = 0
    repeat_threshold = 100
    
    # prepare pygame if needed
    screen = None
    font = None
    header_font = None
    clock = None
    if args.render_mode == 'pygame':
        try:
            pygame.init()
            screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            pygame.display.set_caption(f"2048 {args.policy.upper()} ({args.render_mode})")
            font = pygame.font.SysFont('arial', TILE_SIZE//4, bold=True)
            header_font = pygame.font.SysFont('arial', TILE_SIZE//6, bold=True)
            clock = pygame.time.Clock()
        except pygame.error as e:
            print(f"Error initializing pygame: {e}")
            return
    
    try:
        # main stepping loop
        while not done and step < args.max_steps:
            # Handle pygame events
            if args.render_mode == 'pygame':
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        raise KeyboardInterrupt

            # select action for any policy
            if args.policy == 'ppo':
                with torch.no_grad():
                    # Try the new interface first
                    try:
                        action, _, _, _ = agent.get_action_and_value(state)
                    except (AttributeError, TypeError):
                        # Fall back to old interface
                        logits, _ = agent(state)
                        # Use temperature sampling instead of argmax for exploration
                        if args.temperature <= 0.0:
                            # Greedy (argmax) selection
                            action = torch.argmax(logits[0])
                        else:
                            # Temperature sampling
                            probs = torch.softmax(logits[0] / args.temperature, dim=0)
                            action = torch.multinomial(probs, 1)[0]
            elif args.policy == 'dqn':
                action = dqn_policy(env, state, net, device)
            elif args.policy == 'mpc':
                action = mpc_policy(env, state, args.horizon, args.samples, device)
            else:
                action = random_policy(env, state)
            # convert to Python int and get name
            action_val = action.item() if isinstance(action, torch.Tensor) else int(action)
            action_name = ACTION_NAMES.get(action_val, str(action_val))
            # step env
            state, reward, done = env.step(torch.tensor([action_val], dtype=torch.int64, device=device))
            step += 1
            score += float(reward.item())
            rewards_hist.append(float(reward.item()))
            maxt = int(state.max().item())
            max_tiles_hist.append(maxt)
            
            # Track stagnation metrics
            if reward.item() == 0:
                no_reward_steps += 1
            else:
                no_reward_steps = 0
            
            if maxt == last_max_tile:
                max_tile_unchanged_steps += 1
            else:
                max_tile_unchanged_steps = 0
                last_max_tile = maxt
            
            # Check for repeated states
            if last_state is not None and torch.all(state == last_state):
                repeated_states += 1
            else:
                repeated_states = 0
            last_state = state.clone()
            
            # Exit if stagnating (no progress)
            stagnating = (no_reward_steps >= stagnation_threshold or 
                         max_tile_unchanged_steps >= max_tile_threshold or
                         repeated_states >= repeat_threshold)
            if stagnating:
                print(f"Detected stagnation after {step} steps. Terminating early.")
                break
            
            # Terminate early if we reached the desired max tile
            if args.max_tile is not None and maxt >= args.max_tile:
                done = True
            
            # render
            if args.render_mode == 'terminal':
                print("\033[H\033[J", end="")
                header = (
                    f"{args.policy.upper()} | Step {step} | Action {action_name}({action_val}) "
                    f"| Score {int(score)} | Reward {int(reward.item())} | Max {maxt}"
                )
                print(header)
                render_board(state)
            else:
                screen.fill((187,173,160))
                # draw the game grid first
                grid = state.cpu().numpy()[0] if state.ndim==3 else state.cpu().numpy()
                draw_board(screen, grid, font)
                # then overlay the header text on top
                header = (
                    f"{args.policy.upper()} | Step {step} | Action {action_name}({action_val}) "
                    f"| Score {int(score)} | Reward {int(reward.item())} | Max {maxt}"
                )
                surf = header_font.render(header, True, TEXT_COLOR_DARK)
                screen.blit(surf, (10, 10))
                pygame.display.flip()
                clock.tick(FPS)
    except KeyboardInterrupt:
        print("\nGame interrupted by user.")
    except Exception as e:
        print(f"\nError during game: {e}")
    finally:
        # end-of-game message before cleanup
        # determine win vs. loss vs. max-step
        final_max = int(state.max().item())
        if args.render_mode == 'terminal':
            if final_max >= args.max_tile:
                print(f"Congratulations! You won by reaching a {args.max_tile} tile!")
            elif done:
                print("Game over: no more moves available.")
            else:
                print(f"Stopped after max steps ({args.max_steps}).")
        elif screen is not None:  # Check if pygame was initialized
            # display message overlay on pygame
            if final_max >= args.max_tile:
                msg = f"Congratulations! Reached {args.max_tile}!"
            elif done:
                msg = "Game over: no more moves!"
            else:
                msg = f"Stopped after max steps ({args.max_steps})"
            try:
                # render overlay for 2 seconds
                overlay = header_font.render(msg, True, TEXT_COLOR_DARK)
                rect = overlay.get_rect(center=(WINDOW_SIZE//2, TILE_SIZE//2))
                screen.blit(overlay, rect)
                pygame.display.flip()
                pygame.time.wait(2000)
            except pygame.error:
                pass  # Ignore pygame errors during cleanup
        
        # cleanup
        if args.render_mode == 'pygame':
            try:
                pygame.quit()
            except pygame.error:
                pass
        
        try:
            # save stats plots
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(rewards_hist)
            plt.title(f"{args.policy.upper()} Reward per Step")
            plt.xlabel("Step")
            plt.ylabel("Reward")
            plt.savefig(f"plots/{args.policy}_reward_plot.png")
            plt.close()
            
            plt.figure()
            plt.plot(max_tiles_hist)
            plt.title(f"{args.policy.upper()} Max Tile per Step")
            plt.xlabel("Step")
            plt.ylabel("Max Tile")
            plt.savefig(f"plots/{args.policy}_maxtile_plot.png")
            plt.close()
        except Exception as e:
            print(f"Error saving plots: {e}")


# Entry point for script mode (moved to bottom after helper functions)

# Print a 4x4 board to the terminal

def render_board(state):
    grid_arr = state.cpu().numpy()
    # if batch dimension present, take first board
    if grid_arr.ndim == 3:
        grid_arr = grid_arr[0]
    grid = grid_arr
    for row in grid:
        # row is a 1D array of ints
        print(" ".join(f"{int(v):4d}" if v != 0 else "   ." for v in row))
    print()

# Random policy: uniform random choice

def random_policy(env, state):
    return random.randrange(4)

# DQN policy: greedy from trained Q-network

def dqn_policy(env, state, net, device):
    s = state.unsqueeze(0).to(device)
    with torch.no_grad():
        q = net(s)
    return int(q.argmax(dim=1).item())

# Entry point for script mode
if __name__ == '__main__':
    main() 