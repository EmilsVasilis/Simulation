import numpy as np
import matplotlib.pyplot as plt
import random
import imageio  # Make sure to install via "pip install imageio"
from matplotlib.colors import ListedColormap

def create_mock_world(grid_size=10, num_coop=5, num_ind=5, num_food=5):
    """Create a random 2D grid snapshot."""
    world = np.zeros((grid_size, grid_size), dtype=int)

    # 1 = food, 2 = cooperative agent (A), 3 = individualistic agent (B)

    # Place food randomly
    for _ in range(num_food):
        x = random.randint(0, grid_size - 1)
        y = random.randint(0, grid_size - 1)
        world[y, x] = 1

    # Place cooperative agents
    for _ in range(num_coop):
        x = random.randint(0, grid_size - 1)
        y = random.randint(0, grid_size - 1)
        world[y, x] = 2

    # Place individualistic agents
    for _ in range(num_ind):
        x = random.randint(0, grid_size - 1)
        y = random.randint(0, grid_size - 1)
        world[y, x] = 3

    return world

def random_step(world):
    """
    DEMO approach: Move each agent 1 step up/down/left/right randomly.
    In a real simulation, you'd apply your actual logic (movement, hunger, etc.).
    """
    grid_size = world.shape[0]
    new_world = np.zeros_like(world)  # fresh grid

    for y in range(grid_size):
        for x in range(grid_size):
            cell_value = world[y, x]
            if cell_value in (2, 3):  # agent
                # Random step (-1, 0, +1)
                dx = random.choice([-1, 0, 1])
                dy = random.choice([-1, 0, 1])

                nx = max(0, min(grid_size - 1, x + dx))
                ny = max(0, min(grid_size - 1, y + dy))

                # If new cell is empty, move the agent there
                if new_world[ny, nx] == 0:
                    new_world[ny, nx] = cell_value
                else:
                    # If the cell is occupied or something,
                    # we simply don't move (for simplicity)
                    new_world[y, x] = cell_value

            elif cell_value == 1:
                # Keep food where it was
                new_world[y, x] = 1

    return new_world

def plot_frame(world):
    """
    Plot the grid for a single frame.
    Returns a numpy array (RGB image) that can be appended to a GIF.
    """
    cmap = ListedColormap(["white", "green", "blue", "red"])

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(world, cmap=cmap, origin='upper', vmin=0, vmax=3)

    # Optionally, add grid lines for clarity
    grid_size = world.shape[0]
    ax.set_xticks(np.arange(grid_size + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(grid_size + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()

    # Convert the current plot to a numpy array (RGB)
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return frame

def main():
    # Create a starting grid
    grid_size = 20
    world = create_mock_world(grid_size=grid_size, num_coop=12, num_ind=12, num_food=10)

    frames = []
    num_frames = 30  # how many steps (frames) to animate

    for i in range(num_frames):
        # Capture a snapshot of the current state
        frame_img = plot_frame(world)
        frames.append(frame_img)

        # Then update the world for the next frame
        world = random_step(world)

    # Save frames to a GIF
    imageio.mimsave("mockup_simulation.gif", frames, fps=2, loop=0)
    print("Saved mockup_simulation.gif")

if __name__ == "__main__":
    main()
