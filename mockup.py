import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap

def create_mock_world(grid_size=10, num_coop=5, num_ind=5, num_food=5):
    """
    Create a random 2D grid (world) representing a mock snapshot.
      0 = empty cell
      1 = food
      2 = cooperative agent (A)
      3 = individualistic agent (B)
    """
    world = np.zeros((grid_size, grid_size), dtype=int)

    # Place food randomly
    for _ in range(num_food):
        x = random.randint(0, grid_size - 1)
        y = random.randint(0, grid_size - 1)
        world[y, x] = 1  # 1 for food

    # Place cooperative agents (A)
    for _ in range(num_coop):
        x = random.randint(0, grid_size - 1)
        y = random.randint(0, grid_size - 1)
        world[y, x] = 2  # 2 for A

    # Place individualistic agents (B)
    for _ in range(num_ind):
        x = random.randint(0, grid_size - 1)
        y = random.randint(0, grid_size - 1)
        world[y, x] = 3  # 3 for B

    return world

def plot_mock_world(world):
    """
    Plots the world with:
      - White for empty cells (0)
      - White for cells that contain food (1) [we'll add circles for these]
      - Blue for cooperative agents (2)
      - Red for individualistic agents (3)
    Overlay: draw circles for food cells.
    Also displays gridlines for each cell.
    """
    grid_size = world.shape[0]

    # We define a custom colormap:
    # index 0 => white (empty), 1 => white (food placeholders),
    # index 2 => blue (A), index 3 => red (B).
    cmap = ListedColormap(["white", "white", "blue", "red"])

    # Plot the background with imshow (no food color here, since we do circles next)
    plt.imshow(world, cmap=cmap, origin='upper', vmin=0, vmax=3)

    # Overlay scatter for food cells to get circular markers
    food_y, food_x = np.where(world == 1)
    plt.scatter(food_x, food_y,
                marker='o',            # circle
                s=200,                 # size of circles
                c='green',             # fill color
                edgecolor='black',     # outline
                linewidths=1,
                label='Food')

    # Create a legend by manually adding patches for the other categories
    # (But note that food is already in the scatter's label)
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color="blue", label="Cooperative (2)"),
        mpatches.Patch(color="red",  label="Individualistic (3)"),
        mpatches.Patch(color="white", label="Empty (0)")
        # Food is auto-labeled from scatter
    ]

    # We'll combine the scatter legend entry with these patches
    plt.legend(handles=legend_patches + [plt.scatter([], [], color='green', edgecolor='black', s=100, label='Food')],
               bbox_to_anchor=(1.05, 1),
               loc='upper left',
               borderaxespad=0.)

    # Configure gridlines
    plt.xticks(range(grid_size))
    plt.yticks(range(grid_size))
    # Force the grid to line up exactly with cell boundaries:
    plt.grid(which='major', color='gray', linewidth=0.8, linestyle='-')
    # Because the image is "upper-origin", we invert the y-axis to match array coords:
    plt.gca().invert_yaxis()

    plt.title("Mock 2D Simulation Snapshot (with Grid & Food Circles)")

def main():
    # Example parameters for a 10x10 grid
    grid_size = 30
    num_coop_agents = 20
    num_ind_agents = 20
    num_food = 15

    world = create_mock_world(
        grid_size=grid_size,
        num_coop=num_coop_agents,
        num_ind=num_ind_agents,
        num_food=num_food
    )

    # Plot and save
    plt.figure(figsize=(10, 10))
    plot_mock_world(world)
    plt.tight_layout()
    plt.savefig("mockup_simulation.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
