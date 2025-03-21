import heapq
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import os  # <-- needed to check if file exists

# -------------------
# Default Simulation Settings
# -------------------
GRID_SIZE = 50
INITIAL_POP = 20  # Initial population size for each group
INITIAL_FOOD = 300  # Food spawned initially
MAX_AGE = 100  # Age limit for any individual
HUNGER_THRESHOLD = 20  # Hunger level at which individual dies
HUNGER_INCREASE = 1.0  # Rate at which hunger increases over time
SHARE_RANGE = 4  # Distance within which cooperative group shares
MOVE_RANGE = 2  # Max step size in x or y direction
MEAN_MOVE_INTERVAL = 1.5  # Average time between moves for each individual
TIME_LIMIT = 1000  # Stop simulation after hitting this time
REPRODUCTION_CHANCE = 0.18
REPRODUCTION_DISTANCE = 5
FOOD_SPAWN_INTERVAL = 1.5
COST_OF_SHARING = 0.1
WAIT_SHARING = 200
FOOD_SHARING = 2
HUNGER_SHARING = 10


# Global counters
global_event_counter = 0

def load_simulation_settings(filename):
    global GRID_SIZE, INITIAL_POP, INITIAL_FOOD, MAX_AGE, HUNGER_THRESHOLD
    global HUNGER_INCREASE, SHARE_RANGE, MOVE_RANGE, MEAN_MOVE_INTERVAL
    global TIME_LIMIT, REPRODUCTION_CHANCE, REPRODUCTION_DISTANCE
    global FOOD_SPAWN_INTERVAL, WAIT_SHARING, FOOD_SHARING, HUNGER_SHARING

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, value_str = line.split('=')
            key = key.strip()
            value_str = value_str.strip()

            # Cast to int or float based on the known parameter name
            if key in [
                "GRID_SIZE", "INITIAL_POP", "INITIAL_FOOD", "MAX_AGE",
                "HUNGER_THRESHOLD", "SHARE_RANGE", "MOVE_RANGE", "TIME_LIMIT",
                "REPRODUCTION_DISTANCE", "WAIT_SHARING", "FOOD_SHARING", "HUNGER_SHARING"
            ]:
                value = int(value_str)
            else:
                value = float(value_str)

            # Assign to the correct global variable
            if key == "GRID_SIZE":
                GRID_SIZE = value
            elif key == "INITIAL_POP":
                INITIAL_POP = value
            elif key == "INITIAL_FOOD":
                INITIAL_FOOD = value
            elif key == "MAX_AGE":
                MAX_AGE = value
            elif key == "HUNGER_THRESHOLD":
                HUNGER_THRESHOLD = value
            elif key == "HUNGER_INCREASE":
                HUNGER_INCREASE = value
            elif key == "SHARE_RANGE":
                SHARE_RANGE = value
            elif key == "MOVE_RANGE":
                MOVE_RANGE = value
            elif key == "MEAN_MOVE_INTERVAL":
                MEAN_MOVE_INTERVAL = value
            elif key == "TIME_LIMIT":
                TIME_LIMIT = value
            elif key == "REPRODUCTION_CHANCE":
                REPRODUCTION_CHANCE = value
            elif key == "REPRODUCTION_DISTANCE":
                REPRODUCTION_DISTANCE = value
            elif key == "FOOD_SPAWN_INTERVAL":
                FOOD_SPAWN_INTERVAL = value
            elif key == "WAIT_SHARING":
                WAIT_SHARING = value
            elif key == "FOOD_SHARING":
                FOOD_SHARING = value
            elif key == "HUNGER_SHARING":
                HUNGER_SHARING = value

def next_event_id():
    global global_event_counter
    global_event_counter += 1
    return global_event_counter

def process_food_spawn_event(current_time):
    alive_count = sum(1 for p in population if p.is_alive())
    if alive_count == 0:
        print("No one is alive. Stopping food spawning.")
        return
    spawn_food()
    schedule_event(current_time + FOOD_SPAWN_INTERVAL, "food_spawn", {})

def spawn_food():
    for _ in range(30):  # Try spawning food in 30 different spots
        x = random.randint(0, GRID_SIZE - 1)
        y = random.randint(0, GRID_SIZE - 1)
        food_grid[x, y] += 1.0  # Add only 1 per spot to prevent over-concentration

class Person:
    def __init__(self, x, y, group_id, hunger=0, age=0):
        self.x = x
        self.y = y
        self.group_id = group_id
        self.hunger = hunger
        self.age = age
        self.alive = True
        self.food_carried = 0.0  # Track how much food this individual currently holds
        # Immediately place person on the grid
        grid_people[self.x][self.y].append(self)

    def is_alive(self):
        return self.alive

# Grids to track
food_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
grid_people = [[[] for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
population = []
events = []

food_deaths = 0
age_deaths = 0

def schedule_event(time, event_type, data):
    eid = next_event_id()
    heapq.heappush(events, (time, eid, event_type, data))

def initialize_simulation():
    global food_grid, population, events, food_deaths, age_deaths, grid_people

    # Reset per-run states
    population = []
    events = []
    food_deaths = 0
    age_deaths = 0

    # Clear the event queue and grids
    heapq.heapify(events)
    grid_people = [[[] for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    food_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    # Schedule first food spawn event
    schedule_event(time=1.0, event_type="food_spawn", data={})

    # Spawn initial food
    for _ in range(INITIAL_FOOD):
        fx = random.randint(0, GRID_SIZE - 1)
        fy = random.randint(0, GRID_SIZE - 1)
        food_grid[fx, fy] += 1

    # Create two groups: 0 (Cooperative), 1 (Individualistic)
    for i in range(INITIAL_POP):
        # Group 0, somewhere in the left half
        x0 = random.randint(0, GRID_SIZE // 2 - 1)
        y0 = random.randint(0, GRID_SIZE - 1)
        person_coop = Person(x0, y0, group_id=0)
        population.append(person_coop)

        # Group 1, somewhere in the right half
        x1 = random.randint(GRID_SIZE // 2, GRID_SIZE - 1)
        y1 = random.randint(0, GRID_SIZE - 1)
        person_ind = Person(x1, y1, group_id=1)
        population.append(person_ind)

    # Schedule an initial "move" event for each individual
    for person in population:
        schedule_next_move(person, current_time=0.0)

def schedule_next_move(person, current_time):
    if not person.is_alive():
        return
    dt = random.expovariate(1.0 / MEAN_MOVE_INTERVAL)
    move_time = current_time + dt
    schedule_event(move_time, "move", {"person": person})

def process_event_move(event_time, data, time):
    global age_deaths, food_deaths

    person = data["person"]
    # Aging, hunger
    person.age += 1
    person.hunger += HUNGER_INCREASE

    # Check death by old age or hunger
    if person.age >= MAX_AGE:
        age_deaths += 1
        person.alive = False
        return
    elif person.hunger >= HUNGER_THRESHOLD:
        food_deaths += 1
        person.alive = False
        return

    # Remove from old cell
    old_x, old_y = person.x, person.y
    if person in grid_people[old_x][old_y]:
        grid_people[old_x][old_y].remove(person)

    # Random move
    dx = random.randint(-MOVE_RANGE, MOVE_RANGE)
    dy = random.randint(-MOVE_RANGE, MOVE_RANGE)
    person.x = min(max(person.x + dx, 0), GRID_SIZE - 1)
    person.y = min(max(person.y + dy, 0), GRID_SIZE - 1)

    # Add to new cell
    grid_people[person.x][person.y].append(person)

    # Attempt to eat food at location
    if food_grid[person.x, person.y] > 0:
        person.food_carried += food_grid[person.x, person.y]
        food_grid[person.x, person.y] = 0

    # Use carried food if hungry
    if person.hunger >= 5 and person.food_carried >= 1:
        person.hunger = max(0, person.hunger - 5)
        person.food_carried -= 1

    # Cooperative resource sharing if conditions met
    if (person.group_id == 0 and person.hunger <= HUNGER_SHARING
            and person.food_carried >= FOOD_SHARING and time > WAIT_SHARING):
        share_resources(person)

    # Possibly reproduce
    maybe_reproduce(person, event_time)

    # Schedule next move
    schedule_next_move(person, event_time)

def distance(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

def share_resources(person):
    if not person.is_alive():
        return

    neighbors = []
    x_min = max(0, person.x - SHARE_RANGE)
    x_max = min(GRID_SIZE - 1, person.x + SHARE_RANGE)
    y_min = max(0, person.y - SHARE_RANGE)
    y_max = min(GRID_SIZE - 1, person.y + SHARE_RANGE)

    for i in range(x_min, x_max + 1):
        for j in range(y_min, y_max + 1):
            for occupant in grid_people[i][j]:
                if (occupant.is_alive()
                        and occupant.group_id == person.group_id):
                    neighbors.append(occupant)

    if not neighbors:
        return

    total_food = sum(p.food_carried for p in neighbors) + person.food_carried
    if total_food <= 0:
        return

    share_each = total_food / (len(neighbors) + 1)
    person.food_carried = share_each
    for p in neighbors:
        p.food_carried = share_each

def maybe_reproduce(person, event_time):
    neighbors = []
    x_min = max(0, person.x - REPRODUCTION_DISTANCE)
    x_max = min(GRID_SIZE - 1, person.x + REPRODUCTION_DISTANCE)
    y_min = max(0, person.y - REPRODUCTION_DISTANCE)
    y_max = min(GRID_SIZE - 1, person.y + REPRODUCTION_DISTANCE)

    for i in range(x_min, x_max + 1):
        for j in range(y_min, y_max + 1):
            for occupant in grid_people[i][j]:
                if (occupant.is_alive()
                        and occupant.group_id == person.group_id
                        and occupant != person
                        and occupant.age >= 18
                        and occupant.hunger <= 10
                        and occupant.food_carried >= 1
                        and person.hunger <= 10
                        and person.age >= 18
                        and person.food_carried >= 1):
                    neighbors.append(occupant)

    if neighbors and random.random() < REPRODUCTION_CHANCE:
        child = Person(person.x, person.y, group_id=person.group_id)
        population.append(child)
        schedule_next_move(child, current_time=event_time)

def run_simulation(run_index=0):
    # Re-initialize for each run
    initialize_simulation()

    AA = 0   # Accumulates living pop of group A across snapshots
    AB = 0   # Accumulates living pop of group B across snapshots
    PeakA = 0
    PeakB = 0
    current_time = 0.0
    snapshots = []

    while events:
        event_time, eid, event_type, data = heapq.heappop(events)
        current_time = event_time

        if current_time > TIME_LIMIT:
            break

        if event_type == "food_spawn":
            process_food_spawn_event(current_time)
        elif event_type == "move":
            process_event_move(event_time, data, current_time)

        # Example: gather snapshot every 1000 events
        if eid % 1000 == 0:
            aliveA = sum(1 for p in population if p.is_alive() and p.group_id == 0)
            aliveB = sum(1 for p in population if p.is_alive() and p.group_id == 1)
            if aliveA > PeakA:
                PeakA = aliveA
            if aliveB > PeakB:
                PeakB = aliveB
            AA += aliveA
            AB += aliveB
            total_food = np.sum(food_grid)
            snapshots.append((current_time, aliveA, aliveB, total_food))

    # Final snapshot
    aliveA = sum(1 for p in population if p.is_alive() and p.group_id == 0)
    aliveB = sum(1 for p in population if p.is_alive() and p.group_id == 1)
    if aliveA > PeakA:
        PeakA = aliveA
    if aliveB > PeakB:
        PeakB = aliveB

    AA += aliveA
    AB += aliveB
    total_food = np.sum(food_grid)
    snapshots.append((current_time, aliveA, aliveB, total_food))
    snapshots.sort(key=lambda x: x[0])

    A_win_flag = 1 if aliveA > 0 else 0
    B_win_flag = 1 if aliveB > 0 else 0

    # Optionally plot
    if len(snapshots) > 1:
        time_history, popA_history, popB_history, food_history = zip(*snapshots)
        plt.figure()
        plt.plot(time_history, popA_history, label="Cooperative Group")
        plt.plot(time_history, popB_history, label="Individualistic Group")
        plt.plot(time_history, food_history, label="Total Food", linestyle="--")
        plt.xlabel("Simulation Time (events)")
        plt.ylabel("Population / Food")
        plt.title(f"Run {run_index}: Cooperative vs. Individualistic")
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        "PeakPopA": PeakA,
        "PeakPopB": PeakB,
        "TotalPopA": AA,
        "TotalPopB": AB,
        "A_win": A_win_flag,
        "B_win": B_win_flag
    }, snapshots

if __name__ == "__main__":
    # Optional: load user-specified settings
    # load_simulation_settings("DefaultSettings.txt")

    num_runs = 100
    load_simulation_settings("DefaultSettings.txt")
    # collect time-series data for each run (optional)
    all_runs_data = []
    # collect final summary data for each run
    summary_data = []

    for run_index in range(num_runs):
        run_stats, snapshots = run_simulation(run_index)


        for (t, pa, pb, fd) in snapshots:
            all_runs_data.append([run_index, t, pa, pb, fd])

        # Collect final summary for this run
        summary_data.append([
            run_stats["PeakPopA"],
            run_stats["PeakPopB"],
            run_stats["TotalPopA"],
            run_stats["TotalPopB"],
            run_stats["A_win"],
            run_stats["B_win"],
            GRID_SIZE,
            INITIAL_POP,
            INITIAL_FOOD,
            MAX_AGE,
            HUNGER_THRESHOLD,
            HUNGER_INCREASE,
            SHARE_RANGE,
            MOVE_RANGE,
            MEAN_MOVE_INTERVAL,
            TIME_LIMIT,
            REPRODUCTION_CHANCE,
            REPRODUCTION_DISTANCE,
            FOOD_SPAWN_INTERVAL,
            WAIT_SHARING,
            HUNGER_SHARING,
            FOOD_SHARING
        ])


    # Check if file already exists to see if we need a header
    file_exists_data = os.path.exists("simulation_data.csv")
    with open("simulation_data.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists_data:
            writer.writerow(["Run", "Time", "PopA", "PopB", "TotalFood"])
        writer.writerows(all_runs_data)
    print("Appended per-timestep data to simulation_data.csv")

    # -------------------------------------------------------------------------
    # APPEND RUN-BY-RUN SUMMARY
    # -------------------------------------------------------------------------
    file_exists_summary = os.path.exists("simulation_summary.csv")
    with open("simulation_summary.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # If file didn't exist, write the header first
        if not file_exists_summary:
            writer.writerow([
                "PeakPopA", "PeakPopB", "TotalPopA", "TotalPopB",
                "AWin", "BWin",
                "GRID_SIZE", "INITIAL_POP", "INITIAL_FOOD", "MAX_AGE",
                "HUNGER_THRESHOLD", "HUNGER_INCREASE", "SHARE_RANGE", "MOVE_RANGE",
                "MEAN_MOVE_INTERVAL", "TIME_LIMIT", "REPRODUCTION_CHANCE",
                "REPRODUCTION_DISTANCE", "FOOD_SPAWN_INTERVAL", "WAIT_SHARING", "HUNGER_SHARING", "FOOD_SHARING"
            ])
        writer.writerows(summary_data)
    print("Appended final run-by-run summary to simulation_summary.csv")

    print("Done.")
