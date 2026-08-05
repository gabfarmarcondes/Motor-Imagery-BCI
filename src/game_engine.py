"""
PUre game logic for the Motor Imagery BCI game. Kept separate so it can be smoke-tested without a graphical enviroment.

The game is a simple 2D grid. A player moves one cell per turn in one of four directions, trying to reach a target.
Each turn's direction comes from an EEG classifier elsewhere (play_bci.py); this module only knows about grid moviment.
"""

from dataclasses import dataclass, field
import random


# Maps the 4 motor imagery classes (0-3) to grid movementes.
# Intuitive: hands to the sides, foot down, tongue up
CLASS_TO_DIRECTION = {
    0: (-1, 0), # Left Hand -> move left (dx=-1)
    1: (1, 0), # Right Hand -> move right (dx=+1)
    2: (0, 1), # Foot -> move down (dy=+1), y grows downward
    3: (0, -1) # Tongue -> move up (dy=-1)
}

CLASS_NAME = {0: "Left Hand", 1: "Right Hand", 2: "Foot", 3: "Tongue"}
DIRECTION_NAMES = {0: "LEFT", 1: "RIGHT", 2: "DOWN", 3: "UP"}

@dataclass
class GameState:
    grid_width: int = 8
    grid_height: int = 8
    player_x: int = 4
    player_y: int = 4
    target_x: int = 0
    target_y: int = 0
    score: int = 0
    turns_taken: int = 0
    correct_predictions: int = 0 # how often prediction matched the trial's the label

    def __post_init__(self):
        self._place_new_target()

    def _place_new_target(self):
        # Places the target at a random cell that isn't the player's current cell
        while True:
            self.target_x = random.randint(0, self.grid_width - 1)
            self.target_y = random.randint(0, self.grid_height - 1)
            if (self.target_x, self.target_y) != (self.player_x, self.player_y):
                break

    def apply_move(self, predicted_class: int, true_class:int = None):
        """
        Moves the player one cell in the direction for predicted_class, clamped to the grid edges.
        If true_class is given (we're replaying labeled data), tracks whether the model predicted correctly,
        purely for display, it does not change movement (the game moves by the prediction, right or wrong,
        because that's whant an EEG game would actually do).

        Returns True if this move reached the target (a pointed score)
        """
        dx, dy = CLASS_TO_DIRECTION[predicted_class]
        self.player_x = max(0, min(self.grid_width - 1, self.player_x + dx))
        self.player_y = max(0, min(self.grid_height - 1, self.player_y + dy))

        self.turns_taken += 1

        if true_class is not None and predicted_class == true_class:
            self.correct_predictions += 1

        if (self.player_x, self.player_y) == (self.target_x, self.target_y):
            self.score += 1
            self._place_new_target()
            return True
        return False


# smoke test (runs the game logic with random predictions but without display)
if __name__ == "__main__":
    random.seed(42)
    state = GameState()
    print(f"Start: player at ({state.player_x},{state.player_y}) -> "
          f"target at ({state.target_x},{state.target_y})")

    for turn in range(20) :
        fake_prediction = random.randint(0, 3)
        scored = state.apply_move(fake_prediction)
        marker = " <-- Scored!" if scored else ""
        print(f"Turn {turn}: moved {DIRECTION_NAMES[fake_prediction]:5s} -> "
            f"player at ({state.player_x},{state.player_y}){marker}")

    print(f"\nFinal score: {state.score} in {state.turns_taken} turns")