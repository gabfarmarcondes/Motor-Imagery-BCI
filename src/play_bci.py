"""
Motor Imagery BCI game: controla  2D grid player using EEG motor imagery classifications, to reach target.

Each turn replays one recorded EEG trial: the trial goes through the trained EEGNet model, and the player moves 
one cell in the direction of the PREDICTED class, right or wrong. Because the replayed trials are labeled, the
game shows both the true motor imagery class and the model's prediction side by side, making the classifier's
real behavior (including its errors) visible.

This is replay of recorded data, not live acquisition. There's no EEG hardware in the loop. It satisfies
the simulated/replayed EEG demo goal honestly.

Controls:
    SPACE: process the next trial (one turn)
    R: reset the game
    ESQ/Q: quit
"""
import sys
import argparse

import numpy as np
import torch
import pygame

from eeg_dataset import MotorImageryDataset
from train_final_model import load_model
from game_engine import GameState, CLASS_NAME, DIRECTION_NAMES, CLASS_TO_DIRECTION

# The run train_final_model.py holds out from training (its val_run=5 default, and
# train_all_final_models never overrides it). It's the only data the model didn't
# memorize, so it's the only set the game's live accuracy is honest over: replaying
# the training runs would show ~92% instead of the real ~69% held-out accuracy.
HELD_OUT_RUN = 5

# Display config
CELL_SIZE = 60
GRID_W, GRID_H = 8, 8
PANEL_W = 300 # side panel for the brain info
FPS = 30

# Colors (R, G, B)
BG = (18, 18, 24)
GRID_LINE = (40, 40, 52)
PLAYER_COLOR = (90, 200, 160)
TARGET_COLOR = (230, 160, 80)
TEXT_COLOR = (220, 220, 230)
CORRECT_COLOR = (90, 200, 160)
WRONG_COLOR = (220, 100, 100)
DIM_TEXT = (130, 130, 145)

class BCIGame:
    def __init__(self, subject_id=8, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.subject_id = subject_id

        # Load the trained model and the subject's trials (same preprocessing
        # the model was trained on, wideband, normalized, artifact-rejected).
        print(f"Loading model and data for subject {subject_id}")
        self.model, self.checkpoint = load_model(subject_id, self.device)
        self.dataset = MotorImageryDataset(subject_id=subject_id)

        # Replay only the held-out run, so what the panel reports is the model's
        # generalization, not its memory of the training runs.
        # Shuffle the order trials are replayed in, so the demo isn't the same
        # sequence every run. Seeded for reproductibility of a given session
        held_out_mask = self.dataset.run_ids == HELD_OUT_RUN
        self.trial_order = np.where(held_out_mask)[0]
        np.random.seed(42)
        np.random.shuffle(self.trial_order)
        self.next_trial_idx = 0

        self.state = GameState(grid_width=GRID_W, grid_height=GRID_H)

        # Info about the most recent turn, for the side panel
        self.last_true = None
        self.last_pred = None
        self.last_scored = 0

        pygame.init()
        self.screen = pygame.display.set_mode((GRID_W * CELL_SIZE + PANEL_W, GRID_H * CELL_SIZE))
        pygame.display.set_caption(f"Motor Imagery BCI -- Subject {subject_id}")
        self.font = pygame.font.SysFont("consolas", 20)
        self.font_small = pygame.font.SysFont("consolas", 16)
        self.font_big = pygame.font.SysFont("consolas", 28)
        self.clock = pygame.time.Clock()

    def process_next_trial(self):
        # Replays one trial: model predicts, player moves by the prediction
        if self.next_trial_idx >= len(self.trial_order):
            # Ran out of trials, reshuffle and loop, so the demo can continue
            np.random.shuffle(self.trial_order)
            self.next_trial_idx = 0

        trial_i = self.trial_order[self.next_trial_idx]
        self.next_trial_idx += 1

        X = self.dataset.X[trial_i:trial_i + 1].to(self.device)
        true_class = int(self.dataset.y[trial_i].item())

        with torch.no_grad():
            pred_class = int(self.model(X).argmax(dim=1).item())

        self.last_true = true_class
        self.last_pred = pred_class
        self.last_scored = self.state.apply_move(pred_class, true_class=true_class)

    def reset(self):
        self.state = GameState(grid_width=GRID_W, grid_height=GRID_H)
        self.last_true = self.last_pred = None
        self.last_scored = False

    def draw(self):
        self.screen.fill(BG)

        # Grid lines
        for x in range(GRID_W + 1):
            pygame.draw.line(self.screen, GRID_LINE, (x * CELL_SIZE, 0), (x * CELL_SIZE, GRID_H * CELL_SIZE))
        for y in range(GRID_H + 1):
            pygame.draw.line(self.screen, GRID_LINE, (0, y * CELL_SIZE), (GRID_W * CELL_SIZE, y * CELL_SIZE))

        # Target (drawn as a ring)
        tx, ty = self.state.target_x, self.state.target_y
        pygame.draw.rect(self.screen, TARGET_COLOR,
            (tx * CELL_SIZE + 8, ty * CELL_SIZE + 8, CELL_SIZE - 16, CELL_SIZE - 16), width=4)

        # Player (filled)
        px, py = self.state.player_x, self.state.player_y
        pygame.draw.rect(self.screen, PLAYER_COLOR,
            (px * CELL_SIZE + 12, py * CELL_SIZE + 12, CELL_SIZE - 24, CELL_SIZE - 24))

        self._draw_panel()
        pygame.display.flip()

    def _draw_panel(self):
        # Side panel: shows the model's decision for the last turn + running stats.
        panel_x = GRID_W * CELL_SIZE + 20
        y = 20

        def line(text, color=TEXT_COLOR, font=None, dy=28):
            nonlocal y
            surf = (font or self.font).render(text, True, color)
            self.screen.blit(surf, (panel_x, y))
            y += dy

        line(f"Subject {self.subject_id}", font=self.font_big, dy=44)
        line(f"Model acc (trained): {self.checkpoint['val_accuracy']:.0%}", color=DIM_TEXT, font=self.font_small, dy=22)
        line("Replaying held-out run", color=DIM_TEXT, font=self.font_small, dy=32)

        line("Last turn:", color=DIM_TEXT, font=self.font_small, dy=26)
        if self.last_true is not None:
            line(f"Imagined: {CLASS_NAME[self.last_true]}", dy=26)
            pred_color = CORRECT_COLOR if self.last_pred == self.last_true else WRONG_COLOR
            line(f"Predicted: {CLASS_NAME[self.last_pred]}", color=pred_color, dy=26)
            line(f"Moved: {DIRECTION_NAMES[self.last_pred]}", color=DIM_TEXT, font=self.font_small, dy=26)
            verdict = "correct" if self.last_pred == self.last_true else "wrong"
            line(f"-> {verdict}", color=pred_color, font=self.font_small, dy=36)
        else:
            line("(press SPACE)", color=DIM_TEXT, font=self.font_small, dy=36)

        line(f"Score: {self.state.score}", font=self.font_big, dy=40)

        # Live classification accuracy this session (guarded against divide-by-zero)
        if self.state.turns_taken > 0:
            live_acc = self.state.correct_predictions / self.state.turns_taken
            line(f"Turns: {self.state.turns_taken}", color=DIM_TEXT, font=self.font_small, dy=24)
            line(f"Live acc: {live_acc:.0%}", color=DIM_TEXT, font=self.font_small, dy=36)

        # Controls hint at the bottom
        y = GRID_H * CELL_SIZE - 90
        line("SPACE  next trial", color=DIM_TEXT, font=self.font_small, dy=22)
        line("R      reset", color=DIM_TEXT, font=self.font_small, dy=22)
        line("ESC/Q  quit", color=DIM_TEXT, font=self.font_small, dy=22)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.process_next_trial()
                    elif event.key == pygame.K_r:
                        self.reset()

            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motor Imagery BCI game")
    parser.add_argument("--subject", type=int, default=8,
                        help="Subject ID (1-9) whose model and trials to use. Default 8 (one of the strongest).")
    args = parser.parse_args()

    game = BCIGame(subject_id=args.subject)
    game.run()