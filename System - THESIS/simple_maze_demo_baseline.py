#!/usr/bin/env python3

"""
simple_maze_demo_baseline.py
-------------------------------------------------------------------
Same visual maze demo as simple_maze_demo.py, but wired to
BaselineConfidenceAgent instead of ReflectionAgent, so you can watch
the EXISTING/baseline algorithm move through the maze specifically.

Place this file in the same folder as:
  - dynamic_maze_env.py
  - baseline_confidence_agent.py

Run with:  py simple_maze_demo_baseline.py
"""

import pygame
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dynamic_maze_env import DynamicMazeEnv
from baseline_confidence_agent import BaselineConfidenceAgent


class SimpleMazeVisualization:
    def __init__(self, width=1000, height=600):

        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Maze Demonstration - Baseline Agent")

        # Grid settings
        self.grid_size = 10
        self.cell_size = min(width, height - 100) // self.grid_size
        self.grid_offset_x = 350
        self.grid_offset_y = (height - 100 - self.cell_size * self.grid_size) // 2

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (128, 128, 128)
        self.RED = (255, 0, 0)      # Baseline agent
        self.GREEN = (0, 255, 0)    # Goal
        self.YELLOW = (255, 255, 0) # Walls / hazards

        # Font settings
        pygame.font.init()
        self.font = pygame.font.Font(None, 24)

        # Running state
        self.running = True
        self.paused = False
        self.last_action_text = "None"
        self.episode_num = 1
        self.status_text = "Press ENTER to start Episode 1"
        self.waiting_for_input = True

        self.steps_taken = 0
        self.goal_status = "In Progress..."
        self.path_history = []

    def draw_maze(self, maze, agent_pos, goal_pos):
        """Draw the maze"""
        self.screen.fill(self.WHITE)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = self.grid_offset_x + j * self.cell_size
                y = self.grid_offset_y + i * self.cell_size

                if maze[i][j]:  # Walls / hazards
                    pygame.draw.rect(self.screen, self.YELLOW,
                                   (x, y, self.cell_size, self.cell_size))
                else:
                    pygame.draw.rect(self.screen, self.WHITE,
                                   (x, y, self.cell_size, self.cell_size))

                pygame.draw.rect(self.screen, self.GRAY,
                               (x, y, self.cell_size, self.cell_size), 1)

                if i == agent_pos[0] and j == agent_pos[1]:
                    pygame.draw.circle(self.screen, self.RED,
                                    (x + self.cell_size//2, y + self.cell_size//2),
                                    self.cell_size//3)

                if i == goal_pos[0] and j == goal_pos[1]:
                    pygame.draw.circle(self.screen, self.GREEN,
                                    (x + self.cell_size//2, y + self.cell_size//2),
                                    self.cell_size//3)

        info_text = [
            "Maze Demonstration - Baseline (Existing) Agent",
            f"Episode: {self.episode_num}",
            f"Status: {self.status_text}",
            f"Goal Result: {self.goal_status}",
            f"Steps Taken: {self.steps_taken}/100",
            f"Last Action: {self.last_action_text}",
            "---",
            "Red Ball: Baseline (Existing) Agent",
            "Green Circle: Goal",
            "Yellow Blocks: Walls / hazards",
            "Spacebar: Pause/Resume",
            "ESC Key: Exit"
        ]

        for i, text in enumerate(info_text):
            text_surface = self.font.render(text, True, self.BLACK)
            self.screen.blit(text_surface, (20, 20 + i * 25))

        pygame.display.flip()

    def handle_events(self):
        """Handle events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_RETURN:
                    self.waiting_for_input = False
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
        return self.running


def main():
    """Main program"""
    print("Starting maze demonstration (Baseline / Existing agent)...")
    print("The program will display a pygame window showing the baseline agent moving through the maze")
    print("Press the spacebar to pause/resume, and the ESC key to exit")

    viz = SimpleMazeVisualization(width=1000, height=600)

    env = DynamicMazeEnv(size=10, obstacle_ratio=0.25, change_frequency=20)
    env.max_steps = 100

    # NOTE: BaselineConfidenceAgent has no confidence_threshold/adaptation_threshold
    # attributes and no set_goal_position() method -- those are ReflectionAgent-only.
    # It doesn't need them: select_action()/learn() are all it uses.
    agent = BaselineConfidenceAgent(env.action_space)

    action_names = ["Up", "Down", "Left", "Right"]

    try:
        episode = 0
        while episode < 5 and viz.running:  # Only run 5 episodes
            state, _ = env.reset()

            viz.status_text = f"Ready for Episode {episode + 1}. Press ENTER."
            viz.waiting_for_input = True

            while viz.waiting_for_input and viz.running:
                viz.handle_events()
                viz.draw_maze(env.maze, state, env.goal_pos)
            print(f"\nStarting episode {episode + 1}")

            state, _ = env.reset()
            viz.path_history = []
            steps = 0
            done = False

            while not done and steps < env.max_steps and viz.running:
                if not viz.handle_events():
                    break

                if viz.paused:
                    pygame.time.wait(100)
                    continue

                action = agent.select_action(state)
                next_state, reward, done, _, _ = env.step(action)

                viz.last_action_text = action_names[action]

                shortest_path = env.get_optimal_path_length()
                agent.learn(state, action, reward, next_state, done, steps, shortest_path)

                state = next_state
                viz.path_history.append(state)
                steps += 1

                viz.draw_maze(env.maze, state, env.goal_pos)
                pygame.time.wait(200)

                if done and np.array_equal(state, env.goal_pos):
                    viz.status_text = "Status: Goal Achieved!"
                    viz.goal_status = "Goal Achieved!"
                    print(f"Episode {episode + 1} completed successfully! Steps: {steps}")
                elif steps >= env.max_steps:
                    viz.status_text = "Status: Goal Not Achieved (Timeout)"
                    viz.goal_status = "Goal Not Achieved"
                    print(f"Episode {episode + 1} timed out, Steps: {steps}")

                viz.steps_taken = steps
                viz.draw_maze(env.maze, state, env.goal_pos)
            episode += 1

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Program finished")
        viz.running = False
        pygame.quit()


if __name__ == "__main__":
    main()