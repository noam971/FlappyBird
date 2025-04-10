# Flappy Bird

## Overview

This project is a clone of the popular Flappy Bird game, featuring two modes of play:

- **AI Mode**: Uses NEAT (NeuroEvolution of Augmenting Topologies) to evolve a neural network that controls the bird.
- **Human Mode**: You control the bird manually using the keyboard.


## Project Structure


    my_flappy_project/
    ├── ai.py           # AI-specific game mode logic
    ├── bird.py         # Bird class and behavior
    ├── constants.py    # Common constants (dimensions, fonts, global variables)
    ├── game.py         # Core game logic (game loop, drawing, event handling)
    ├── ground.py       # Ground/base class logic
    ├── main.py         # Entry point for the project
    ├── menu.py         # Main menu logic for selecting game mode
    ├── pipe.py         # Pipe class and behavior
    └── imgs/
        └── bg.png      # Background image


## Requirements

- Python 3.x (e.g., Python 3.9)
- [Pygame](https://www.pygame.org/)
- [NEAT-Python](https://github.com/CodeReclaimers/neat-python)
- [Pillow (PIL)](https://python-pillow.org/)

## Installation

1. **Clone the repository:**

```` 
git clone https://github.com/noam971/FlappyBird.git
cd my_flappy_project
````
2. **Install the required packages:**

````
pip install pygame neat-python pillow
````
## How to Run

````
python main.py
````

### Controls
- **Main Menu:**
  - Press **1** for AI Mode.

  - Press **2** for Human Mode.

  - Close the menu window to exit.


- **In Game:**

  - Press **SPACE** (Human Mode) to jump.

  - Press **ESC** to open the pause menu, then:

    - Press **Y** to return to the main menu.

    - Press **N** to resume the game.


## AI Mode Configuration
The NEAT configuration file is named config_feedforward.txt. Ensure this file is present in the project root and is configured according to your requirements.

## Customization and Contributions
Feel free to modify and extend the code to improve gameplay, adjust AI parameters, or enhance the user interface. Contributions and suggestions are always welcome!
