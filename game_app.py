from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

# Game data
games = {
    "tic_tac_toe": {"name": "Tic Tac Toe", "description": "Classic X and O game for two players"},
    "number_guess": {
        "name": "Number Guess",
        "description": "Guess the secret number between 1-100",
    },
    "rock_paper_scissors": {
        "name": "Rock Paper Scissors",
        "description": "Play the classic game against the computer",
    },
    "memory_game": {"name": "Memory Game", "description": "Match pairs of cards"},
}


@app.route("/")
def index():
    """Render the homepage that lists available mini-games."""
    return render_template("index.html", games=games)


@app.route("/game/<game_name>")
def game(game_name):
    if game_name in games:
        return render_template("game.html", game_name=game_name, game=games[game_name])
    else:
        return "Game not found", 404


# Tic Tac Toe Game
@app.route("/api/tic_tac_toe/new")
def tic_tac_toe_new():
    return jsonify({"board": [["" for _ in range(3)] for _ in range(3)], "current_player": "X"})


@app.route("/api/tic_tac_toe/move", methods=["POST"])
def tic_tac_toe_move():
    data = request.get_json()
    board = data.get("board", [])
    row = int(data.get("row", -1))
    col = int(data.get("col", -1))
    player = data.get("player")

    if not (0 <= row < 3 and 0 <= col < 3) or player not in ("X", "O"):
        return jsonify({"error": "Invalid move"}), 400

    if board[row][col] == "":
        board[row][col] = player
        return jsonify(
            {
                "board": board,
                "winner": check_winner(board),
                "current_player": "O" if player == "X" else "X",
            }
        )

    return jsonify({"error": "Invalid move"}), 400


def check_winner(board):
    # Check rows
    for row in board:
        if row[0] == row[1] == row[2] and row[0] != "":
            return row[0]

    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != "":
            return board[0][col]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != "":
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != "":
        return board[0][2]

    # Check for tie
    for row in board:
        if "" in row:
            return None
    return "Tie"


# Number Guess Game
@app.route("/api/number_guess/new")
def number_guess_new():
    return jsonify(
        {
            "target": random.randint(1, 100),
            "attempts": 0,
            "message": "Guess a number between 1 and 100",
        }
    )


@app.route("/api/number_guess/guess", methods=["POST"])
def number_guess_guess():
    data = request.get_json()
    target = int(data.get("target", 0))
    guess = int(data.get("guess", 0))
    attempts = int(data.get("attempts", 0))

    attempts += 1

    if guess < target:
        message = "Too low! Try again."
    elif guess > target:
        message = "Too high! Try again."
    else:
        message = f"Congratulations! You guessed it in {attempts} attempts!"

    return jsonify({"message": message, "attempts": attempts, "target": target})


# Rock Paper Scissors
@app.route("/api/rock_paper_scissors/new")
def rock_paper_scissors_new():
    return jsonify({"choices": ["rock", "paper", "scissors"]})


@app.route("/api/rock_paper_scissors/play", methods=["POST"])
def rock_paper_scissors_play():
    data = request.get_json()
    player_choice = data["choice"]
    choices = ["rock", "paper", "scissors"]
    computer_choice = random.choice(choices)

    if player_choice == computer_choice:
        result = "It's a tie!"
    elif (
        (player_choice == "rock" and computer_choice == "scissors")
        or (player_choice == "paper" and computer_choice == "rock")
        or (player_choice == "scissors" and computer_choice == "paper")
    ):
        result = "You win!"
    else:
        result = "Computer wins!"

    return jsonify(
        {"player_choice": player_choice, "computer_choice": computer_choice, "result": result}
    )


# Memory Game
@app.route("/api/memory_game/new")
def memory_game_new():
    # Create pairs of cards
    symbols = ["🍎", "🍌", "🍒", "🍇", "🍊", "🍓", "🍑", "🍍"]
    cards = symbols * 2
    random.shuffle(cards)

    return jsonify(
        {"cards": cards, "flipped": [False] * len(cards), "matched": [False] * len(cards)}
    )


@app.route("/api/memory_game/reveal", methods=["POST"])
def memory_game_reveal():
    data = request.get_json()
    cards = data["cards"]
    flipped = data["flipped"]
    matched = data["matched"]
    index = data["index"]

    if not flipped[index] and not matched[index]:
        flipped[index] = True

    return jsonify({"flipped": flipped, "cards": cards, "matched": matched})


if __name__ == "__main__":
    app.run(debug=True)
