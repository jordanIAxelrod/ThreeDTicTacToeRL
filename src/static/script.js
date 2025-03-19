document.addEventListener('DOMContentLoaded', () => {
    const gameStatus = document.getElementById('gameStatus');
    const newGameBtn = document.getElementById('newGameBtn');
    let currentBoard = null;

    // Initialize the game board UI
    const layers = ['layer0', 'layer1', 'layer2'];
    layers.forEach(layerId => {
        const grid = document.querySelector(`#${layerId} .grid`);
        for (let i = 0; i < 9; i++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.dataset.x = Math.floor(i / 3);
            cell.dataset.y = i % 3;
            cell.dataset.layer = layerId.replace('layer', '');
            cell.addEventListener('click', handleCellClick);
            grid.appendChild(cell);
        }
    });

    // Start a new game
    newGameBtn.addEventListener('click', startNewGame);

    function startNewGame() {
        fetch('/start_game', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                currentBoard = data.board;
                updateBoard();
                gameStatus.textContent = "Your turn! (You are X)";
                // Remove winning-cell class from all cells
                document.querySelectorAll('.cell').forEach(cell => {
                    cell.classList.remove('winning-cell');
                });
                newGameBtn.textContent = "Restart Game";
            });
    }

    function handleCellClick(event) {
        const cell = event.target;
        const x = parseInt(cell.dataset.x);
        const y = parseInt(cell.dataset.y);
        if (cell.dataset.layer !== "0") {
            return;
        }

        fetch('/make_move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ x, y }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                gameStatus.textContent = data.error;
                return;
            }

            currentBoard = data.board;
            updateBoard();

            if (data.status) {
                handleGameEnd(data.status, data.winning_coordinates);
            } else if (data.bot_move) {
                gameStatus.textContent = "Your turn! (You are X)";
            }
        });
    }

    function updateBoard() {
        layers.forEach((layerId, z) => {
            const grid = document.querySelector(`#${layerId} .grid`);
            const cells = grid.children;
            for (let i = 0; i < 9; i++) {
                const x = Math.floor(i / 3);
                const y = i % 3;
                const value = currentBoard[x][y][2 - z];
                cells[i].textContent = value === 1 ? 'X' : value === -1 ? 'O' : '';
            }
        });
    }

    function handleGameEnd(status, winningCoordinates) {
        let message;
        switch (status) {
            case 'player_wins':
                message = "Congratulations! You won!";
                break;
            case 'bot_wins':
                message = "Game Over - Bot wins!";
                break;
            case 'draw':
                message = "Game Over - It's a draw!";
                break;
        }
        gameStatus.textContent = message;

        // Highlight winning cells if there are winning coordinates
        if (winningCoordinates) {
            highlightWinningCells(winningCoordinates);
        }
    }

    function highlightWinningCells(coordinates) {
        // Remove previous highlights
        document.querySelectorAll('.cell').forEach(cell => {
            cell.classList.remove('winning-cell');
        });

        // Add highlights to winning cells
        coordinates.forEach(([x, y, z]) => {
            const layerId = `layer${2 -z}`;
            const grid = document.querySelector(`#${layerId} .grid`);
            const index = x * 3 + y;
            const cell = grid.children[index];
            cell.classList.add('winning-cell');
        });
    }

    // Start the game automatically when the page loads
    startNewGame();
}); 