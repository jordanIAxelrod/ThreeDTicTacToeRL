document.addEventListener('DOMContentLoaded', () => {
    const gameStatus = document.getElementById('gameStatus');
    const newGameBtn = document.getElementById('newGameBtn');
    const aiVsAiToggle = document.getElementById('aiVsAiToggle');
    const minimaxToggle = document.getElementById('minimaxToggle');
    const playerFirstToggle = document.getElementById('playerFirstToggle');
    let currentBoard = null;
    let isAiVsAi = false;
    let useMinimax = false;
    let playerGoesFirst = true;
    let aiMoveInterval = null;
    let waitingForPlayerMove = false;

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

    // Handle AI vs AI toggle
    aiVsAiToggle.addEventListener('change', (e) => {
        isAiVsAi = e.target.checked;
        if (currentBoard) {
            startNewGame();
        }
    });

    // Handle minimax toggle
    minimaxToggle.addEventListener('change', (e) => {
        useMinimax = e.target.checked;
        if (currentBoard) {
            startNewGame();
        }
    });

    // Handle player first toggle
    playerFirstToggle.addEventListener('change', (e) => {
        playerGoesFirst = e.target.checked;
        if (currentBoard) {
            startNewGame();
        }
    });

    function startNewGame() {
        // Clear any existing AI move interval
        if (aiMoveInterval) {
            clearInterval(aiMoveInterval);
        }
        waitingForPlayerMove = false;

        fetch('/start_game', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                ai_vs_ai: isAiVsAi,
                use_minimax: useMinimax,
                player_first: playerGoesFirst
            }),
        })
        .then(response => response.json())
        .then(data => {
            currentBoard = data.board;
            updateBoard();
            if (isAiVsAi) {
                gameStatus.textContent = "AI vs AI Mode - Game in progress...";
                startAiVsAiGame();
            } else {
                if (playerGoesFirst) {
                    gameStatus.textContent = "Your turn! (You are X)";
                    waitingForPlayerMove = true;
                } else {
                    if (data.bot_move) {
                        // AI has made its first move
                        gameStatus.textContent = "Your turn! (You are O)";
                        waitingForPlayerMove = true;
                    }
                }
            }
            // Remove winning-cell class from all cells
            document.querySelectorAll('.cell').forEach(cell => {
                cell.classList.remove('winning-cell');
            });
            newGameBtn.textContent = "Restart Game";
        });
    }

    function startAiVsAiGame() {
        // Make moves every 1 second
        aiMoveInterval = setInterval(() => {
            fetch('/make_move', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    ai_vs_ai: true,
                    use_minimax: useMinimax 
                }),
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
                    clearInterval(aiMoveInterval);
                    handleGameEnd(data.status, data.winning_coordinates);
                }
            });
        }, 1000);
    }

    function handleCellClick(event) {
        if (isAiVsAi && !waitingForPlayerMove) return; // Ignore clicks in AI vs AI mode unless waiting for player move
        if (!waitingForPlayerMove) return; // Ignore clicks when it's not player's turn

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
            body: JSON.stringify({ 
                x, 
                y, 
                ai_vs_ai: isAiVsAi,
                use_minimax: useMinimax,
                player_first: playerGoesFirst
            }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                gameStatus.textContent = data.error;
                return;
            }

            currentBoard = data.board;
            updateBoard();
            waitingForPlayerMove = false;

            if (data.status) {
                handleGameEnd(data.status, data.winning_coordinates);
            } else if (isAiVsAi) {
                gameStatus.textContent = "AI vs AI Mode - Game in progress...";
                startAiVsAiGame();
            } else {
                gameStatus.textContent = "AI is thinking...";
                // Get AI's move
                fetch('/make_move', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ 
                        ai_vs_ai: false,
                        use_minimax: useMinimax,
                        player_first: playerGoesFirst
                    }),
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
                    } else {
                        gameStatus.textContent = playerGoesFirst ? "Your turn! (You are X)" : "Your turn! (You are O)";
                        waitingForPlayerMove = true;
                    }
                });
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