// Three.js setup
let scene, camera, renderer, controls;
let board, pegs = [];
let pieces = [];
let raycaster, mouse;
let selectedPiece = null;
let isDragging = false;
let hoveredPeg = null;

// Game state
let currentBoard = null;
let previousBoard = null;  // Add this to track previous board state
let isAiVsAi = false;
let useMinimax = false;
let playerGoesFirst = true;
let aiMoveInterval = null;
let waitingForPlayerMove = false;
let isSpacePressed = false; // Add this to track space bar state

// Animation state
let animatingPieces = new Set();
const ANIMATION_DURATION = 500; // Duration in milliseconds
const ANIMATION_STEPS = 30; // Number of steps in the animation

// Constants
const PEG_HEIGHT = 3;
const PEG_RADIUS = 0.2;
const BOARD_SIZE = 4;
const PIECE_RADIUS = 0.3;
const PIECE_HEIGHT = 0.5;
const PIECE_OFFSET = 0.1; // Small offset to prevent z-fighting
const BOARD_HEIGHT = 0.5; // Height of the board

function init() {
    // Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);

    // Camera setup
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(8, 8, 8);
    camera.lookAt(0, 0, 0);

    // Renderer setup
    const canvas = document.getElementById('game-canvas');
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    // Controls setup
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 5;
    controls.maxDistance = 15;
    controls.enabled = false; // Start with controls disabled

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
    directionalLight.position.set(5, 8, 5);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 1024;
    directionalLight.shadow.mapSize.height = 1024;
    directionalLight.shadow.camera.near = 0.5;
    directionalLight.shadow.camera.far = 50;
    directionalLight.shadow.camera.left = -10;
    directionalLight.shadow.camera.right = 10;
    directionalLight.shadow.camera.top = 10;
    directionalLight.shadow.camera.bottom = -10;
    scene.add(directionalLight);

    const secondaryLight = new THREE.DirectionalLight(0xffffff, 0.5);
    secondaryLight.position.set(-5, 8, -5);
    scene.add(secondaryLight);

    // Create board
    createBoard();
    createPegs();

    // Raycaster setup for interaction
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    // Event listeners
    window.addEventListener('resize', onWindowResize, false);
    renderer.domElement.addEventListener('mousedown', onMouseDown, false);
    renderer.domElement.addEventListener('mousemove', onMouseMove, false);
    renderer.domElement.addEventListener('mouseup', onMouseUp, false);

    // Add keyboard event listeners
    window.addEventListener('keydown', onKeyDown, false);
    window.addEventListener('keyup', onKeyUp, false);

    // Start animation loop
    animate();
}

function createBoard() {
    const geometry = new THREE.BoxGeometry(BOARD_SIZE, 0.5, BOARD_SIZE);
    const material = new THREE.MeshPhongMaterial({ color: 0x8b4513 });
    board = new THREE.Mesh(geometry, material);
    board.receiveShadow = true;
    scene.add(board);
}

function createPegs() {
    const pegGeometry = new THREE.CylinderGeometry(PEG_RADIUS, PEG_RADIUS, PEG_HEIGHT, 32);
    const originalColor = 0x8b4513;
    const pegMaterial = new THREE.MeshPhongMaterial({ 
        color: originalColor,
        shininess: 50,
        specular: 0x444444
    });

    for (let x = -1; x <= 1; x++) {
        for (let y = -1; y <= 1; y++) {
            const peg = new THREE.Mesh(pegGeometry, pegMaterial.clone());
            peg.position.set(x, PEG_HEIGHT / 2, y);
            peg.castShadow = true;
            peg.receiveShadow = true;
            peg.userData = { 
                x, 
                y,
                originalColor: originalColor
            };
            pegs.push(peg);
            scene.add(peg);
        }
    }
}

function createPiece(color, position, animate = true) {
    console.log('Creating piece with color:', color, 'at position:', position);
    const geometry = new THREE.CylinderGeometry(PIECE_RADIUS, PIECE_RADIUS, PIECE_HEIGHT, 32);
    const material = new THREE.MeshPhongMaterial({ 
        color,
        shininess: 100,
        specular: 0x444444
    });
    const piece = new THREE.Mesh(geometry, material);
    
    // Calculate the actual position (wrapping around the peg)
    const actualPosition = new THREE.Vector3(
        position.x,
        position.y, // Use the provided y position directly
        position.z
    );
    
    if (animate) {
        // Start position above the peg
        piece.position.set(actualPosition.x, actualPosition.y + PEG_HEIGHT, actualPosition.z);
        piece.userData = { 
            x: position.x, 
            y: position.z, 
            z: (position.y - BOARD_HEIGHT) / (PIECE_HEIGHT + PIECE_OFFSET),
            value: color === 0xff0000 ? 1 : -1,
            targetPosition: actualPosition.clone(),
            animationStart: Date.now()
        };
        animatingPieces.add(piece);
    } else {
        piece.position.copy(actualPosition);
        piece.userData = { 
            x: position.x, 
            y: position.z, 
            z: (position.y - BOARD_HEIGHT) / (PIECE_HEIGHT + PIECE_OFFSET),
            value: color === 0xff0000 ? 1 : -1,
            originalPosition: actualPosition.clone()
        };
    }
    
    piece.castShadow = true;
    piece.receiveShadow = true;
    pieces.push(piece);
    scene.add(piece);
    return piece;
}

function onWindowResize() {
    const canvas = renderer.domElement;
    camera.aspect = canvas.clientWidth / canvas.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
}

function onKeyDown(event) {
    if (event.code === 'Space') {
        isSpacePressed = true;
        controls.enabled = true;
    }
}

function onKeyUp(event) {
    if (event.code === 'Space') {
        isSpacePressed = false;
        if (!isAiVsAi && waitingForPlayerMove) {
            controls.enabled = false;
        }
    }
}

function onMouseDown(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    
    if (waitingForPlayerMove) {
        // Check if clicking on a peg
        const pegIntersects = raycaster.intersectObjects(pegs);
        if (pegIntersects.length > 0) {
            const peg = pegIntersects[0].object;
            const x = peg.userData.x + 1; // Convert from -1 to 1 to 0 to 2
            const y = peg.userData.y + 1;
            
            console.log('Clicked peg at:', x, y);
            
            // Make the move
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
                    controls.enabled = isSpacePressed; // Only enable if space is pressed
                } else if (isAiVsAi) {
                    gameStatus.textContent = "AI vs AI Mode - Game in progress...";
                    controls.enabled = true; // Enable controls for AI vs AI mode
                    startAiVsAiGame();
                } else {
                    gameStatus.textContent = "AI is thinking...";
                    controls.enabled = isSpacePressed; // Only enable if space is pressed
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
                            controls.enabled = isSpacePressed; // Only enable if space is pressed
                        } else {
                            gameStatus.textContent = playerGoesFirst ? "Your turn! (You are X)" : "Your turn! (You are O)";
                            waitingForPlayerMove = true;
                            controls.enabled = false; // Disable controls during player's turn
                        }
                    });
                }
            });
        }
    } else {
        // Handle piece dragging
        const intersects = raycaster.intersectObjects(pieces);
        if (intersects.length > 0) {
            selectedPiece = intersects[0].object;
            isDragging = true;
            controls.enabled = false;
        }
    }
}

function onMouseMove(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);

    if (isDragging && selectedPiece) {
        // Handle piece dragging
        const intersects = raycaster.intersectObject(board);
        if (intersects.length > 0) {
            selectedPiece.position.x = intersects[0].point.x;
            selectedPiece.position.z = intersects[0].point.z;
            selectedPiece.position.y = PEG_HEIGHT + PIECE_HEIGHT / 2;
        }
    } else if (waitingForPlayerMove) {
        // Handle peg hover effect
        const pegIntersects = raycaster.intersectObjects(pegs);
        
        // Reset previous hovered peg if it exists
        if (hoveredPeg) {
            hoveredPeg.material.color.setHex(hoveredPeg.userData.originalColor);
        }
        
        // Set new hovered peg
        if (pegIntersects.length > 0) {
            hoveredPeg = pegIntersects[0].object;
            hoveredPeg.material.color.setHex(0xa0522d); // Lighter brown for hover
        } else {
            hoveredPeg = null;
        }
    }
}

function onMouseUp() {
    if (!isDragging || !selectedPiece) return;

    // Find nearest peg
    let nearestPeg = null;
    let minDistance = Infinity;

    pegs.forEach(peg => {
        const distance = selectedPiece.position.distanceTo(peg.position);
        if (distance < minDistance) {
            minDistance = distance;
            nearestPeg = peg;
        }
    });

    if (nearestPeg && minDistance < 1) {
        // Snap to peg at board height
        selectedPiece.position.copy(nearestPeg.position);
        selectedPiece.position.y = BOARD_HEIGHT; // Place at board height
    } else {
        // Return to original position
        selectedPiece.position.copy(selectedPiece.userData.originalPosition);
    }

    isDragging = false;
    selectedPiece = null;
    controls.enabled = isSpacePressed; // Only enable if space is pressed
}

function animate() {
    requestAnimationFrame(animate);
    
    // Update piece animations
    const currentTime = Date.now();
    animatingPieces.forEach(piece => {
        const elapsed = currentTime - piece.userData.animationStart;
        const progress = Math.min(elapsed / ANIMATION_DURATION, 1);
        
        // Easing function for smooth animation
        const easeOutBounce = (x) => {
            const n1 = 7.5625;
            const d1 = 2.75;
            if (x < 1 / d1) {
                return n1 * x * x;
            } else if (x < 2 / d1) {
                return n1 * (x -= 1.5 / d1) * x + 0.75;
            } else if (x < 2.5 / d1) {
                return n1 * (x -= 2.25 / d1) * x + 0.9375;
            } else {
                return n1 * (x -= 2.625 / d1) * x + 0.984375;
            }
        };
        
        const easedProgress = easeOutBounce(progress);
        
        // Interpolate position
        piece.position.lerp(piece.userData.targetPosition, easedProgress);
        
        // Remove from animating set when done
        if (progress >= 1) {
            animatingPieces.delete(piece);
            piece.userData.originalPosition = piece.position.clone();
        }
    });
    
    controls.update();
    renderer.render(scene, camera);
}

// Initialize the game
document.addEventListener('DOMContentLoaded', () => {
    init();
    
    // Game controls setup
    const gameStatus = document.getElementById('gameStatus');
    const newGameBtn = document.getElementById('newGameBtn');
    const aiVsAiToggle = document.getElementById('aiVsAiToggle');
    const minimaxToggle = document.getElementById('minimaxToggle');
    const playerFirstToggle = document.getElementById('playerFirstToggle');

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

    // Start the game automatically when the page loads
    startNewGame();
});

function startNewGame() {
    // Clear any existing AI move interval
    if (aiMoveInterval) {
        clearInterval(aiMoveInterval);
    }
    waitingForPlayerMove = false;
    previousBoard = null;  // Reset previous board state

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
        console.log('Received game state:', data);
        currentBoard = data.board;
        updateBoard();
        if (isAiVsAi) {
            gameStatus.textContent = "AI vs AI Mode - Game in progress...";
            controls.enabled = true; // Enable controls for AI vs AI mode
            startAiVsAiGame();
        } else {
            if (playerGoesFirst) {
                gameStatus.textContent = "Your turn! (You are X)";
                waitingForPlayerMove = true;
                controls.enabled = false; // Disable controls during player's turn
            } else {
                if (data.bot_move) {
                    // AI has made its first move
                    gameStatus.textContent = "Your turn! (You are O)";
                    waitingForPlayerMove = true;
                    controls.enabled = false; // Disable controls during player's turn
                }
            }
        }
        newGameBtn.textContent = "Restart Game";
    })
    .catch(error => {
        console.error('Error starting new game:', error);
        gameStatus.textContent = "Error starting game. Please try again.";
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

function updateBoard() {
    console.log('Updating board with state:', currentBoard);
    
    // Clear existing pieces
    pieces.forEach(piece => {
        scene.remove(piece);
    });
    pieces = [];
    animatingPieces.clear();

    // Create new pieces based on the board state
    for (let x = 0; x < 3; x++) {
        for (let y = 0; y < 3; y++) {
            for (let z = 0; z < 3; z++) {
                const value = currentBoard[x][y][z];
                if (value !== 0) {
                    console.log(`Creating piece at (${x}, ${y}, ${z}) with value ${value}`);
                    // Convert board coordinates to 3D space coordinates
                    const position = new THREE.Vector3(
                        x - 1,  // Convert from 0-2 to -1 to 1
                        BOARD_HEIGHT + (z * (PIECE_HEIGHT + PIECE_OFFSET)), // Start from board height and stack up
                        y - 1   // Convert from 0-2 to -1 to 1
                    );
                    
                    // Create piece with appropriate color
                    const color = value === 1 ? 0xff0000 : 0x0000ff; // Red for X, Blue for O
                    
                    // Only animate if this is a new piece (not present in previous board)
                    const shouldAnimate = !previousBoard || previousBoard[x][y][z] === 0;
                    createPiece(color, position, shouldAnimate);
                }
            }
        }
    }
    
    // Update previous board state
    previousBoard = JSON.parse(JSON.stringify(currentBoard));
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

    // Highlight winning pieces if there are winning coordinates
    if (winningCoordinates) {
        highlightWinningPieces(winningCoordinates);
    }
}

function highlightWinningPieces(coordinates) {
    // Remove previous highlights
    pieces.forEach(piece => {
        piece.material.color.setHex(piece.userData.value === 1 ? 0xff0000 : 0x0000ff);
    });

    // Add highlights to winning pieces
    coordinates.forEach(([x, y, z]) => {
        pieces.forEach(piece => {
            // The piece's userData.x and userData.y are already in the -1 to 1 range
            // and match the peg positions
            if (piece.userData.x === (x - 1) && 
                piece.userData.y === (y - 1) && 
                piece.userData.z === z) {
                piece.material.color.setHex(0x00ff00); // Green highlight for winning pieces
            }
        });
    });
} 