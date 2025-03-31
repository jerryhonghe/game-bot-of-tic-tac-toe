// Game state management
const gameState = {
    // 6x6 board
    board: Array(6).fill().map(() => Array(6).fill('')),
    currentPlayer: 'X',
    gameOver: false,
    winner: null,
    stats: {
        xWins: 0,
        oWins: 0,
        draws: 0
    },
    autoPlayGames: 0,
    autoPlaying: false,
    started: false  // Flag to mark if the game has started
};

// Game constants
const ROWS = 6;
const COLS = 6;

// AI API URLs - using relative paths
const ALPHAZERO_API = '/alphazero_move';
const MCTS_API = '/mcts_pure_move';

// DOM elements
const boardElement = document.getElementById('game-board');
const startButton = document.getElementById('start-game');
const resetButton = document.getElementById('reset-game');
const autoPlayButton = document.getElementById('auto-play');
const playerXSelect = document.getElementById('player-x');
const playerOSelect = document.getElementById('player-o');
const currentPlayerElement = document.getElementById('current-player');
const gameStatusElement = document.getElementById('game-status');
const xWinsElement = document.getElementById('x-wins');
const oWinsElement = document.getElementById('o-wins');
const drawsElement = document.getElementById('draws');
const gameLogElement = document.getElementById('game-log');
const aiAnalysisElement = document.getElementById('ai-analysis');

// Initialize game board
function createBoard() {
    console.log("Creating game board...");
    
    // Clear existing board
    boardElement.innerHTML = '';
    
    // Create board
    for (let row = 0; row < ROWS; row++) {
        for (let col = 0; col < COLS; col++) {
            const cell = document.createElement('div');
            cell.className = 'cell empty';
            cell.setAttribute('data-row', row);
            cell.setAttribute('data-col', col);
            
            // Add click event for each cell
            cell.addEventListener('click', () => handleCellClick(row, col));
            
            boardElement.appendChild(cell);
        }
    }
    
    console.log(`Created a board with ${ROWS} rows and ${COLS} columns`);
}

// Initialize game
function initGame() {
    console.log("Initializing game...");
    
    // Reset game state
    gameState.board = Array(ROWS).fill().map(() => Array(COLS).fill(''));
    gameState.currentPlayer = 'X';
    gameState.gameOver = false;
    gameState.winner = null;
    gameState.started = true;
    
    console.log("Game state has been reset");
    
    // Create board
    createBoard();
    
    currentPlayerElement.textContent = 'X';
    gameStatusElement.textContent = 'In Progress';
    
    console.log("UI elements updated");
    
    // Update entire UI
    updateUI();
    
    // Add initial game log
    addLog('Game started. Player X goes first.');
    
    console.log("Starting player:", gameState.currentPlayer);
    console.log("Player X type:", playerXSelect.value);
    console.log("Player O type:", playerOSelect.value);
    
    // Check if AI should go first
    checkAITurn();
}

// Add log
function addLog(message) {
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    logEntry.textContent = message;
    gameLogElement.appendChild(logEntry);
    gameLogElement.scrollTop = gameLogElement.scrollHeight;
}

// Update UI based on current game state
function updateUI() {
    // Update each cell
    const cells = boardElement.querySelectorAll('.cell');
    cells.forEach(cell => {
        const row = parseInt(cell.getAttribute('data-row'));
        const col = parseInt(cell.getAttribute('data-col'));
        
        // Reset class name
        cell.className = 'cell';
        
        // Add player marker or keep empty
        if (gameState.board[row][col] === 'X') {
            cell.classList.add('X');
            cell.textContent = 'X';
        } else if (gameState.board[row][col] === 'O') {
            cell.classList.add('O');
            cell.textContent = 'O';
        } else {
            cell.classList.add('empty');
            cell.textContent = '';
        }
    });
    
    // Update status elements
    currentPlayerElement.textContent = gameState.currentPlayer;
    
    if (gameState.gameOver) {
        if (gameState.winner) {
            gameStatusElement.textContent = `${gameState.winner} Wins!`;
        } else {
            gameStatusElement.textContent = 'Draw!';
        }
    } else if (gameState.started) {
        gameStatusElement.textContent = 'In Progress';
    } else {
        gameStatusElement.textContent = 'Not Started';
    }
    
    // Update statistics
    xWinsElement.textContent = gameState.stats.xWins;
    oWinsElement.textContent = gameState.stats.oWins;
    drawsElement.textContent = gameState.stats.draws;
}

// Handle cell click
function handleCellClick(row, col) {
    console.log(`Click at position [${row}, ${col}]`);
    
    // If the game hasn't started or it's not a human player's turn, don't allow placement
    if (!gameState.started) {
        console.log('Game has not started yet');
        return;
    }
    
    if (gameState.gameOver) {
        console.log('Game is over');
        return;
    }
    
    const currentAI = gameState.currentPlayer === 'X' ?
        playerXSelect.value : playerOSelect.value;
        
    if (currentAI !== 'human') {
        console.log(`Current player (${gameState.currentPlayer}) is controlled by AI (${currentAI}), not a human player`);
        return;
    }
    
    // Check if the position is already occupied
    if (gameState.board[row][col] !== '') {
        console.log(`Position [${row}, ${col}] is already occupied`);
        return;
    }
    
    console.log(`Making move at position [${row}, ${col}]`);
    makeMove(row, col);
}

// Make a move
function makeMove(row, col) {
    // If position is already occupied, ignore this move
    if (gameState.board[row][col] !== '') {
        console.log(`Cannot place: Position [${row}, ${col}] is already occupied`);
        return false;
    }
    
    console.log(`Placing ${gameState.currentPlayer} piece at position [${row}, ${col}]`);
    
    // Update game state
    gameState.board[row][col] = gameState.currentPlayer;
    
    // Record this move
    addLog(`${gameState.currentPlayer} placed piece at position [${row+1}, ${col+1}]`);
    
    // Update UI
    updateUI();
    
    // Check for win or draw
    if (!checkGameOver()) {
        // Switch player
        gameState.currentPlayer = gameState.currentPlayer === 'X' ? 'O' : 'X';
        console.log(`Switching to player: ${gameState.currentPlayer}`);
        updateUI();
        
        // Check if it's AI's turn
        setTimeout(checkAITurn, 500);
    } else if (gameState.autoPlaying && gameState.autoPlayGames > 0) {
        gameState.autoPlayGames--;
        setTimeout(initGame, 300);
    }
    
    return true;
}

// Check for win
function checkWin(player) {
    // Function checks if position is valid and contains player's piece
    function checkPos(row, col) {
        return row >= 0 && row < ROWS && 
               col >= 0 && col < COLS && 
               gameState.board[row][col] === player;
    }
    
    // Check horizontal
    for (let row = 0; row < ROWS; row++) {
        for (let col = 0; col <= COLS - 4; col++) {
            if (checkPos(row, col) && 
                checkPos(row, col + 1) && 
                checkPos(row, col + 2) && 
                checkPos(row, col + 3)) {
                return true;
            }
        }
    }
    
    // Check vertical
    for (let row = 0; row <= ROWS - 4; row++) {
        for (let col = 0; col < COLS; col++) {
            if (checkPos(row, col) && 
                checkPos(row + 1, col) && 
                checkPos(row + 2, col) && 
                checkPos(row + 3, col)) {
                return true;
            }
        }
    }
    
    // Check diagonal (positive slope)
    for (let row = 3; row < ROWS; row++) {
        for (let col = 0; col <= COLS - 4; col++) {
            if (checkPos(row, col) && 
                checkPos(row - 1, col + 1) && 
                checkPos(row - 2, col + 2) && 
                checkPos(row - 3, col + 3)) {
                return true;
            }
        }
    }
    
    // Check diagonal (negative slope)
    for (let row = 0; row <= ROWS - 4; row++) {
        for (let col = 0; col <= COLS - 4; col++) {
            if (checkPos(row, col) && 
                checkPos(row + 1, col + 1) && 
                checkPos(row + 2, col + 2) && 
                checkPos(row + 3, col + 3)) {
                return true;
            }
        }
    }
    
    return false;
}

// Check if game is over (win or draw)
function checkGameOver() {
    // Check for win
    if (checkWin(gameState.currentPlayer)) {
        gameState.gameOver = true;
        gameState.winner = gameState.currentPlayer;
        
        // Update statistics
        if (gameState.currentPlayer === 'X') {
            gameState.stats.xWins++;
            addLog('Game over: X wins!');
        } else {
            gameState.stats.oWins++;
            addLog('Game over: O wins!');
        }
        
        updateUI();
        return true;
    }
    
    // Check for draw - if all cells are filled
    let isDraw = true;
    for (let row = 0; row < ROWS; row++) {
        for (let col = 0; col < COLS; col++) {
            if (gameState.board[row][col] === '') {
                isDraw = false;
                break;
            }
        }
        if (!isDraw) break;
    }
    
    if (isDraw) {
        gameState.gameOver = true;
        gameState.stats.draws++;
        addLog('Game over: Draw!');
        updateUI();
        return true;
    }
    
    return false;
}

// Check if it's AI's turn
function checkAITurn() {
    // If game hasn't started or is over, don't process AI move
    if (gameState.gameOver || !gameState.started) return;
    
    const currentPlayerType = gameState.currentPlayer === 'X' ?
        playerXSelect.value : playerOSelect.value;
    
    console.log(`Checking if it's AI's turn. Current player: ${gameState.currentPlayer}, Type: ${currentPlayerType}`);
    
    if (currentPlayerType !== 'human') {
        getAIMove(currentPlayerType);
    }
}

// AI move handling function part (only showing the parts that need modification)

async function getAIMove(aiType) {
    try {
        console.log(`Getting move for ${aiType} AI...`);
        
        let response;
        let move;
        
        aiAnalysisElement.textContent = `${aiType} AI is thinking...`;
        
        if (aiType === 'alphazero') {
            // Convert board format for AlphaZero API
            // Map X to red, O to yellow to be compatible with backend
            const backendBoard = gameState.board.map(row => 
                row.map(cell => {
                    if (cell === 'X') return 'red';
                    if (cell === 'O') return 'yellow';
                    return '';
                })
            );
            
            const boardState = {
                board: backendBoard,
                current_player: gameState.currentPlayer === 'X' ? 'red' : 'yellow'  // Mapping for backend
            };
            
            console.log("Sending request to AlphaZero API...");
            
            response = await fetch(ALPHAZERO_API, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(boardState)
            });
            
            const data = await response.json();
            if (data.error) {
                console.error('AlphaZero API error:', data.error);
                aiAnalysisElement.textContent = `AlphaZero API error: ${data.error}`;
                return;
            }
            
            // Directly use row and col coordinates returned by server
            move = data.move;
            console.log(`AlphaZero AI chose position [${move.row}, ${move.col}]`);
            aiAnalysisElement.textContent = `AlphaZero AI evaluation: ${data.evaluation || 'N/A'}`;
            
        } else if (aiType === 'mcts') {
            // Convert board format for MCTS API
            // Map X to red, O to yellow to be compatible with backend
            const backendBoard = gameState.board.map(row => 
                row.map(cell => {
                    if (cell === 'X') return 'red';
                    if (cell === 'O') return 'yellow';
                    return '';
                })
            );
            
            const boardState = {
                board: backendBoard,
                current_player: gameState.currentPlayer === 'X' ? 'red' : 'yellow'  // Mapping for backend
            };
            
            console.log("Sending request to MCTS API...");
            
            response = await fetch(MCTS_API, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(boardState)
            });
            
            const data = await response.json();
            if (data.error) {
                console.error('MCTS API error:', data.error);
                aiAnalysisElement.textContent = `MCTS API error: ${data.error}`;
                return;
            }
            
            // Directly use row and col coordinates returned by server
            move = data.move;
            console.log(`MCTS AI chose position [${move.row}, ${move.col}]`);
            aiAnalysisElement.textContent = `MCTS AI chose position [${move.row + 1}, ${move.col + 1}]`;
        }
        
        // Execute AI's move (add small delay to improve user experience)
        if (move !== undefined) {
            console.log(`AI will place piece at position [${move.row}, ${move.col}]`);
            setTimeout(() => makeMove(move.row, move.col), 300);
        } else {
            console.error("AI returned undefined position");
        }
    } catch (error) {
        console.error('Error getting AI move:', error);
        aiAnalysisElement.textContent = `Error getting AI move: ${error.message}`;
    }
}

// Find all available cells in a column
function findAvailableCellsInColumn(col) {
    const cells = [];
    for (let row = 0; row < ROWS; row++) {
        if (gameState.board[row][col] === '') {
            cells.push({ row, col });
        }
    }
    return cells;
}

// Select the most suitable cell for AI (in this modified version, we randomly select an available cell)
function selectCellForAI(availableCells) {
    return availableCells[Math.floor(Math.random() * availableCells.length)];
}

// Start auto play
function startAutoPlay() {
    if (playerXSelect.value === 'human' || playerOSelect.value === 'human') {
        alert('Auto play requires two AI players');
        return;
    }
    
    gameState.autoPlaying = true;
    gameState.autoPlayGames = 10;
    autoPlayButton.disabled = true;
    startButton.disabled = true;
    
    addLog('Starting auto play (10 rounds)');
    initGame();
}

// Event listeners for game control buttons
startButton.addEventListener('click', () => {
    console.log("Start button clicked");
    startButton.disabled = true;
    autoPlayButton.disabled = false;
    initGame();
});

resetButton.addEventListener('click', () => {
    console.log("Reset button clicked");
    gameState.autoPlaying = false;
    gameState.autoPlayGames = 0;
    gameState.started = false;
    startButton.disabled = false;
    autoPlayButton.disabled = false;
    
    // Reset statistics
    gameState.stats.xWins = 0;
    gameState.stats.oWins = 0;
    gameState.stats.draws = 0;
    
    // Clear board
    gameState.board = Array(ROWS).fill().map(() => Array(COLS).fill(''));
    gameState.currentPlayer = 'X';
    gameState.gameOver = false;
    gameState.winner = null;
    
    // Update UI
    createBoard();
    currentPlayerElement.textContent = 'X';
    gameStatusElement.textContent = 'Not Started';
    aiAnalysisElement.textContent = 'Game not started';
    
    updateUI();
    
    // Clear log
    gameLogElement.innerHTML = '<div class="log-entry">Game log will be displayed here</div>';
});

autoPlayButton.addEventListener('click', startAutoPlay);

// Initialize board
console.log("Initial page load - creating board");
createBoard();
updateUI();