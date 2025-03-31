// Game state management
const gameState = {
    board: Array(9).fill(''),
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
    started: false  // New flag to track if game has officially started
};

// AI API URLs - Using relative paths instead of absolute URLs
const MINIMAX_API = '/minimax_move';
const QLEARNING_API = '/make_move';

// DOM elements
const boardElement = document.getElementById('game-board');
const cells = document.querySelectorAll('.cell');
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

// Initialize game
function initGame() {
    gameState.board = Array(9).fill('');
    gameState.currentPlayer = 'X';
    gameState.gameOver = false;
    gameState.winner = null;
    gameState.started = true;  // Mark game as started
    
    cells.forEach(cell => {
        cell.textContent = '';
        cell.style.backgroundColor = '#f0f0f0';
    });
    
    currentPlayerElement.textContent = 'X';
    gameStatusElement.textContent = 'In Progress';
    
    updateUI();
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

// Update UI
function updateUI() {
    cells.forEach((cell, index) => {
        cell.textContent = gameState.board[index];
    });
    
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
    
    xWinsElement.textContent = gameState.stats.xWins;
    oWinsElement.textContent = gameState.stats.oWins;
    drawsElement.textContent = gameState.stats.draws;
}

// Check winner
function checkWinner(board) {
    const winPatterns = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], // rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8], // columns
        [0, 4, 8], [2, 4, 6]             // diagonals
    ];
    
    for (const pattern of winPatterns) {
        const [a, b, c] = pattern;
        if (board[a] && board[a] === board[b] && board[a] === board[c]) {
            return board[a];
        }
    }
    
    return null;
}

// Check if game is over
function checkGameOver() {
    const winner = checkWinner(gameState.board);
    
    if (winner) {
        gameState.gameOver = true;
        gameState.winner = winner;
        
        if (winner === 'X') {
            gameState.stats.xWins++;
            addLog(`Game over: X wins!`);
        } else {
            gameState.stats.oWins++;
            addLog(`Game over: O wins!`);
        }
        
        updateUI();
        return true;
    }
    
    if (gameState.board.every(cell => cell !== '')) {
        gameState.gameOver = true;
        gameState.stats.draws++;
        addLog(`Game over: Draw!`);
        updateUI();
        return true;
    }
    
    return false;
}

// Make move
function makeMove(index) {
    if (gameState.gameOver || gameState.board[index] !== '') return false;
    
    gameState.board[index] = gameState.currentPlayer;
    addLog(`${gameState.currentPlayer} selected position ${index}`);
    
    if (!checkGameOver()) {
        gameState.currentPlayer = gameState.currentPlayer === 'X' ? 'O' : 'X';
        updateUI();
        setTimeout(checkAITurn, 500);
    } else if (gameState.autoPlaying && gameState.autoPlayGames > 0) {
        gameState.autoPlayGames--;
        setTimeout(initGame, 100);
    }
    
    return true;
}

// Handle player click
function handleCellClick(event) {
    // Don't allow moves if game hasn't started
    if (!gameState.started) return;
    
    const index = parseInt(event.target.getAttribute('data-index'));
    
    if (gameState.currentPlayer === 'X' && playerXSelect.value === 'human' ||
        gameState.currentPlayer === 'O' && playerOSelect.value === 'human') {
        makeMove(index);
    }
}

// Get current player's AI type
function getCurrentAI() {
    if (gameState.currentPlayer === 'X') {
        return playerXSelect.value;
    } else {
        return playerOSelect.value;
    }
}

// Check if it's AI's turn
function checkAITurn() {
    // Don't process AI moves if game hasn't started
    if (gameState.gameOver || !gameState.started) return;
    
    const currentPlayerType = gameState.currentPlayer === 'X' 
        ? playerXSelect.value 
        : playerOSelect.value;
    
    if (currentPlayerType !== 'human') {
        getAIMove(currentPlayerType);
    }
}

// Get AI move
async function getAIMove(aiType) {
    try {
        let response;
        let move;
        
        aiAnalysisElement.textContent = `${aiType} AI is thinking...`;
        
        if (aiType === 'minimax') {
            // Convert board format for Minimax algorithm API
            const minimaxBoard = gameState.board.map(cell => {
                if (cell === 'X') return 1;
                if (cell === 'O') return -1;
                return 0;
            });
            
            response = await fetch(MINIMAX_API, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ board: minimaxBoard })
            });
            
            const data = await response.json();
            if (data.error) {
                console.error('Minimax API error:', data.error);
                aiAnalysisElement.textContent = `Minimax API error: ${data.error}`;
                return;
            }
            
            // API returns move format as [row, col]
            const [row, col] = data.move;
            move = row * 3 + col;
            
            aiAnalysisElement.textContent = `Minimax AI evaluation: ${data.evaluation || 'N/A'}`;
        } else if (aiType === 'qlearning') {
            response = await fetch(QLEARNING_API, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ board: gameState.board })
            });
            
            const data = await response.json();
            if (data.error) {
                console.error('Q-Learning API error:', data.error);
                aiAnalysisElement.textContent = `Q-Learning API error: ${data.error}`;
                return;
            }
            
            move = data.move;
            
            // Display Q-learning analysis
            if (data.analysis) {
                aiAnalysisElement.innerHTML = `
                    <p>Q-Learning Analysis:</p>
                    <p>Selected position: ${move}</p>
                    <p>Threat positions: ${data.analysis.threats ? data.analysis.threats.join(', ') : 'None'}</p>
                    <p>Opportunity positions: ${data.analysis.opportunities ? data.analysis.opportunities.join(', ') : 'None'}</p>
                `;
            } else {
                aiAnalysisElement.textContent = `Q-Learning AI selected position ${move}`;
            }
        }
        
        // Execute AI's move
        if (move !== undefined) {
            setTimeout(() => makeMove(move), 300);
        }
    } catch (error) {
        console.error('Error getting AI move:', error);
        aiAnalysisElement.textContent = `Error getting AI move: ${error.message}`;
    }
}

// Start auto play
function startAutoPlay() {
    if (playerXSelect.value === 'human' || playerOSelect.value === 'human') {
        alert('Auto play requires two AI players');
        return;
    }
    
    gameState.autoPlaying = true;
    gameState.autoPlayGames = 100;
    autoPlayButton.disabled = true;
    startButton.disabled = true;
    
    addLog('Starting auto play (100 rounds)');
    initGame();
}

// Event listeners
cells.forEach(cell => {
    cell.addEventListener('click', handleCellClick);
});

startButton.addEventListener('click', () => {
    startButton.disabled = true;
    autoPlayButton.disabled = false;
    initGame();
});

resetButton.addEventListener('click', () => {
    gameState.autoPlaying = false;
    gameState.autoPlayGames = 0;
    gameState.started = false;  // Reset game started status
    startButton.disabled = false;
    autoPlayButton.disabled = false;
    
    gameState.stats.xWins = 0;
    gameState.stats.oWins = 0;
    gameState.stats.draws = 0;
    
    // Clear the board without starting a new game
    gameState.board = Array(9).fill('');
    gameState.currentPlayer = 'X';
    gameState.gameOver = false;
    gameState.winner = null;
    
    cells.forEach(cell => {
        cell.textContent = '';
        cell.style.backgroundColor = '#f0f0f0';
    });
    
    currentPlayerElement.textContent = 'X';
    gameStatusElement.textContent = 'Not Started';
    aiAnalysisElement.textContent = 'Game not started';
    
    updateUI();
});

autoPlayButton.addEventListener('click', startAutoPlay);

// Display initial state without starting the game
updateUI();