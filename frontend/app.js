// Application State
const state = {
    sessionId: `session_${Math.random().toString(36).substr(2, 9)}`,
    isBusy: false,
    apiUrl: window.location.origin === 'http://localhost:8000' || window.location.origin === 'http://127.0.0.1:8000' 
            ? window.location.origin + '/api' 
            : '/api' // Relative path for production
};

// DOM Elements
const chatMessages = document.getElementById('chatMessages');
const chatForm = document.getElementById('chatForm');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const serverStatus = document.getElementById('serverStatus');
const statusText = document.getElementById('statusText');
const sourcesHistory = document.getElementById('sourcesHistory');

// Initial Setup
document.addEventListener('DOMContentLoaded', () => {
    checkServerStatus();
    userInput.focus();
    
    // Auto-resize textarea
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = (userInput.scrollHeight) + 'px';
    });

    // History shortcuts
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });
});

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = userInput.value.trim();
    if (!query || state.isBusy) return;

    // Send the query
    await sendQuery(query);
});

clearHistoryBtn.addEventListener('click', async () => {
    if (confirm("Clear chat history? This won't delete the documents, only the context of this conversation.")) {
        try {
            await fetch(`${state.apiUrl}/chat/clear/${state.sessionId}`, { method: 'DELETE' });
            chatMessages.innerHTML = '';
            addWelcomeMessage();
            sourcesHistory.innerHTML = '<div class="empty-sources">Ask a question to see citations</div>';
            showNotification('History cleared', 'success');
        } catch (err) {
            console.error('Clear error:', err);
        }
    }
});

// Main Send Query Function
async function sendQuery(query) {
    if (state.isBusy) return;
    
    // Update local UI
    state.isBusy = true;
    appendMessage('user', query);
    userInput.value = '';
    userInput.style.height = 'auto';
    setLoadingState(true);

    const typingId = appendLoadingIndicator();

    try {
        const response = await fetch(`${state.apiUrl}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                session_id: state.sessionId
            })
        });

        const data = await response.json();
        removeLoadingIndicator(typingId);

        if (response.ok) {
            appendMessage('assistant', data.answer);
            updateCitations(data.citations);
        } else {
            // Handle cases where data.detail might be an object or array (FastAPI validation)
            const errorMsg = typeof data.detail === 'object' 
                ? JSON.stringify(data.detail, null, 2) 
                : (data.detail || 'Something went wrong.');
            appendMessage('assistant', `Error: ${errorMsg}`);
        }
    } catch (err) {
        removeLoadingIndicator(typingId);
        appendMessage('assistant', 'Connection error. Is the backend server running?');
        console.error('Fetch error:', err);
    } finally {
        state.isBusy = false;
        setLoadingState(false);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

// UI Helpers
function appendMessage(role, text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;
    
    // Convert newlines to paragraphs for better readability
    const formattedText = text.replace(/\n/g, '<br>');
    
    msgDiv.innerHTML = `
        <div class="message-content">
            <p>${formattedText}</p>
        </div>
    `;
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function updateCitations(citations) {
    if (!citations || citations.length === 0) return;
    
    // If it's the first citation, clear the placeholder
    if (sourcesHistory.querySelector('.empty-sources')) {
        sourcesHistory.innerHTML = '';
    }

    citations.forEach(cit => {
        const citDiv = document.createElement('div');
        citDiv.className = 'source-item';
        
        citDiv.innerHTML = `
            <span class="source-title" title="${cit.source}">${cit.source}</span>
            <div class="source-meta">
                ${cit.page ? `<span>Page ${cit.page}</span> | ` : ''}
                <span class="score-badge">${cit.method} • ${Math.round(cit.score * 100)}% match</span>
            </div>
        `;
        // Prepend to show newest first or top sources
        sourcesHistory.prepend(citDiv);
    });
}

function appendLoadingIndicator() {
    const id = `typing_${Date.now()}`;
    const indicator = document.createElement('div');
    indicator.className = 'message assistant loading-msg';
    indicator.id = id;
    indicator.innerHTML = `
        <div class="message-content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    chatMessages.appendChild(indicator);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return id;
}

function removeLoadingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function setLoadingState(loading) {
    userInput.disabled = loading;
    sendBtn.disabled = loading;
    sendBtn.style.opacity = loading ? '0.5' : '1';
    sendBtn.style.cursor = loading ? 'wait' : 'pointer';
}

async function checkServerStatus() {
    try {
        const resp = await fetch(`${state.apiUrl}/health`);
        if (resp.ok) {
            serverStatus.classList.add('online');
            statusText.textContent = 'Server Online';
        }
    } catch (err) {
        console.warn('Backend server not reachable yet.');
        statusText.textContent = 'Backend Offline';
    }
}

function addWelcomeMessage() {
    chatMessages.innerHTML += `
        <div class="message assistant welcome">
            <div class="message-content">
                <h2>Welcome to VPA Chatbot!</h2>
                <p>I am your smart assistant for the <strong>Visakhapatnam Port Authority</strong>. I can answer questions based on official PDFs and the VPT website.</p>
                <ul class="suggestions">
                    <li><button onclick="sendQuery('What is the Master Plan of VPT?')">What is the Master Plan of VPT?</button></li>
                    <li><button onclick="sendQuery('Tell me about the cargo handled by VPA.')">Tell me about VPA cargo.</button></li>
                    <li><button onclick="sendQuery('What are the upcoming projects at Visakhapatnam Port?')">What are the upcoming projects?</button></li>
                </ul>
            </div>
        </div>
    `;
}

function showNotification(text, type = 'info') {
    // Simple alert or custom toast could be added here
    console.log(`[${type}] ${text}`);
}