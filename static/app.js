document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM fully loaded and parsed');
    document.getElementById('user-input').addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // 페이지 로드 시 새로운 채팅 시작
    startNewChat();
});

let currentChatId = null;

async function startNewChat() {
    try {
        const response = await fetch('/start_chat', { method: 'POST' });
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const newChat = await response.json();
        currentChatId = newChat.id;
        loadChat(newChat.id);
    } catch (error) {
        console.error('Failed to start new chat:', error);
        alert('로그인이 필요합니다.');
    }
}

async function loadChat(chatId) {
    currentChatId = chatId;
    document.getElementById('chat-output').innerHTML = '';
    const response = await fetch(`/get_chat/${chatId}`);
    const chat = await response.json();
    displayChat(chat);
}

function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    console.log(`User input: ${userInput}`);
    if (userInput.trim() === '') return;

    const chatOutput = document.getElementById('chat-output');
    if (!chatOutput) {
        console.error('Cannot find chat-output element');
        return;
    }

    // 유저 메시지 추가
    const userMessage = document.createElement('div');
    userMessage.classList.add('message', 'user');
    userMessage.textContent = userInput;
    chatOutput.appendChild(userMessage);

    // 챗봇 메시지 가져오기
    fetch('/send_message', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ chat_id: currentChatId, message: userInput }) // chat_id는 현재 채팅 ID
    })
    .then(response => {
        console.log('Response status:', response.status);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        console.log('Response data:', data);
        if (data.error) {
            console.error(data.error);
            return;
        }
        const botMessage = document.createElement('div');
        botMessage.classList.add('message', 'bot');
        botMessage.textContent = data.messages[data.messages.length - 1].text; // 마지막 메시지 가져오기
        chatOutput.appendChild(botMessage);

        // 스크롤을 맨 아래로
        chatOutput.scrollTop = chatOutput.scrollHeight;
    })
    .catch(error => {
        console.error('There has been a problem with your fetch operation:', error);
    });

    document.getElementById('user-input').value = '';
}

function displayChat(chat) {
    const chatContainer = document.getElementById('chat-output');
    chatContainer.innerHTML = '';
    chat.messages.forEach(msg => {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        if (msg.sender === 'bot') {
            messageDiv.classList.add('bot');
        }
        messageDiv.textContent = `${msg.sender}: ${msg.text}`;
        chatContainer.appendChild(messageDiv);
    });
}
