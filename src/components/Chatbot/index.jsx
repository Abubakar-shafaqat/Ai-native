import React, { useState, useCallback, useRef, useEffect } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './styles.module.css';

const Chatbot = () => {
  const {siteConfig} = useDocusaurusContext();
  const RAG_API_URL = siteConfig.customFields.ragApiUrl || 'http://localhost:8000/chat';
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = useCallback(async () => {
    if (input.trim() === '') return;

    const userMessage = { text: input, sender: 'user' };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setInput('');
    setIsLoading(true);

    const selectedText = window.getSelection().toString(); // Get currently selected text

    try {
      const response = await fetch(RAG_API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: input, selected_text: selectedText || null }), // Send selected_text
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const botMessage = { text: data.answer, sender: 'bot' };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
      console.error('Error sending message to RAG API:', error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: 'Sorry, I could not get a response. Please try again later.', sender: 'bot' },
      ]);
    } finally {
      setIsLoading(false);
    }
  }, [input, RAG_API_URL]);

  const handleKeyPress = useCallback(
    (e) => {
      if (e.key === 'Enter') {
        handleSendMessage();
      }
    },
    [handleSendMessage]
  );

  return (
    <div className={styles.chatbotContainer}>
      <button className={styles.chatbotToggle} onClick={() => setIsOpen(!isOpen)}>
        {isOpen ? '✕' : '💬'}
      </button>

      {isOpen && (
        <div className={styles.chatWindow}>
          <div className={styles.chatHeader}>
            <h3>Book Assistant</h3>
          </div>
          <div className={styles.chatMessages}>
            {messages.length === 0 && !isLoading && (
              <div className={styles.welcomeMessage}>
                Hi there! Ask me anything about the "Physical AI & Humanoid Robotics" textbook.
              </div>
            )}
            {messages.map((msg, index) => (
              <div key={index} className={`${styles.chatMessage} ${styles[msg.sender]}`}>
                {msg.text.split('\n').map((line, i) => (
                    <p key={i}>{line}</p>
                ))}
              </div>
            ))}
            {isLoading && (
              <div className={`${styles.chatMessage} ${styles.bot}`}>
                <div className={styles.loadingDots}>
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          <div className={styles.chatInputContainer}>
            <input
              type="text"
              className={styles.chatInput}
              placeholder="Ask a question..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={isLoading}
            />
            <button className={styles.sendButton} onClick={handleSendMessage} disabled={isLoading}>
              Send
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Chatbot;
