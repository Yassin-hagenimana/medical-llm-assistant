import React, { useState, useEffect, useRef } from 'react';
import MessageBubble from './MessageBubble';
import InputArea from './InputArea';
import Sidebar from './Sidebar';
import { sendQuery, checkHealth, getModelInfo } from '../services/api';
import { FaStethoscope } from 'react-icons/fa';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState('checking');
  const [modelInfo, setModelInfo] = useState(null);
  const [settings, setSettings] = useState({
    temperature: 0.7,
    maxLength: 200,
  });
  
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  // Scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Check API health on mount
  useEffect(() => {
    const initializeApp = async () => {
      const health = await checkHealth();
      setApiStatus(health.status === 'healthy' ? 'connected' : 'disconnected');
      
      const info = await getModelInfo();
      setModelInfo(info);

      // Add welcome message
      if (health.status === 'healthy') {
        setMessages([
          {
            id: 0,
            text: "Hello! I'm your Medical LLM Assistant powered by TinyLlama fine-tuned on medical data. I can help answer medical questions. How can I assist you today?",
            isUser: false,
            timestamp: new Date().toLocaleTimeString(),
          },
        ]);
      }
    };

    initializeApp();
  }, []);

  const handleSendMessage = async (text) => {
    if (!text.trim()) return;

    // Add user message
    const userMessage = {
      id: messages.length,
      text,
      isUser: true,
      timestamp: new Date().toLocaleTimeString(),
    };
    setMessages((prev) => [...prev, userMessage]);

    // Show loading
    setIsLoading(true);

    try {
      // Call API
      const response = await sendQuery(text, settings.temperature, settings.maxLength);
      
      // Add assistant response
      const assistantMessage = {
        id: messages.length + 1,
        text: response.response || response.answer || 'Sorry, I could not generate a response.',
        isUser: false,
        timestamp: new Date().toLocaleTimeString(),
        processingTime: response.processing_time,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      // Add error message
      const errorMessage = {
        id: messages.length + 1,
        text: `Error: ${error.message}`,
        isUser: false,
        timestamp: new Date().toLocaleTimeString(),
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleExampleClick = (question) => {
    handleSendMessage(question);
  };

  const handleClearHistory = () => {
    setMessages([
      {
        id: 0,
        text: "Conversation history cleared. How can I help you?",
        isUser: false,
        timestamp: new Date().toLocaleTimeString(),
      },
    ]);
  };

  const handleSettingsChange = (key, value) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
  };

  return (
    <div className="App">
      {/* Header */}
      <div className="chat-header">
        <div className="header-content">
          <div className="header-left">
            <div className="header-icon">
              <FaStethoscope />
            </div>
            <div className="header-title">
              <h1>MediCare Assistant</h1>
              <p>AI-Powered Medical Support</p>
            </div>
          </div>
          <div className="header-status">
            <span className={`status-indicator ${apiStatus}`}></span>
            <span>{apiStatus === 'connected' ? 'Connected' : 'Checking...'}</span>
          </div>
        </div>
      </div>

      {/* Main Layout */}
      <div className="main-layout">
        {/* Sidebar */}
        <Sidebar
          settings={settings}
          onSettingsChange={handleSettingsChange}
          onClearHistory={handleClearHistory}
          onExampleClick={handleExampleClick}
          modelInfo={modelInfo}
        />

        {/* Chat Area */}
        <div className="chat-container">
          <div className="messages-container scrollbar-light" ref={chatContainerRef}>
            {messages.length === 0 ? (
              <div className="animate-fade-in" style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100%',
                textAlign: 'center',
                padding: '2rem'
              }}>
                <div style={{
                  width: '4rem',
                  height: '4rem',
                  background: 'linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%)',
                  borderRadius: '1rem',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginBottom: '1rem',
                  boxShadow: '0 4px 12px rgba(59, 130, 246, 0.3)'
                }}>
                  <FaStethoscope style={{ fontSize: '2rem', color: 'white' }} />
                </div>
                <h2 style={{ fontSize: '1.5rem', fontWeight: '600', color: '#1f2937', marginBottom: '0.5rem' }}>
                  Welcome to MediCare Assistant
                </h2>
                <p style={{ color: '#6b7280', marginBottom: '1.5rem' }}>
                  Ask medical questions to get started
                </p>
              </div>
            ) : (
              messages.map((message) => (
                <MessageBubble
                  key={message.id}
                  message={message.text}
                  isUser={message.isUser}
                  timestamp={message.timestamp}
                />
              ))
            )}
            {isLoading && (
              <div className="typing-indicator">
                <div className="message-avatar bot-avatar">
                  <FaStethoscope />
                </div>
                <div className="typing-bubble">
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <InputArea
            onSendMessage={handleSendMessage}
            isLoading={isLoading}
            disabled={apiStatus === 'disconnected'}
          />
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
