import React, { useState, useRef, useEffect } from 'react';
import { FaPaperPlane } from 'react-icons/fa';

const InputArea = ({ onSendMessage, isLoading, disabled }) => {
  const [input, setInput] = useState('');
  const textareaRef = useRef(null);

  useEffect(() => {
    // Auto-resize textarea
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [input]);

  const handleSend = () => {
    if (input.trim() && !disabled && !isLoading) {
      onSendMessage(input);
      setInput('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="input-area">
      <div className="input-wrapper">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your medical question..."
          className="message-input"
          disabled={disabled || isLoading}
          rows={1}
        />
        <button
          onClick={handleSend}
          disabled={!input.trim() || disabled || isLoading}
          className="send-button"
        >
          <FaPaperPlane />
        </button>
      </div>
      <p className="input-hint">
        Press Enter to send • Shift+Enter for new line
      </p>
    </div>
  );
};

export default InputArea;
