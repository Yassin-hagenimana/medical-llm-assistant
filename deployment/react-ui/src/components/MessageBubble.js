import React from 'react';
import ReactMarkdown from 'react-markdown';
import { FaUser, FaStethoscope, FaCopy, FaCheck } from 'react-icons/fa';
import '../App.css';

const MessageBubble = ({ message, isUser, timestamp }) => {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(message);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`message-wrapper ${isUser ? 'user-message' : ''}`}>
      <div className={`message-avatar ${isUser ? 'user-avatar' : 'bot-avatar'}`}>
        {isUser ? <FaUser /> : <FaStethoscope />}
      </div>
      <div className="message-content">
        <div className={`message-bubble ${isUser ? 'user-bubble' : 'bot-bubble'}`}>
          {isUser ? (
            <p style={{ margin: 0 }}>{message}</p>
          ) : (
            <ReactMarkdown className="markdown-content">{message}</ReactMarkdown>
          )}
        </div>
        <span className="message-timestamp">{timestamp}</span>
        {!isUser && (
          <button 
            className="copy-button" 
            onClick={handleCopy}
            title="Copy response"
          >
            {copied ? <><FaCheck /> Copied!</> : <><FaCopy /> Copy</>}
          </button>
        )}
      </div>
    </div>
  );
};

export default MessageBubble;
