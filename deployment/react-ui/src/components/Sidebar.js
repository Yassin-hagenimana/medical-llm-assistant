import React from 'react';
import { FaCog, FaTrash, FaLightbulb, FaExclamationTriangle, FaTint, FaHeartbeat, FaPills, FaBrain, FaStethoscope, FaHeart } from 'react-icons/fa';

const Sidebar = ({ 
  settings, 
  onSettingsChange, 
  onClearHistory,
  onExampleClick,
  modelInfo 
}) => {
  const exampleQuestions = [
    { text: "What are the symptoms of diabetes?", icon: FaTint },
    { text: "How is hypertension treated?", icon: FaHeartbeat },
    { text: "What are the side effects of aspirin?", icon: FaPills },
    { text: "What causes migraine headaches?", icon: FaBrain },
    { text: "How does insulin work in the body?", icon: FaTint },
    { text: "What is the difference between Type 1 and Type 2 diabetes?", icon: FaStethoscope },
    { text: "What are the risk factors for heart disease?", icon: FaHeart },
    { text: "How is asthma diagnosed?", icon: FaHeartbeat },
  ];

  return (
    <div className="sidebar">
      {/* Settings */}
      <div className="sidebar-section">
        <h3>
          <FaCog /> Generation Settings
        </h3>
        
        <div className="setting-item">
          <label>
            Temperature: {settings.temperature.toFixed(2)}
            <span className="setting-hint">
              Higher = more creative, Lower = more focused
            </span>
          </label>
          <input
            type="range"
            min="0.1"
            max="1.0"
            step="0.1"
            value={settings.temperature}
            onChange={(e) => onSettingsChange('temperature', parseFloat(e.target.value))}
          />
        </div>

        <div className="setting-item">
          <label>
            Max Length: {settings.maxLength} tokens
            <span className="setting-hint">
              Maximum response length
            </span>
          </label>
          <input
            type="range"
            min="50"
            max="500"
            step="50"
            value={settings.maxLength}
            onChange={(e) => onSettingsChange('maxLength', parseInt(e.target.value))}
          />
        </div>
      </div>

      {/* Example Questions */}
      <div className="sidebar-section">
        <h3>
          <FaLightbulb /> Example Questions
        </h3>
        <div className="example-questions">
          {exampleQuestions.map((item, index) => {
            const IconComponent = item.icon;
            return (
              <button
                key={index}
                className="example-button"
                onClick={() => onExampleClick(item.text)}
              >
                <IconComponent />
                <span>{item.text}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Clear Button */}
      <button className="clear-button" onClick={onClearHistory}>
        <FaTrash />
        Clear Conversation History
      </button>

      {/* How to Use */}
      <div className="sidebar-section info-section">
        <h4>How to Use</h4>
        <ol>
          <li>
            <span>1.</span>
            <span>Type your medical question in the input box</span>
          </li>
          <li>
            <span>2.</span>
            <span>Click Send or press Enter to submit</span>
          </li>
          <li>
            <span>3.</span>
            <span>Wait for the AI assistant to generate a response</span>
          </li>
          <li>
            <span>4.</span>
            <span>Adjust temperature for different response styles</span>
          </li>
        </ol>
      </div>

      {/* Disclaimer */}
      <div className="sidebar-section disclaimer-section">
        <div className="disclaimer-content">
          <FaExclamationTriangle />
          <div>
            <h4>Disclaimer</h4>
            <p>
              This is an AI assistant for educational purposes. Always consult qualified healthcare professionals for medical advice.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
