import React, { useState, useEffect, useRef } from 'react';
import { Heart, Pill, Brain, Droplet, Activity, Stethoscope, AlertCircle, Send, Settings, Trash2, Info, ChevronRight } from 'lucide-react';

export default function MedicalChatbot() {
  const [messages, setMessages] = useState([
    {
      type: 'bot',
      text: "Hello! I'm your medical AI assistant. I can help answer questions about health, medical conditions, and general wellness. How can I assist you today?",
      time: new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })
    }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [maxLength, setMaxLength] = useState(200);
  const [showSettings, setShowSettings] = useState(false);
  const messagesEndRef = useRef(null);

  const responses = {
    "symptoms of diabetes": "Common symptoms of diabetes include:\n\n• Increased thirst and frequent urination\n• Extreme hunger and unexplained weight loss\n• Fatigue and weakness\n• Blurred vision\n• Slow-healing sores\n• Frequent infections\n\nIf you experience these symptoms, please consult a healthcare provider.",
    
    "side effects of aspirin": "Common side effects of aspirin:\n\n• Upset stomach or heartburn\n• Nausea\n• Easy bruising\n• Prolonged bleeding\n• Ringing in ears\n\nSerious side effects include severe stomach pain or bleeding. Always consult your doctor.",
    
    "hypertension treated": "Hypertension treatment includes:\n\n• Lifestyle modifications (diet, exercise)\n• Medications (ACE inhibitors, beta blockers)\n• Reducing sodium intake\n• Weight management\n• Stress reduction\n• Regular monitoring\n\nYour doctor will create a personalized treatment plan.",
    
    "migraine headaches": "Migraine triggers can include:\n\n• Hormonal changes\n• Certain foods and drinks\n• Stress and anxiety\n• Bright lights or loud sounds\n• Sleep pattern changes\n• Weather changes\n\nKeeping a headache diary helps identify personal triggers.",
    
    "how does insulin work": "Insulin is a hormone that:\n\n• Allows cells to absorb glucose from blood\n• Converts glucose into energy\n• Regulates blood sugar levels\n• Stores excess glucose in the liver\n\nIt acts like a key, unlocking cells to let glucose in for energy.",
    
    "type 1 type 2 diabetes": "Type 1 Diabetes:\n• Autoimmune condition\n• Little to no insulin production\n• Requires insulin therapy\n• Usually diagnosed in youth\n\nType 2 Diabetes:\n• Insulin resistance\n• Often develops in adults\n• Managed with lifestyle changes\n• May require medication",
    
    "heart disease risk": "Risk factors for heart disease:\n\nModifiable:\n• High blood pressure\n• High cholesterol\n• Smoking\n• Obesity\n• Physical inactivity\n\nNon-modifiable:\n• Age\n• Family history\n• Gender",
    
    "asthma diagnosed": "Asthma diagnosis involves:\n\n• Medical history review\n• Physical examination\n• Lung function tests (spirometry)\n• Peak flow measurement\n• Allergy testing\n• Chest X-ray\n\nProper diagnosis ensures effective treatment."
  };

  const exampleQuestions = [
    { text: "What are the symptoms of diabetes?", icon: Droplet },
    { text: "How is hypertension treated?", icon: Activity },
    { text: "What are the side effects of aspirin?", icon: Pill },
    { text: "What causes migraine headaches?", icon: Brain },
    { text: "How does insulin work in the body?", icon: Droplet },
    { text: "What is the difference between Type 1 and Type 2 diabetes?", icon: Stethoscope },
    { text: "What are the risk factors for heart disease?", icon: Heart },
    { text: "How is asthma diagnosed?", icon: Activity }
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const getAIResponse = (question) => {
    const lowerQuestion = question.toLowerCase();
    
    for (const [key, response] of Object.entries(responses)) {
      if (lowerQuestion.includes(key)) {
        return response;
      }
    }
    
    return `Thank you for asking: "${question}"\n\nI'm here to provide general medical information. For personalized advice, please consult a healthcare professional.\n\nTry asking about:\n• Common symptoms\n• Medication information\n• Health conditions\n• Wellness tips`;
  };

  const sendMessage = (text = input) => {
    if (!text.trim()) return;

    const newMessage = {
      type: 'user',
      text: text,
      time: new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })
    };

    setMessages(prev => [...prev, newMessage]);
    setInput('');
    setIsTyping(true);

    setTimeout(() => {
      const response = {
        type: 'bot',
        text: getAIResponse(text),
        time: new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })
      };
      setMessages(prev => [...prev, response]);
      setIsTyping(false);
    }, 1200 + Math.random() * 800);
  };

  const clearHistory = () => {
    setMessages([
      {
        type: 'bot',
        text: "Conversation cleared. How can I help you today?",
        time: new Date().toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })
      }
    ]);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-4 md:p-8">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {
          font-family: 'Inter', sans-serif;
        }
        
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes slideUp {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes bounce {
          0%, 80%, 100% { transform: translateY(0); }
          40% { transform: translateY(-6px); }
        }
        
        .animate-fade-in {
          animation: fadeIn 0.5s ease-out;
        }
        
        .animate-slide-up {
          animation: slideUp 0.6s ease-out;
        }
        
        .typing-dot {
          animation: bounce 1.4s infinite;
        }
        
        .typing-dot:nth-child(2) {
          animation-delay: 0.2s;
        }
        
        .typing-dot:nth-child(3) {
          animation-delay: 0.4s;
        }
        
        .glass-card {
          background: rgba(255, 255, 255, 0.7);
          backdrop-filter: blur(16px);
          border: 1px solid rgba(255, 255, 255, 0.9);
          box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06);
        }
        
        .glass-card-dark {
          background: rgba(255, 255, 255, 0.85);
          backdrop-filter: blur(20px);
          border: 1px solid rgba(255, 255, 255, 0.95);
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
        }
        
        .scrollbar-light::-webkit-scrollbar {
          width: 6px;
        }
        
        .scrollbar-light::-webkit-scrollbar-track {
          background: rgba(0, 0, 0, 0.02);
          border-radius: 10px;
        }
        
        .scrollbar-light::-webkit-scrollbar-thumb {
          background: rgba(99, 102, 241, 0.3);
          border-radius: 10px;
        }
        
        .scrollbar-light::-webkit-scrollbar-thumb:hover {
          background: rgba(99, 102, 241, 0.5);
        }

        input[type="range"] {
          -webkit-appearance: none;
          appearance: none;
          width: 100%;
          height: 6px;
          border-radius: 5px;
          background: #e5e7eb;
          outline: none;
        }

        input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 16px;
          height: 16px;
          border-radius: 50%;
          background: #6366f1;
          cursor: pointer;
        }

        input[type="range"]::-moz-range-thumb {
          width: 16px;
          height: 16px;
          border-radius: 50%;
          background: #6366f1;
          cursor: pointer;
          border: none;
        }
      `}</style>

      <div className="max-w-7xl mx-auto animate-slide-up">
        {/* Header */}
        <div className="glass-card rounded-3xl p-6 mb-6">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-4">
              <div className="w-14 h-14 bg-gradient-to-br from-blue-500 to-purple-500 rounded-2xl flex items-center justify-center shadow-lg">
                <Stethoscope className="w-7 h-7 text-white" />
              </div>
              <div>
                <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  MediCare Assistant
                </h1>
                <p className="text-gray-500 text-sm">AI-Powered Medical Support</p>
              </div>
            </div>
            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-3 hover:bg-gray-100 rounded-xl transition-colors"
            >
              <Settings className="w-5 h-5 text-gray-600" />
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar */}
          <div className="lg:col-span-1 space-y-4">
            {/* Settings Panel */}
            {showSettings && (
              <div className="glass-card rounded-2xl p-5 animate-fade-in">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-semibold text-gray-800 uppercase tracking-wide">Generation Settings</h3>
                  <button onClick={() => setShowSettings(false)}>
                    <ChevronRight className="w-4 h-4 text-gray-400" />
                  </button>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <label className="text-sm font-medium text-gray-700">Temperature: {temperature.toFixed(2)}</label>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      value={temperature}
                      onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    />
                    <p className="text-xs text-gray-500 mt-1">Higher = more creative, Lower = more focused</p>
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <label className="text-sm font-medium text-gray-700">Max Length: {maxLength} tokens</label>
                    </div>
                    <input
                      type="range"
                      min="50"
                      max="500"
                      step="10"
                      value={maxLength}
                      onChange={(e) => setMaxLength(parseInt(e.target.value))}
                    />
                    <p className="text-xs text-gray-500 mt-1">Maximum response length</p>
                  </div>
                </div>
              </div>
            )}

            {/* Example Questions */}
            <div className="glass-card rounded-2xl p-5">
              <h3 className="text-sm font-semibold text-gray-800 mb-4 uppercase tracking-wide flex items-center gap-2">
                <Info className="w-4 h-4" />
                Example Questions
              </h3>
              <div className="space-y-2">
                {exampleQuestions.map((item, idx) => {
                  const IconComponent = item.icon;
                  return (
                    <button
                      key={idx}
                      onClick={() => sendMessage(item.text)}
                      className="w-full text-left p-3 rounded-xl bg-white/60 hover:bg-white border border-gray-200 hover:border-blue-300 text-gray-700 hover:text-blue-600 text-sm transition-all duration-200 hover:shadow-md flex items-center gap-3 group"
                    >
                      <IconComponent className="w-4 h-4 text-gray-400 group-hover:text-blue-500 flex-shrink-0" />
                      <span className="line-clamp-2">{item.text}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Clear Button */}
            <button
              onClick={clearHistory}
              className="w-full glass-card rounded-xl p-3 text-red-600 hover:bg-red-50 border border-red-200 transition-all duration-200 flex items-center justify-center gap-2 text-sm font-medium"
            >
              <Trash2 className="w-4 h-4" />
              Clear Conversation History
            </button>

            {/* How to Use */}
            <div className="glass-card rounded-2xl p-5 bg-blue-50/50 border-blue-200">
              <h4 className="text-sm font-semibold text-blue-700 mb-3 uppercase tracking-wide">How to Use</h4>
              <ol className="space-y-2 text-xs text-gray-600">
                <li className="flex items-start gap-2">
                  <span className="font-semibold text-blue-600 flex-shrink-0">1.</span>
                  <span>Type your medical question in the input box</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="font-semibold text-blue-600 flex-shrink-0">2.</span>
                  <span>Click Send or press Enter to submit</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="font-semibold text-blue-600 flex-shrink-0">3.</span>
                  <span>Wait for the AI assistant to generate a response</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="font-semibold text-blue-600 flex-shrink-0">4.</span>
                  <span>Adjust temperature for different response styles</span>
                </li>
              </ol>
            </div>

            {/* Disclaimer */}
            <div className="glass-card rounded-2xl p-5 border-orange-200 bg-orange-50/50">
              <div className="flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-orange-600 flex-shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-orange-700 font-semibold text-sm mb-1">Disclaimer</h4>
                  <p className="text-gray-600 text-xs leading-relaxed">
                    This is an AI assistant for educational purposes. Always consult qualified healthcare professionals for medical advice.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Chat Area */}
          <div className="lg:col-span-3">
            <div className="glass-card-dark rounded-3xl flex flex-col h-[700px] overflow-hidden">
              {/* Messages */}
              <div className="flex-1 overflow-y-auto p-6 space-y-4 scrollbar-light">
                {messages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`flex gap-3 animate-fade-in ${msg.type === 'user' ? 'flex-row-reverse' : ''}`}
                  >
                    <div className={`w-10 h-10 rounded-2xl flex items-center justify-center flex-shrink-0 shadow-md ${
                      msg.type === 'user' 
                        ? 'bg-gradient-to-br from-purple-400 to-pink-400' 
                        : 'bg-gradient-to-br from-blue-400 to-indigo-400'
                    }`}>
                      {msg.type === 'user' ? (
                        <Heart className="w-5 h-5 text-white" />
                      ) : (
                        <Stethoscope className="w-5 h-5 text-white" />
                      )}
                    </div>
                    <div className={`max-w-[80%] ${msg.type === 'user' ? 'items-end' : 'items-start'} flex flex-col gap-1`}>
                      <div className={`p-4 rounded-2xl ${
                        msg.type === 'user'
                          ? 'bg-gradient-to-br from-purple-500 to-pink-500 text-white shadow-lg shadow-purple-200'
                          : 'bg-white/80 text-gray-800 border border-gray-200 shadow-md'
                      }`}>
                        <p className="whitespace-pre-line text-sm leading-relaxed">{msg.text}</p>
                      </div>
                      <span className="text-xs text-gray-400 px-2">{msg.time}</span>
                    </div>
                  </div>
                ))}
                
                {isTyping && (
                  <div className="flex gap-3 animate-fade-in">
                    <div className="w-10 h-10 bg-gradient-to-br from-blue-400 to-indigo-400 rounded-2xl flex items-center justify-center shadow-md">
                      <Stethoscope className="w-5 h-5 text-white" />
                    </div>
                    <div className="bg-white/80 border border-gray-200 p-4 rounded-2xl shadow-md flex gap-2">
                      <div className="w-2 h-2 bg-gray-400 rounded-full typing-dot"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full typing-dot"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full typing-dot"></div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              <div className="border-t border-gray-200 bg-white/60 p-5">
                <div className="flex gap-3 items-center bg-white rounded-2xl p-3 border border-gray-200 focus-within:border-blue-400 focus-within:shadow-lg transition-all">
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Type your medical question..."
                    className="flex-1 bg-transparent border-none outline-none text-gray-800 placeholder-gray-400 px-2"
                  />
                  <button
                    onClick={() => sendMessage()}
                    disabled={!input.trim()}
                    className="w-11 h-11 bg-gradient-to-br from-blue-500 to-purple-500 rounded-xl flex items-center justify-center hover:shadow-lg transition-all disabled:opacity-40 disabled:cursor-not-allowed active:scale-95"
                  >
                    <Send className="w-5 h-5 text-white" />
                  </button>
                </div>
                <p className="text-xs text-gray-400 mt-2 text-center">Press Enter to send • Shift+Enter for new line</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
