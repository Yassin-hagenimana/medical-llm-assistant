import axios from 'axios';

const API_BASE_URL = 'http://0.0.0.0:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 120000, // 120 seconds (2 minutes) for CPU inference
});

/**
 * Send a medical question to the API
 * @param {string} query - The medical question
 * @param {number} temperature - Sampling temperature (0.1-1.0)
 * @param {number} maxLength - Maximum response length
 * @returns {Promise} API response
 */
export const sendQuery = async (query, temperature = 0.7, maxLength = 50) => {
  try {
    const response = await api.post('/query', {
      query,
      temperature,
      max_tokens: maxLength,  // Reduced default from 200 to 150 for faster responses
    });
    return response.data;
  } catch (error) {
    console.error('API Error:', error);
    throw new Error(
      error.response?.data?.detail || 
      'Failed to get response from the server. Please ensure the FastAPI backend is running on port 8000.'
    );
  }
};

/**
 * Send multiple queries in batch
 * @param {Array<string>} queries - Array of medical questions
 * @returns {Promise} API response
 */
export const sendBatchQuery = async (queries) => {
  try {
    const response = await api.post('/batch_query', { queries });
    return response.data;
  } catch (error) {
    console.error('Batch API Error:', error);
    throw new Error(
      error.response?.data?.detail || 
      'Failed to process batch queries'
    );
  }
};

/**
 * Check API health status
 * @returns {Promise} Health check response
 */
export const checkHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('Health Check Error:', error);
    return { status: 'error', message: 'API unavailable' };
  }
};

/**
 * Get model information
 * @returns {Promise} Model info response
 */
export const getModelInfo = async () => {
  try {
    const response = await api.get('/info');
    return response.data;
  } catch (error) {
    console.error('Model Info Error:', error);
    return null;
  }
};

export default api;
