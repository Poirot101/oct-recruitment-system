import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Health check route
app.get('/', (req, res) => {
  res.json({ message: 'OCS Recruitment API is running' });
});

// Import routes after app is created
import authRoutes from './api/auth.js';
import userRoutes from './api/users.js';
import profileRoutes from './api/profiles.js';
import applicationRoutes from './api/applications.js';

// Use routes
app.use('/api', authRoutes);
app.use('/api', userRoutes);
app.use('/api', profileRoutes);
app.use('/api', applicationRoutes);

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Error:', err.stack);
  res.status(500).json({ error: 'Something went wrong!', message: err.message });
});

// Start server
app.listen(PORT, () => {
  console.log(`✅ Server running on port ${PORT}`);
  console.log(`🌐 API URL: http://localhost:${PORT}`);
});