import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';


dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;


const allowedOrigins = [
  'https://backend.ocs-backend-6dl9gddcb-aadi-bansals-projects.vercel.app',
  'https://ocs-frontend-ra70e1hp3-aadi-bansals-projects.vercel.app'
];

// Allow localhost origins when testing locally — enable with CORS_ALLOW_LOCALHOST=true
if (process.env.CORS_ALLOW_LOCALHOST === 'true') {
  const localhostOrigins = [
    'http://localhost:3000',
    'http://127.0.0.1:3000',
    'http://localhost:5173',
    'http://127.0.0.1:5173'
  ];
  allowedOrigins.push(...localhostOrigins);
}

app.use(cors({
  origin: (origin, callback) => {
    if (!origin) return callback(null, true);
    if (allowedOrigins.includes(origin)) return callback(null, true);
    return callback(new Error('CORS origin not allowed'));
  },
  credentials: true,
}));

app.use(express.json());


app.get('/', (req, res) => {
  res.json({ message: 'OCS Recruitment API is running' });
});


import authRoutes from './api/auth.js';
import userRoutes from './api/users.js';
import profileRoutes from './api/profiles.js';
import applicationRoutes from './api/applications.js';


app.use('/api', authRoutes);
app.use('/api', userRoutes);
app.use('/api', profileRoutes);
app.use('/api', applicationRoutes);


app.use((req, res) => {
  res.status(404).json({ error: 'Route not found' });
});


app.use((err, req, res, next) => {
  console.error('Error:', err.stack);
  res.status(500).json({ error: 'Something went wrong!', message: err.message });
});

// Export for Vercel
export default app;