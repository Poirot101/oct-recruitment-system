import express from 'express';
import jwt from 'jsonwebtoken';
import { supabase } from '../db/supabase.js';

const router = express.Router();

// Login endpoint
router.post('/login', async (req, res) => {
  try {
    const { userid, password_md5 } = req.body;

    if (!userid || !password_md5) {
      return res.status(400).json({ error: 'userid and password_md5 are required' });
    }

    // Query user from database
    const { data: user, error } = await supabase
      .from('users')
      .select('userid, password_hash, role')
      .eq('userid', userid)
      .single();

    if (error || !user) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    // Verify password hash
    if (user.password_hash !== password_md5) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    // Generate JWT token
    const token = jwt.sign(
      { userid: user.userid, role: user.role },
      process.env.JWT_SECRET,
      { expiresIn: '24h' }
    );

    res.json({
      token,
      role: user.role,
      userid: user.userid
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: 'Login failed' });
  }
});

export default router;