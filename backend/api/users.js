import express from 'express';
import { supabase } from '../db/supabase.js';
import { authenticateToken, authorizeRoles } from '../middleware/auth.js';

const router = express.Router();

// Get current user's details
router.get('/users/me', authenticateToken, async (req, res) => {
  try {
    const { userid, role } = req.user;

    const { data: user, error } = await supabase
      .from('users')
      .select('userid, role')
      .eq('userid', userid)
      .single();

    if (error) {
      return res.status(404).json({ error: 'User not found' });
    }

    res.json(user);
  } catch (error) {
    console.error('Get user error:', error);
    res.status(500).json({ error: 'Failed to fetch user' });
  }
});

// Get all users (admin only)
router.get('/users', authenticateToken, authorizeRoles('admin'), async (req, res) => {
  try {
    const { data: users, error } = await supabase
      .from('users')
      .select('userid, role')
      .order('userid');

    if (error) {
      return res.status(500).json({ error: 'Failed to fetch users' });
    }

    res.json(users);
  } catch (error) {
    console.error('Get users error:', error);
    res.status(500).json({ error: 'Failed to fetch users' });
  }
});

export default router;