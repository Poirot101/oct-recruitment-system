import express from 'express';
import { supabase } from '../db/supabase.js';
import { authenticateToken, authorizeRoles } from '../middleware/auth.js';

const router = express.Router();

// Get all profiles
router.get('/profiles', authenticateToken, async (req, res) => {
  try {
    const { role, userid } = req.user;

    let query = supabase
      .from('profile')
      .select('*')
      .order('profile_code');

    // Recruiters only see their own profiles
    if (role === 'recruiter') {
      query = query.eq('recruiter_email', userid);
    }

    const { data: profiles, error } = await query;

    if (error) {
      return res.status(500).json({ error: 'Failed to fetch profiles' });
    }

    res.json(profiles);
  } catch (error) {
    console.error('Get profiles error:', error);
    res.status(500).json({ error: 'Failed to fetch profiles' });
  }
});

// Create a new profile
router.post('/create_profile', authenticateToken, authorizeRoles('recruiter', 'admin'), async (req, res) => {
  try {
    const { company_name, designation, recruiter_email } = req.body;
    const { role, userid } = req.user;

    if (!company_name || !designation) {
      return res.status(400).json({ error: 'company_name and designation are required' });
    }

    // Determine recruiter email
    let finalRecruiterEmail = recruiter_email;
    
    if (role === 'recruiter') {
      // Recruiters can only create profiles for themselves
      finalRecruiterEmail = userid;
    } else if (role === 'admin') {
      // Admins must specify recruiter_email
      if (!recruiter_email) {
        return res.status(400).json({ error: 'recruiter_email is required for admin' });
      }
    }

    const { data: profile, error } = await supabase
      .from('profile')
      .insert([{
        recruiter_email: finalRecruiterEmail,
        company_name,
        designation
      }])
      .select()
      .single();

    if (error) {
      return res.status(500).json({ error: 'Failed to create profile' });
    }

    res.status(201).json(profile);
  } catch (error) {
    console.error('Create profile error:', error);
    res.status(500).json({ error: 'Failed to create profile' });
  }
});

export default router;
