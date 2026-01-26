import express from 'express';
import { supabase } from '../db/supabase.js';
import { authenticateToken, authorizeRoles } from '../middleware/auth.js';

const router = express.Router();

// Get applications
router.get('/applications', authenticateToken, async (req, res) => {
  try {
    const { role, userid } = req.user;

    let query = supabase
      .from('application')
      .select(`
        *,
        profile:profile_code (
          profile_code,
          company_name,
          designation,
          recruiter_email
        )
      `);

    if (role === 'student') {
      // Students see only their own applications
      query = query.eq('entry_number', userid);
    } else if (role === 'recruiter') {
      // Recruiters see applications for their profiles
      const { data: recruiterProfiles } = await supabase
        .from('profile')
        .select('profile_code')
        .eq('recruiter_email', userid);

      const profileCodes = recruiterProfiles.map(p => p.profile_code);
      query = query.in('profile_code', profileCodes);
    }

    const { data: applications, error } = await query;

    if (error) {
      return res.status(500).json({ error: 'Failed to fetch applications' });
    }

    res.json(applications);
  } catch (error) {
    console.error('Get applications error:', error);
    res.status(500).json({ error: 'Failed to fetch applications' });
  }
});

// Apply to a profile
router.post('/apply', authenticateToken, authorizeRoles('student'), async (req, res) => {
  try {
    const { profile_code } = req.body;
    const { userid } = req.user;

    if (!profile_code) {
      return res.status(400).json({ error: 'profile_code is required' });
    }

    // Check if student already has an accepted offer
    const { data: acceptedApps } = await supabase
      .from('application')
      .select('*')
      .eq('entry_number', userid)
      .eq('status', 'Accepted');

    if (acceptedApps && acceptedApps.length > 0) {
      return res.status(403).json({ error: 'You have already accepted an offer' });
    }

    // Check if already applied
    const { data: existingApp } = await supabase
      .from('application')
      .select('*')
      .eq('profile_code', profile_code)
      .eq('entry_number', userid)
      .single();

    if (existingApp) {
      return res.status(400).json({ error: 'Already applied to this profile' });
    }

    // Create application
    const { data: application, error } = await supabase
      .from('application')
      .insert([{
        profile_code,
        entry_number: userid,
        status: 'Applied'
      }])
      .select()
      .single();

    if (error) {
      return res.status(500).json({ error: 'Failed to create application' });
    }

    res.status(201).json(application);
  } catch (error) {
    console.error('Apply error:', error);
    res.status(500).json({ error: 'Failed to apply' });
  }
});

// Change application status
router.post('/application/change_status', authenticateToken, authorizeRoles('recruiter', 'admin'), async (req, res) => {
  try {
    const { profile_code, entry_number, status } = req.body;
    const { role, userid } = req.user;

    if (!profile_code || !entry_number || !status) {
      return res.status(400).json({ error: 'profile_code, entry_number, and status are required' });
    }

    const validStatuses = ['Applied', 'Not Selected', 'Selected', 'Accepted'];
    if (!validStatuses.includes(status)) {
      return res.status(400).json({ error: 'Invalid status' });
    }

    // If recruiter, verify they own the profile
    if (role === 'recruiter') {
      const { data: profile } = await supabase
        .from('profile')
        .select('recruiter_email')
        .eq('profile_code', profile_code)
        .single();

      if (!profile || profile.recruiter_email !== userid) {
        return res.status(403).json({ error: 'You can only modify applications for your profiles' });
      }
    }

    // Update application status
    const { data: application, error } = await supabase
      .from('application')
      .update({ status })
      .eq('profile_code', profile_code)
      .eq('entry_number', entry_number)
      .select()
      .single();

    if (error) {
      return res.status(500).json({ error: 'Failed to update application status' });
    }

    res.json(application);
  } catch (error) {
    console.error('Change status error:', error);
    res.status(500).json({ error: 'Failed to change status' });
  }
});

// Accept a selected application
router.post('/application/accept', authenticateToken, authorizeRoles('student'), async (req, res) => {
  try {
    const { profile_code } = req.body;
    const { userid } = req.user;

    if (!profile_code) {
      return res.status(400).json({ error: 'profile_code is required' });
    }

    // Verify application exists and is in Selected status
    const { data: application } = await supabase
      .from('application')
      .select('*')
      .eq('profile_code', profile_code)
      .eq('entry_number', userid)
      .single();

    if (!application) {
      return res.status(404).json({ error: 'Application not found' });
    }

    if (application.status !== 'Selected') {
      return res.status(400).json({ error: 'Can only accept applications with Selected status' });
    }

    // Update to Accepted
    const { data: updatedApp, error } = await supabase
      .from('application')
      .update({ status: 'Accepted' })
      .eq('profile_code', profile_code)
      .eq('entry_number', userid)
      .select()
      .single();

    if (error) {
      return res.status(500).json({ error: 'Failed to accept application' });
    }

    res.json(updatedApp);
  } catch (error) {
    console.error('Accept application error:', error);
    res.status(500).json({ error: 'Failed to accept application' });
  }
});

// Reject a selected application
router.post('/application/reject', authenticateToken, authorizeRoles('student'), async (req, res) => {
  try {
    const { profile_code } = req.body;
    const { userid } = req.user;

    if (!profile_code) {
      return res.status(400).json({ error: 'profile_code is required' });
    }

    // Update to Not Selected
    const { data: updatedApp, error } = await supabase
      .from('application')
      .update({ status: 'Not Selected' })
      .eq('profile_code', profile_code)
      .eq('entry_number', userid)
      .select()
      .single();

    if (error) {
      return res.status(500).json({ error: 'Failed to reject application' });
    }

    res.json(updatedApp);
  } catch (error) {
    console.error('Reject application error:', error);
    res.status(500).json({ error: 'Failed to reject application' });
  }
});

export default router;
