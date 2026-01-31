const API_URL = import.meta.env.VITE_API_URL || '';

// Check authentication
const token = localStorage.getItem('token');
const role = localStorage.getItem('role');
const userid = localStorage.getItem('userid');

if (!token || role !== 'student') {
    window.location.href = '/';
}

// Global logout function
window.logout = function() {
    localStorage.clear();
    window.location.href = '/';
};

// Set user name
document.getElementById('user-name').textContent = userid;

let hasAcceptedOffer = false;
let selectedApplications = [];

// API Helper
async function apiCall(endpoint, options = {}) {
    const response = await fetch(`${API_URL}${endpoint}`, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`,
            ...options.headers
        }
    });
    
    const data = await response.json();
    
    if (!response.ok) {
        throw new Error(data.error || 'Request failed');
    }
    
    return data;
}

// Show message
function showMessage(message, type = 'error') {
    const messageArea = document.getElementById('message-area');
    messageArea.innerHTML = `<div class="${type}-message">${message}</div>`;
    setTimeout(() => {
        messageArea.innerHTML = '';
    }, 5000);
}

// Load applications
async function loadApplications() {
    try {
        const applications = await apiCall('/api/applications');
        
        // Check for accepted and selected applications
        const acceptedApps = applications.filter(app => app.status === 'Accepted');
        selectedApplications = applications.filter(app => app.status === 'Selected');
        
        if (acceptedApps.length > 0) {
            hasAcceptedOffer = true;
            showAcceptedOffer(acceptedApps[0]);
        } else if (selectedApplications.length > 0) {
            showSelectedOffers(selectedApplications);
        } else {
            document.getElementById('profiles-section').style.display = 'block';
            loadProfiles();
        }
        
        // Always show applications table
        displayApplications(applications);
    } catch (error) {
        showMessage(error.message);
    }
}

// Display applications
function displayApplications(applications) {
    const tbody = document.getElementById('applications-body');
    
    if (applications.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" style="text-align: center;">No applications yet</td></tr>';
        return;
    }
    
    tbody.innerHTML = applications.map(app => `
        <tr>
            <td>${app.profile_code}</td>
            <td>${app.profile.company_name}</td>
            <td>${app.profile.designation}</td>
            <td><span class="status-badge status-${app.status.toLowerCase().replace(' ', '-')}">${app.status}</span></td>
        </tr>
    `).join('');
}

// Show accepted offer
function showAcceptedOffer(application) {
    document.getElementById('selected-section').style.display = 'none';
    document.getElementById('profiles-section').style.display = 'none';
    document.getElementById('accepted-section').style.display = 'block';
    
    document.getElementById('accepted-message').textContent = 
        `You have accepted an offer from ${application.profile.company_name} (${application.profile.designation})`;
}

// Show selected offers
function showSelectedOffers(applications) {
    document.getElementById('profiles-section').style.display = 'none';
    document.getElementById('accepted-section').style.display = 'none';
    document.getElementById('selected-section').style.display = 'block';
    
    const container = document.getElementById('selected-offers');
    container.innerHTML = applications.map(app => `
        <div style="background: #fef5e7; padding: 20px; border-radius: 8px; margin-bottom: 15px;">
            <h3>${app.profile.company_name} - ${app.profile.designation}</h3>
            <p>Profile Code: ${app.profile_code}</p>
            <div style="margin-top: 15px; display: flex; gap: 10px;">
                <button onclick="acceptOffer(${app.profile_code})" class="btn btn-success">Accept</button>
                <button onclick="rejectOffer(${app.profile_code})" class="btn btn-danger">Reject</button>
            </div>
        </div>
    `).join('');
}

// Accept offer
window.acceptOffer = async function(profile_code) {
    if (!confirm('Are you sure you want to accept this offer? This action cannot be undone.')) {
        return;
    }
    
    try {
        await apiCall('/api/application/accept', {
            method: 'POST',
            body: JSON.stringify({ profile_code })
        });
        
        showMessage('Offer accepted successfully!', 'success');
        loadApplications();
    } catch (error) {
        showMessage(error.message);
    }
};

// Reject offer
window.rejectOffer = async function(profile_code) {
    if (!confirm('Are you sure you want to reject this offer?')) {
        return;
    }
    
    try {
        await apiCall('/api/application/reject', {
            method: 'POST',
            body: JSON.stringify({ profile_code })
        });
        
        showMessage('Offer rejected', 'success');
        loadApplications();
    } catch (error) {
        showMessage(error.message);
    }
};

// Load profiles
async function loadProfiles() {
    try {
        const profiles = await apiCall('/api/profiles');
        const applications = await apiCall('/api/applications');
        
        const appliedProfileCodes = applications.map(app => app.profile_code);
        
        const tbody = document.getElementById('profiles-body');
        
        if (profiles.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" style="text-align: center;">No profiles available</td></tr>';
            return;
        }
        
        tbody.innerHTML = profiles.map(profile => {
            const hasApplied = appliedProfileCodes.includes(profile.profile_code);
            
            return `
                <tr>
                    <td>${profile.profile_code}</td>
                    <td>${profile.company_name}</td>
                    <td>${profile.designation}</td>
                    <td>
                        ${hasApplied 
                            ? '<span class="status-badge status-applied">Applied</span>' 
                            : `<button onclick="applyToProfile(${profile.profile_code})" class="btn btn-primary btn-small">Apply</button>`
                        }
                    </td>
                </tr>
            `;
        }).join('');
    } catch (error) {
        showMessage(error.message);
    }
}

// Apply to profile
window.applyToProfile = async function(profile_code) {
    try {
        await apiCall('/api/apply', {
            method: 'POST',
            body: JSON.stringify({ profile_code })
        });
        
        showMessage('Application submitted successfully!', 'success');
        loadProfiles();
        loadApplications();
    } catch (error) {
        showMessage(error.message);
    }
};

// Initialize
loadApplications();