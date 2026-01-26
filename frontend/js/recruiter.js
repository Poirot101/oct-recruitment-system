const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:3000';

// Check authentication
const token = localStorage.getItem('token');
const role = localStorage.getItem('role');
const userid = localStorage.getItem('userid');

if (!token || role !== 'recruiter') {
    window.location.href = '/';
}

// Global logout function
window.logout = function() {
    localStorage.clear();
    window.location.href = '/';
};

// Set user name
document.getElementById('user-name').textContent = userid;

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

// Create profile form handler
document.getElementById('create-profile-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const company_name = document.getElementById('company_name').value.trim();
    const designation = document.getElementById('designation').value.trim();
    
    try {
        await apiCall('/api/create_profile', {
            method: 'POST',
            body: JSON.stringify({ company_name, designation })
        });
        
        showMessage('Profile created successfully!', 'success');
        document.getElementById('create-profile-form').reset();
        loadProfiles();
    } catch (error) {
        showMessage(error.message);
    }
});

// Load profiles
async function loadProfiles() {
    try {
        const profiles = await apiCall('/api/profiles');
        const applications = await apiCall('/api/applications');
        
        // Count applications per profile
        const appCounts = {};
        applications.forEach(app => {
            appCounts[app.profile_code] = (appCounts[app.profile_code] || 0) + 1;
        });
        
        const tbody = document.getElementById('profiles-body');
        
        if (profiles.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" style="text-align: center;">No profiles created yet</td></tr>';
            return;
        }
        
        tbody.innerHTML = profiles.map(profile => `
            <tr>
                <td>${profile.profile_code}</td>
                <td>${profile.company_name}</td>
                <td>${profile.designation}</td>
                <td>${appCounts[profile.profile_code] || 0}</td>
            </tr>
        `).join('');
    } catch (error) {
        showMessage(error.message);
    }
}

// Load applications
async function loadApplications() {
    try {
        const applications = await apiCall('/api/applications');
        
        const tbody = document.getElementById('applications-body');
        
        if (applications.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" style="text-align: center;">No applications yet</td></tr>';
            return;
        }
        
        tbody.innerHTML = applications.map(app => `
            <tr>
                <td>${app.profile_code}</td>
                <td>${app.profile.company_name}</td>
                <td>${app.profile.designation}</td>
                <td>${app.entry_number}</td>
                <td><span class="status-badge status-${app.status.toLowerCase().replace(' ', '-')}">${app.status}</span></td>
                <td>
                    <select onchange="changeStatus(${app.profile_code}, '${app.entry_number}', this.value)" 
                            class="btn btn-small">
                        <option value="">Change Status</option>
                        <option value="Applied" ${app.status === 'Applied' ? 'selected' : ''}>Applied</option>
                        <option value="Selected" ${app.status === 'Selected' ? 'selected' : ''}>Selected</option>
                        <option value="Not Selected" ${app.status === 'Not Selected' ? 'selected' : ''}>Not Selected</option>
                    </select>
                </td>
            </tr>
        `).join('');
    } catch (error) {
        showMessage(error.message);
    }
}

// Change application status
window.changeStatus = async function(profile_code, entry_number, status) {
    if (!status) return;
    
    try {
        await apiCall('/api/application/change_status', {
            method: 'POST',
            body: JSON.stringify({ profile_code, entry_number, status })
        });
        
        showMessage('Status updated successfully!', 'success');
        loadApplications();
    } catch (error) {
        showMessage(error.message);
    }
};

// Initialize
loadProfiles();
loadApplications();