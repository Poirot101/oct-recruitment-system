# OCS Recruitment System

A full-stack web application for managing recruitment processes with role-based access control for students, recruiters, and administrators.

## 🚀 Live Demo

- **Backend API**: [Your Vercel Backend URL]
- **Frontend**: [Your Vercel Frontend URL]

## 📋 Features

### Student Features
- View all available job/internship profiles
- Apply to profiles
- View application status
- Accept or reject selected offers
- View congratulatory message after accepting an offer

### Recruiter Features
- Create new job/internship profiles
- View all created profiles
- View applications to their profiles
- Change application status (Applied, Selected, Not Selected)

### Admin Features
- Full system access
- Create profiles for any recruiter
- View all users, profiles, and applications
- Modify any application status

## 🛠️ Technology Stack

- **Backend**: Node.js, Express.js
- **Frontend**: HTML, CSS, JavaScript (Vanilla)
- **Database**: PostgreSQL (via Supabase)
- **Authentication**: JWT tokens
- **Password Hashing**: MD5 (client-side)
- **Deployment**: Vercel

## 📁 Project Structure

```
ocs-recruitment-system/
├── backend/
│   ├── server.js
│   ├── package.json
│   ├── vercel.json
│   ├── api/
│   │   ├── auth.js
│   │   ├── users.js
│   │   ├── profiles.js
│   │   └── applications.js
│   ├── middleware/
│   │   └── auth.js
│   └── db/
│       └── supabase.js
├── frontend/
│   ├── index.html
│   ├── student.html
│   ├── recruiter.html
│   ├── admin.html
│   ├── package.json
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── login.js
│       ├── student.js
│       ├── recruiter.js
│       └── admin.js
├── .gitignore
└── README.md
```

## 🔧 Local Development Setup

### Prerequisites
- Node.js (v18+)
- npm or yarn
- Git
- Supabase account
- Vercel account

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Install dependencies:
```bash
npm install
```

3. Create `.env` file:
```env
PORT=3000
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
JWT_SECRET=your_secret_key
```

4. Run development server:
```bash
npm run dev
```

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create `.env` file:
```env
VITE_API_URL=http://localhost:3000
```

4. Run development server:
```bash
npm run dev
```

## 🗄️ Database Setup (Supabase)

1. Create a new Supabase project
2. Run the SQL script in the SQL Editor:

```sql
-- Create tables (see setup guide for full script)
CREATE TABLE users (...);
CREATE TABLE profile (...);
CREATE TABLE application (...);

-- Insert test data
INSERT INTO users VALUES (...);
INSERT INTO profile VALUES (...);
INSERT INTO application VALUES (...);
```

## 🚀 Deployment

### Deploy Backend to Vercel

```bash
cd backend
vercel login
vercel
```

Add environment variables in Vercel dashboard:
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `JWT_SECRET`

### Deploy Frontend to Vercel

```bash
cd frontend
vercel
```

Add environment variable in Vercel dashboard:
- `VITE_API_URL` (your backend URL)

## 🔐 Test Credentials

| Role | User ID | Password |
|------|---------|----------|
| Admin | admin | password |
| Student | student1 | spassword1 |
| Student | student2 | spassword2 |
| Student | student3 | spassword3 |
| Recruiter | recruiter1@techcorp.com | rpassword1 |
| Recruiter | recruiter2@dataworks.com | rpassword2 |

## 📡 API Endpoints

### Authentication
- `POST /api/login` - User login

### Users
- `GET /api/users/me` - Get current user
- `GET /api/users` - Get all users (admin only)

### Profiles
- `GET /api/profiles` - Get profiles
- `POST /api/create_profile` - Create profile

### Applications
- `GET /api/applications` - Get applications
- `POST /api/apply` - Apply to profile (student)
- `POST /api/application/change_status` - Change status (recruiter/admin)
- `POST /api/application/accept` - Accept offer (student)
- `POST /api/application/reject` - Reject offer (student)

## 🔒 Security Features

1. **Client-side password hashing**: Passwords are hashed using MD5 on the client before transmission
2. **JWT authentication**: All API requests require valid JWT token
3. **Role-based access control**: Different permissions for student, recruiter, and admin
4. **No sensitive data in frontend**: All database operations through secure backend API

## 📝 Assignment Requirements Completed

- ✅ Secure REST API with JWT authentication
- ✅ Browser-based client with no sensitive data exposure
- ✅ MD5 password hashing on client-side
- ✅ All data exchange through server API
- ✅ Role-based UI and access control
- ✅ Complete student application flow
- ✅ Recruiter profile management
- ✅ Admin full access capabilities
- ✅ Cloud hosting (Vercel)
- ✅ Cloud database (Supabase)

## 👨‍💻 Author

[Your Name] - IIT Delhi

## 📄 License

This project is created as part of OCS Technical Team Recruitment assignment.
