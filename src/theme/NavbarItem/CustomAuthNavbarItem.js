import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Link from '@docusaurus/Link';

// Modal Component
const Modal = ({ show, onClose, children }) => {
  if (!show) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      backgroundColor: 'rgba(0,0,0,0.5)',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      zIndex: 1000
    }}>
      <div style={{
        backgroundColor: 'var(--docusaurus-sidebar-background-color)',
        padding: '20px',
        borderRadius: '8px',
        minWidth: '300px',
        boxShadow: '0 4px 8px rgba(0,0,0,0.1)',
        position: 'relative',
        color: 'var(--ifm-font-color-base)'
      }}>
        <button
          onClick={onClose}
          style={{
            position: 'absolute',
            top: '10px',
            right: '10px',
            background: 'none',
            border: 'none',
            fontSize: '1.2rem',
            cursor: 'pointer',
            color: 'var(--ifm-font-color-base)'
          }}
        >
          &times;
        </button>
        {children}
      </div>
    </div>
  );
};

// Login Form
const LoginForm = ({ onLoginSuccess, onClose }) => {
  const { login } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    const result = await login(username, password);
    if (result.success) {
      onLoginSuccess();
      onClose();
    } else {
      setError(result.error);
    }
  };

  return (
    <div>
      <h2>Log In</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label>Username:</label>
          <input type="text" value={username} onChange={(e) => setUsername(e.target.value)} required />
        </div>
        <div>
          <label>Password:</label>
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
        </div>
        {error && <p style={{ color: 'red' }}>{error}</p>}
        <button type="submit">Log In</button>
      </form>
    </div>
  );
};

// Signup Form
const SignupForm = ({ onSignupSuccess, onClose }) => {
  const { signup } = useAuth();
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    const result = await signup(username, email, password);
    if (result.success) {
      onSignupSuccess();
      onClose();
    } else {
      setError(result.error);
    }
  };

  return (
    <div>
      <h2>Sign Up</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label>Username:</label>
          <input type="text" value={username} onChange={(e) => setUsername(e.target.value)} required />
        </div>
        <div>
          <label>Email:</label>
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        </div>
        <div>
          <label>Password:</label>
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
        </div>
        {error && <p style={{ color: 'red' }}>{error}</p>}
        <button type="submit">Sign Up</button>
      </form>
    </div>
  );
};

const CustomAuthNavbarItem = ({ className }) => {
  const { isAuthenticated, user, logout } = useAuth();
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [showSignupModal, setShowSignupModal] = useState(false);

  const handleLogout = async () => {
    const result = await logout();
    alert(result.success ? 'Logged out successfully!' : 'Logout failed!');
  };

  return (
    <div className={className} style={{ display: 'flex', alignItems: 'center' }}>
      {isAuthenticated ? (
        <div className="dropdown dropdown--hoverable">
          <Link to="#" className="navbar__item navbar__link">
            {user?.name || 'Profile'}
          </Link>
          <ul className="dropdown__menu">
            <li><Link to="/profile" className="dropdown__link">My Profile</Link></li>
            <li><Link to="#" className="dropdown__link" onClick={handleLogout}>Log Out</Link></li>
          </ul>
        </div>
      ) : (
        <>
          <button className="button button--secondary button--sm" onClick={() => setShowSignupModal(true)} style={{ marginRight: '5px' }}>Sign Up</button>
          <button className="button button--primary button--sm" onClick={() => setShowLoginModal(true)}>Log In</button>
        </>
      )}

      <Modal show={showLoginModal} onClose={() => setShowLoginModal(false)}>
        <LoginForm onLoginSuccess={() => alert('Login Successful!')} onClose={() => setShowLoginModal(false)} />
      </Modal>

      <Modal show={showSignupModal} onClose={() => setShowSignupModal(false)}>
        <SignupForm onSignupSuccess={() => alert('Signup Successful!')} onClose={() => setShowSignupModal(false)} />
      </Modal>
    </div>
  );
};

export default CustomAuthNavbarItem;
