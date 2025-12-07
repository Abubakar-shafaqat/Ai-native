import React, { useState } from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';

function Signup() {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [message, setMessage] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log('Fake Sign Up Attempt:', { username, email, password });
    setMessage('Sign Up successful! (This is a fake, frontend-only submission)');
    // In a real app, this would send data to a backend.
    // For now, it just shows a success message.
    setUsername('');
    setEmail('');
    setPassword('');
  };

  return (
    <Layout title="Sign Up" description="Fake Sign Up page">
      <main style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '20px' }}>
        <div style={{
          backgroundColor: 'var(--ifm-background-color)',
          padding: '40px',
          borderRadius: '8px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
          maxWidth: '400px',
          width: '100%',
          color: 'var(--ifm-font-color-base)'
        }}>
          <h1 style={{ textAlign: 'center', marginBottom: '30px' }}>Sign Up</h1>
          <form onSubmit={handleSubmit}>
            <div style={{ marginBottom: '15px' }}>
              <label htmlFor="username" style={{ display: 'block', marginBottom: '5px' }}>Username:</label>
              <input
                type="text"
                id="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                style={{ width: '100%', padding: '10px', border: '1px solid var(--ifm-color-emphasis-300)', borderRadius: '4px' }}
              />
            </div>
            <div style={{ marginBottom: '15px' }}>
              <label htmlFor="email" style={{ display: 'block', marginBottom: '5px' }}>Email:</label>
              <input
                type="email"
                id="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                style={{ width: '100%', padding: '10px', border: '1px solid var(--ifm-color-emphasis-300)', borderRadius: '4px' }}
              />
            </div>
            <div style={{ marginBottom: '20px' }}>
              <label htmlFor="password" style={{ display: 'block', marginBottom: '5px' }}>Password:</label>
              <input
                type="password"
                id="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                style={{ width: '100%', padding: '10px', border: '1px solid var(--ifm-color-emphasis-300)', borderRadius: '4px' }}
              />
            </div>
            {message && <p style={{ color: 'green', textAlign: 'center', marginBottom: '15px' }}>{message}</p>}
            <button
              type="submit"
              style={{
                width: '100%',
                padding: '10px',
                backgroundColor: 'var(--ifm-color-primary)',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '1rem'
              }}
            >
              Sign Up
            </button>
          </form>
          <p style={{ textAlign: 'center', marginTop: '20px' }}>
            Already have an account? <Link to="/login">Log In</Link>
          </p>
        </div>
      </main>
    </Layout>
  );
}

export default Signup;
