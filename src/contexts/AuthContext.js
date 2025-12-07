import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext(null);

// Base URL for your FastAPI backend
const API_BASE_URL = 'http://127.0.0.1:8000'; // Make sure this matches your backend's address

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null); // Stores user info if logged in
  const [loading, setLoading] = useState(true); // To manage initial loading state

  useEffect(() => {
    const loadUserFromToken = async () => {
      const token = localStorage.getItem('authToken');
      if (token) {
        try {
          // Validate token and fetch user info from backend
          const response = await fetch(`${API_BASE_URL}/auth/me`, {
            method: 'GET',
            headers: {
              'Authorization': `Bearer ${token}`,
            },
          });
          if (response.ok) {
            const userData = await response.json();
            setUser({ ...userData, token });
          } else {
            console.error('Token validation failed:', response.status);
            localStorage.removeItem('authToken'); // Token is invalid, remove it
          }
        } catch (error) {
          console.error('Error validating token:', error);
          localStorage.removeItem('authToken');
        }
      }
      setLoading(false);
    };

    loadUserFromToken();
  }, []);

  const login = async (username, password) => {
    setLoading(true);
    try {
      const formBody = new URLSearchParams();
      formBody.append('username', username);
      formBody.append('password', password);

      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: formBody.toString(),
      });
      const data = await response.json();

      if (response.ok && data.access_token) {
        localStorage.setItem('authToken', data.access_token);
        // After successful login, fetch user info
        const userResponse = await fetch(`${API_BASE_URL}/auth/me`, {
          method: 'GET',
          headers: {
            'Authorization': `Bearer ${data.access_token}`,
          },
        });
        if (userResponse.ok) {
          const userData = await userResponse.json();
          setUser({ ...userData, token: data.access_token });
          return { success: true };
        } else {
          console.error('Failed to fetch user info after login:', userResponse.status);
          localStorage.removeItem('authToken');
          return { success: false, error: 'Failed to fetch user info' };
        }
      } else {
        return { success: false, error: data.detail || 'Login failed' };
      }
    } catch (error) {
      console.error('Login error:', error);
      return { success: false, error: 'Network error or server unavailable' };
    } finally {
      setLoading(false);
    }
  };

  const signup = async (username, email, password) => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/auth/signup`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password }),
      });
      const data = await response.json();

      if (response.ok) {
        // Assuming signup is successful, try to log in the user directly
        const loginResult = await login(username, password);
        if (loginResult.success) {
          return { success: true };
        } else {
          return { success: false, error: loginResult.error || 'Signup successful, but auto-login failed.' };
        }
      } else {
        return { success: false, error: data.detail || 'Signup failed' };
      }
    } catch (error) {
      console.error('Signup error:', error);
      return { success: false, error: 'Network error or server unavailable' };
    } finally {
      setLoading(false);
    }
  };

  const logout = async () => {
    setLoading(true);
    try {
      // For JWT, logout is primarily client-side (removing token)
      // If you had a backend /logout to invalidate tokens, you'd call it here.
      localStorage.removeItem('authToken');
      setUser(null);
      return { success: true };
    } catch (error) {
      console.error('Logout error:', error);
      return { success: false, error: 'Logout failed unexpectedly' };
    } finally {
      setLoading(false);
    }
  };

  const value = {
    user,
    isAuthenticated: !!user,
    loading,
    login,
    signup,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};