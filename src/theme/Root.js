import React from 'react';
import Chatbot from '@site/src/components/Chatbot';
import { AuthProvider } from '@site/src/contexts/AuthContext'; // Import AuthProvider

// Default implementation, that you can customize
function Root({ children }) {
  return (
    <AuthProvider> {/* Wrap children with AuthProvider */}
      {children}
      <Chatbot />
    </AuthProvider>
  );
}

export default Root;
