import React from 'react';
import { Settings as SettingsIcon } from 'lucide-react';

const Settings: React.FC = () => {
  return (
    <div className="p-6">
      <div className="max-w-4xl mx-auto">
        <div className="text-center py-16">
          <SettingsIcon className="w-20 h-20 text-gray-300 mx-auto mb-4" />
          <h1 className="text-2xl font-bold text-gray-900 mb-2">Settings</h1>
          <p className="text-gray-600 mb-8">
            Configure your platform preferences
          </p>
          <p className="text-sm text-gray-500">
            Settings page coming soon. Configure API keys, notifications, and trading preferences.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Settings;