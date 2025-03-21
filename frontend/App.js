import React, { useState, useEffect } from 'react';
import Dashboard from './Dashboard.js';
import TransactionTable from './TransactionTable.js';
import ModelTraining from './ModelTraining.js';
import api from './api.js';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [stats, setStats] = useState({
    totalTransactions: 0,
    fraudTransactions: 0,
    flaggedTransactions: 0,
    savedAmount: 0
  });
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        setIsLoading(true);
        const data = await api.getStats();
        setStats(data);
      } catch (error) {
        console.error('Error fetching stats:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchStats();
    const interval = setInterval(fetchStats, 60000); // Update every minute
    
    return () => clearInterval(interval);
  }, []);

  const renderContent = () => {
    switch(activeTab) {
      case 'dashboard':
        return <Dashboard stats={stats} isLoading={isLoading} />;
      case 'transactions':
        return <TransactionTable />;
      case 'model':
        return <ModelTraining />;
      default:
        return <Dashboard stats={stats} isLoading={isLoading} />;
    }
  };

  return (
    <div className="container-fluid">
      <div className="row">
        {/* Sidebar */}
        <div className="col-md-2 sidebar p-0">
          <div className="d-flex flex-column p-3">
            <h3 className="text-center mb-4">Fraud Detector</h3>
            <ul className="nav nav-pills flex-column mb-auto">
              <li className="nav-item">
                <a 
                  href="#" 
                  className={`nav-link ${activeTab === 'dashboard' ? 'active' : ''}`}
                  onClick={() => setActiveTab('dashboard')}
                >
                  <i className="fas fa-tachometer-alt me-2"></i>
                  Dashboard
                </a>
              </li>
              <li className="nav-item">
                <a 
                  href="#" 
                  className={`nav-link ${activeTab === 'transactions' ? 'active' : ''}`}
                  onClick={() => setActiveTab('transactions')}
                >
                  <i className="fas fa-exchange-alt me-2"></i>
                  Transactions
                </a>
              </li>
              <li className="nav-item">
                <a 
                  href="#" 
                  className={`nav-link ${activeTab === 'model' ? 'active' : ''}`}
                  onClick={() => setActiveTab('model')}
                >
                  <i className="fas fa-brain me-2"></i>
                  Model Training
                </a>
              </li>
            </ul>
          </div>
        </div>
        
        {/* Main Content */}
        <div className="col-md-10 main-content">
          {renderContent()}
        </div>
      </div>
    </div>
  );
}

export default App;
