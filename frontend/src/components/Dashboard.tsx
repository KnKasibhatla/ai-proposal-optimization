import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
  Alert,
  Button
} from '@mui/material';
import {
  TrendingUp,
  Assessment,
  DataUsage,
  Speed
} from '@mui/icons-material';
import axios from 'axios';

interface DashboardProps {
  apiStatus: string;
}

interface Stats {
  total_proposals: number;
  win_rate: number;
  avg_score: number;
  last_updated: string;
}

const Dashboard: React.FC<DashboardProps> = ({ apiStatus }) => {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(false);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

  useEffect(() => {
    if (apiStatus === 'connected') {
      fetchStats();
    }
  }, [apiStatus]);

  const fetchStats = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/api/stats`);
      setStats(response.data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    } finally {
      setLoading(false);
    }
  };

  const StatCard = ({ title, value, icon, color = 'primary' }: any) => (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Box>
            <Typography color="textSecondary" gutterBottom variant="overline">
              {title}
            </Typography>
            <Typography variant="h4" component="div">
              {value}
            </Typography>
          </Box>
          <Box color={`${color}.main`}>
            {icon}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>
      
      <Box mb={3}>
        <Chip 
          label={`API Status: ${apiStatus}`} 
          color={apiStatus === 'connected' ? 'success' : 'error'}
          variant="outlined"
        />
      </Box>

      {apiStatus === 'disconnected' && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          Backend API is not available. Please start the backend server to see live data.
          <br />
          <Button 
            variant="outlined" 
            size="small" 
            sx={{ mt: 1 }}
            onClick={() => window.location.reload()}
          >
            Retry Connection
          </Button>
        </Alert>
      )}

      {loading && <LinearProgress sx={{ mb: 2 }} />}

      <Grid container spacing={3}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Total Proposals"
            value={stats?.total_proposals || '0'}
            icon={<Assessment fontSize="large" />}
            color="primary"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Win Rate"
            value={stats?.win_rate ? `${(stats.win_rate * 100).toFixed(1)}%` : '0%'}
            icon={<TrendingUp fontSize="large" />}
            color="success"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Average Score"
            value={stats?.avg_score ? stats.avg_score.toFixed(2) : '0.00'}
            icon={<Speed fontSize="large" />}
            color="info"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Data Status"
            value={stats ? 'Active' : 'No Data'}
            icon={<DataUsage fontSize="large" />}
            color={stats ? 'success' : 'warning'}
          />
        </Grid>
      </Grid>

      <Grid container spacing={3} sx={{ mt: 2 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Actions
              </Typography>
              <Box display="flex" flexDirection="column" gap={2}>
                <Button variant="outlined" href="/upload">
                  Upload New Data
                </Button>
                <Button variant="outlined" href="/analyze">
                  Analyze Proposals
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Information
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Last Updated: {stats?.last_updated || 'Never'}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Frontend: React + TypeScript
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Backend: Flask + Python
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
