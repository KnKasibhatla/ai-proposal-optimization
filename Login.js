// src/pages/Auth/Login.js
import React, { useState } from 'react';
import {
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Box,
  Link,
  Alert,
  CircularProgress,
  Divider,
} from '@mui/material';
import { Link as RouterLink, useNavigate } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { TrendingUp } from '@mui/icons-material';
import { useAuthStore } from '../../store/authStore';

const Login = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const { login } = useAuthStore();
  const navigate = useNavigate();

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm();

  const onSubmit = async (data) => {
    setLoading(true);
    setError('');

    try {
      const result = await login(data);
      
      if (result.success) {
        navigate('/dashboard');
      } else {
        setError(result.error);
      }
    } catch (err) {
      setError('An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container component="main" maxWidth="sm">
      <Box
        sx={{
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        <Paper elevation={3} sx={{ padding: 4, width: '100%' }}>
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <TrendingUp color="primary" sx={{ fontSize: 40, mr: 1 }} />
              <Typography component="h1" variant="h4" fontWeight="bold">
                AI Proposals
              </Typography>
            </Box>
            
            <Typography component="h2" variant="h5" sx={{ mb: 3 }}>
              Sign in to your account
            </Typography>

            {error && (
              <Alert severity="error" sx={{ width: '100%', mb: 2 }}>
                {error}
              </Alert>
            )}

            <Box component="form" onSubmit={handleSubmit(onSubmit)} sx={{ width: '100%' }}>
              <TextField
                margin="normal"
                required
                fullWidth
                label="Email Address"
                type="email"
                autoComplete="email"
                autoFocus
                {...register('email', {
                  required: 'Email is required',
                  pattern: {
                    value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                    message: 'Invalid email address',
                  },
                })}
                error={!!errors.email}
                helperText={errors.email?.message}
              />
              
              <TextField
                margin="normal"
                required
                fullWidth
                label="Password"
                type="password"
                autoComplete="current-password"
                {...register('password', {
                  required: 'Password is required',
                  minLength: {
                    value: 6,
                    message: 'Password must be at least 6 characters',
                  },
                })}
                error={!!errors.password}
                helperText={errors.password?.message}
              />

              <Button
                type="submit"
                fullWidth
                variant="contained"
                sx={{ mt: 3, mb: 2, py: 1.5 }}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : 'Sign In'}
              </Button>

              <Divider sx={{ my: 2 }} />

              <Box textAlign="center">
                <Link component={RouterLink} to="/register" variant="body2">
                  Don't have an account? Sign up
                </Link>
              </Box>
            </Box>
          </Box>
        </Paper>
      </Box>
    </Container>
  );
};

export default Login;

// src/pages/Auth/Register.js
import React, { useState } from 'react';
import {
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Box,
  Link,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Divider,
} from '@mui/material';
import { Link as RouterLink, useNavigate } from 'react-router-dom';
import { useForm, Controller } from 'react-hook-form';
import { TrendingUp } from '@mui/icons-material';
import { useAuthStore } from '../../store/authStore';

const Register = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const { register: registerUser } = useAuthStore();
  const navigate = useNavigate();

  const {
    register,
    handleSubmit,
    control,
    watch,
    formState: { errors },
  } = useForm();

  const selectedRole = watch('role');

  const onSubmit = async (data) => {
    setLoading(true);
    setError('');

    try {
      const result = await registerUser(data);
      
      if (result.success) {
        navigate('/dashboard');
      } else {
        setError(result.error);
      }
    } catch (err) {
      setError('An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container component="main" maxWidth="sm">
      <Box
        sx={{
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          py: 3,
        }}
      >
        <Paper elevation={3} sx={{ padding: 4, width: '100%' }}>
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <TrendingUp color="primary" sx={{ fontSize: 40, mr: 1 }} />
              <Typography component="h1" variant="h4" fontWeight="bold">
                AI Proposals
              </Typography>
            </Box>
            
            <Typography component="h2" variant="h5" sx={{ mb: 3 }}>
              Create your account
            </Typography>

            {error && (
              <Alert severity="error" sx={{ width: '100%', mb: 2 }}>
                {error}
              </Alert>
            )}

            <Box component="form" onSubmit={handleSubmit(onSubmit)} sx={{ width: '100%' }}>
              <TextField
                margin="normal"
                required
                fullWidth
                label="Full Name"
                autoComplete="name"
                autoFocus
                {...register('name', {
                  required: 'Name is required',
                  minLength: {
                    value: 2,
                    message: 'Name must be at least 2 characters',
                  },
                })}
                error={!!errors.name}
                helperText={errors.name?.message}
              />

              <TextField
                margin="normal"
                required
                fullWidth
                label="Email Address"
                type="email"
                autoComplete="email"
                {...register('email', {
                  required: 'Email is required',
                  pattern: {
                    value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                    message: 'Invalid email address',
                  },
                })}
                error={!!errors.email}
                helperText={errors.email?.message}
              />

              <Controller
                name="role"
                control={control}
                defaultValue=""
                rules={{ required: 'Role is required' }}
                render={({ field }) => (
                  <FormControl fullWidth margin="normal" error={!!errors.role}>
                    <InputLabel>I am a...</InputLabel>
                    <Select {...field} label="I am a...">
                      <MenuItem value="client">Client (Looking for services)</MenuItem>
                      <MenuItem value="provider">Provider (Offering services)</MenuItem>
                    </Select>
                    {errors.role && (
                      <Typography variant="caption" color="error" sx={{ mt: 1, ml: 2 }}>
                        {errors.role.message}
                      </Typography>
                    )}
                  </FormControl>
                )}
              />

              {selectedRole === 'client' && (
                <>
                  <TextField
                    margin="normal"
                    fullWidth
                    label="Company Name"
                    {...register('company_name')}
                  />
                  <TextField
                    margin="normal"
                    fullWidth
                    label="Industry"
                    {...register('industry')}
                  />
                </>
              )}

              {selectedRole === 'provider' && (
                <TextField
                  margin="normal"
                  fullWidth
                  label="Areas of Expertise (comma-separated)"
                  placeholder="e.g., Software Development, Web Design, Marketing"
                  {...register('industry_expertise')}
                />
              )}

              <TextField
                margin="normal"
                required
                fullWidth
                label="Password"
                type="password"
                autoComplete="new-password"
                {...register('password', {
                  required: 'Password is required',
                  minLength: {
                    value: 8,
                    message: 'Password must be at least 8 characters',
                  },
                  pattern: {
                    value: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/,
                    message: 'Password must contain at least one uppercase letter, one lowercase letter, and one number',
                  },
                })}
                error={!!errors.password}
                helperText={errors.password?.message}
              />

              <Button
                type="submit"
                fullWidth
                variant="contained"
                sx={{ mt: 3, mb: 2, py: 1.5 }}
                disabled={loading}
              >
                {loading ? <CircularProgress size={24} /> : 'Create Account'}
              </Button>

              <Divider sx={{ my: 2 }} />

              <Box textAlign="center">
                <Link component={RouterLink} to="/login" variant="body2">
                  Already have an account? Sign in
                </Link>
              </Box>
            </Box>
          </Box>
        </Paper>
      </Box>
    </Container>
  );
};

export default Register;

// src/pages/Dashboard/Dashboard.js
import React from 'react';
import {
  Grid,
  Typography,
  Box,
  Card,
  CardContent,
  Button,
  Alert,
} from '@mui/material';
import {
  Work,
  Assignment,
  TrendingUp,
  AccountBalance,
  Add,
  Visibility,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../../store/authStore';
import { useDashboardAnalytics } from '../../hooks/useAnalytics';
import StatCard from '../../components/Common/StatCard';
import LoadingSpinner from '../../components/Common/LoadingSpinner';
import RecentActivity from './RecentActivity';
import QuickActions from './QuickActions';
import PerformanceChart from './PerformanceChart';

const Dashboard = () => {
  const { user } = useAuthStore();
  const navigate = useNavigate();
  const { data: analytics, isLoading, error } = useDashboardAnalytics();

  if (isLoading) {
    return <LoadingSpinner message="Loading dashboard..." />;
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mt: 2 }}>
        Failed to load dashboard data. Please try again later.
      </Alert>
    );
  }

  const isClient = user?.role === 'client';
  const isProvider = user?.role === 'provider';

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" component="h1" fontWeight="bold">
          Dashboard
        </Typography>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => navigate(isClient ? '/projects/new' : '/projects')}
        >
          {isClient ? 'New Project' : 'Browse Projects'}
        </Button>
      </Box>

      {/* Key Metrics */}
      <Grid container spacing={3} mb={4}>
        {isClient && (
          <>
            <Grid item xs={12} sm={6} md={3}>
              <StatCard
                title="Active Projects"
                value={analytics?.total_projects || 0}
                change="+12%"
                changeType="positive"
                icon={<Work />}
                color="primary"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <StatCard
                title="Total Bids Received"
                value={analytics?.total_bids_received || 0}
                change="+8%"
                changeType="positive"
                icon={<Assignment />}
                color="success"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <StatCard
                title="Avg Bids per Project"
                value={analytics?.avg_bids_per_project || 0}
                change="+5%"
                changeType="positive"
                icon={<TrendingUp />}
                color="info"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <StatCard
                title="Completed Projects"
                value={analytics?.completed_projects || 0}
                change="+15%"
                changeType="positive"
                icon={<AccountBalance />}
                color="warning"
              />
            </Grid>
          </>
        )}

        {isProvider && (
          <>
            <Grid item xs={12} sm={6} md={3}>
              <StatCard
                title="Total Bids Submitted"
                value={analytics?.total_bids || 0}
                change="+10%"
                changeType="positive"
                icon={<Assignment />}
                color="primary"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <StatCard
                title="Win Rate"
                value={`${analytics?.win_rate || 0}%`}
                change="+3%"
                changeType="positive"
                icon={<TrendingUp />}
                color="success"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <StatCard
                title="Average Bid Value"
                value={`${(analytics?.avg_bid_price || 0).toLocaleString()}`}
                change="+7%"
                changeType="positive"
                icon={<AccountBalance />}
                color="info"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <StatCard
                title="Won Projects"
                value={analytics?.won_bids || 0}
                change="+20%"
                changeType="positive"
                icon={<Work />}
                color="warning"
              />
            </Grid>
          </>
        )}
      </Grid>

      {/* Main Content Grid */}
      <Grid container spacing={3}>
        {/* Performance Chart */}
        <Grid item xs={12} lg={8}>
          <PerformanceChart userRole={user?.role} />
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12} lg={4}>
          <QuickActions userRole={user?.role} />
        </Grid>

        {/* Recent Activity */}
        <Grid item xs={12}>
          <RecentActivity userRole={user?.role} />
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;

// src/pages/Dashboard/QuickActions.js
import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemButton,
  Divider,
} from '@mui/material';
import {
  Add,
  Search,
  Analytics,
  TrendingUp,
  Assignment,
  Visibility,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

const QuickActions = ({ userRole }) => {
  const navigate = useNavigate();

  const clientActions = [
    {
      icon: <Add color="primary" />,
      text: 'Create New Project',
      subtitle: 'Post a new project for bidding',
      action: () => navigate('/projects/new'),
    },
    {
      icon: <Visibility color="info" />,
      text: 'View All Projects',
      subtitle: 'Manage your existing projects',
      action: () => navigate('/projects'),
    },
    {
      icon: <Assignment color="success" />,
      text: 'Review Bids',
      subtitle: 'Check received proposals',
      action: () => navigate('/bids'),
    },
    {
      icon: <Analytics color="warning" />,
      text: 'Market Intelligence',
      subtitle: 'View market trends and insights',
      action: () => navigate('/analytics'),
    },
  ];

  const providerActions = [
    {
      icon: <Search color="primary" />,
      text: 'Browse Projects',
      subtitle: 'Find new opportunities',
      action: () => navigate('/projects'),
    },
    {
      icon: <TrendingUp color="info" />,
      text: 'Bid Optimization',
      subtitle: 'Get AI-powered bidding advice',
      action: () => navigate('/bids/optimize'),
    },
    {
      icon: <Assignment color="success" />,
      text: 'My Bids',
      subtitle: 'Track your submitted proposals',
      action: () => navigate('/bids'),
    },
    {
      icon: <Analytics color="warning" />,
      text: 'Performance Analytics',
      subtitle: 'Analyze your bidding performance',
      action: () => navigate('/analytics'),
    },
  ];

  const actions = userRole === 'client' ? clientActions : providerActions;

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" component="h2" gutterBottom>
          Quick Actions
        </Typography>
        <List>
          {actions.map((action, index) => (
            <React.Fragment key={index}>
              <ListItem disablePadding>
                <ListItemButton onClick={action.action}>
                  <ListItemIcon>{action.icon}</ListItemIcon>
                  <ListItemText
                    primary={action.text}
                    secondary={action.subtitle}
                  />
                </ListItemButton>
              </ListItem>
              {index < actions.length - 1 && <Divider />}
            </React.Fragment>
          ))}
        </List>
      </CardContent>
    </Card>
  );
};

export default QuickActions;

// src/pages/Dashboard/RecentActivity.js
import React from 'react';
import {
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  Avatar,
  Chip,
  Box,
  Button,
} from '@mui/material';
import {
  Work,
  Assignment,
  CheckCircle,
  Schedule,
  TrendingUp,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

const RecentActivity = ({ userRole }) => {
  const navigate = useNavigate();

  // Mock data - in real app, this would come from API
  const clientActivities = [
    {
      id: 1,
      type: 'new_project',
      title: 'Posted "E-commerce Website Development"',
      timestamp: '2 hours ago',
      status: 'active',
      icon: <Work />,
    },
    {
      id: 2,
      type: 'new_bid',
      title: 'Received 3 new bids for "Mobile App Design"',
      timestamp: '4 hours ago',
      status: 'pending',
      icon: <Assignment />,
    },
    {
      id: 3,
      type: 'project_completed',
      title: 'Completed "Brand Identity Package"',
      timestamp: '1 day ago',
      status: 'completed',
      icon: <CheckCircle />,
    },
  ];

  const providerActivities = [
    {
      id: 1,
      type: 'bid_submitted',
      title: 'Submitted bid for "React Dashboard Development"',
      timestamp: '1 hour ago',
      status: 'pending',
      icon: <Assignment />,
    },
    {
      id: 2,
      type: 'bid_won',
      title: 'Won "Logo Design Contest"',
      timestamp: '3 hours ago',
      status: 'won',
      icon: <CheckCircle />,
    },
    {
      id: 3,
      type: 'optimization',
      title: 'Used AI optimization for "Web Development" bid',
      timestamp: '6 hours ago',
      status: 'info',
      icon: <TrendingUp />,
    },
  ];

  const activities = userRole === 'client' ? clientActivities : providerActivities;

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
      case 'won':
        return 'success';
      case 'pending':
        return 'warning';
      case 'active':
        return 'primary';
      case 'info':
        return 'info';
      default:
        return 'default';
    }
  };

  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6" component="h2">
            Recent Activity
          </Typography>
          <Button size="small" onClick={() => navigate('/activity')}>
            View All
          </Button>
        </Box>
        
        <List>
          {activities.map((activity, index) => (
            <ListItem key={activity.id} divider={index < activities.length - 1}>
              <ListItemAvatar>
                <Avatar sx={{ bgcolor: `${getStatusColor(activity.status)}.light` }}>
                  {activity.icon}
                </Avatar>
              </ListItemAvatar>
              <ListItemText
                primary={activity.title}
                secondary={
                  <Box display="flex" alignItems="center" gap={1} mt={1}>
                    <Schedule fontSize="small" color="action" />
                    <Typography variant="caption" color="text.secondary">
                      {activity.timestamp}
                    </Typography>
                    <Chip
                      label={activity.status}
                      size="small"
                      color={getStatusColor(activity.status)}
                      variant="outlined"
                    />
                  </Box>
                }
              />
            </ListItem>
          ))}
        </List>
      </CardContent>
    </Card>
  );
};

export default RecentActivity;

// src/pages/Dashboard/PerformanceChart.js
import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  ToggleButton,
  ToggleButtonGroup,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';

const PerformanceChart = ({ userRole }) => {
  const [chartType, setChartType] = useState('line');
  const [timeRange, setTimeRange] = useState('30d');

  // Mock data - in real app, this would come from API
  const clientData = [
    { date: 'Jan', projects: 12, bids: 45, completed: 8 },
    { date: 'Feb', projects: 15, bids: 62, completed: 11 },
    { date: 'Mar', projects: 18, bids: 78, completed: 14 },
    { date: 'Apr', projects: 22, bids: 95, completed: 18 },
    { date: 'May', projects: 25, bids: 110, completed: 21 },
    { date: 'Jun', projects: 28, bids: 125, completed: 24 },
  ];

  const providerData = [
    { date: 'Jan', bids: 25, wins: 8, winRate: 32 },
    { date: 'Feb', bids: 30, wins: 12, winRate: 40 },
    { date: 'Mar', bids: 35, wins: 15, winRate: 43 },
    { date: 'Apr', bids: 28, wins: 14, winRate: 50 },
    { date: 'May', bids: 32, wins: 18, winRate: 56 },
    { date: 'Jun', bids: 38, wins: 22, winRate: 58 },
  ];

  const data = userRole === 'client' ? clientData : providerData;

  const handleChartTypeChange = (event, newType) => {
    if (newType !== null) {
      setChartType(newType);
    }
  };

  const handleTimeRangeChange = (event, newRange) => {
    if (newRange !== null) {
      setTimeRange(newRange);
    }
  };

  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Typography variant="h6" component="h2">
            Performance Overview
          </Typography>
          
          <Box display="flex" gap={2}>
            <ToggleButtonGroup
              value={timeRange}
              exclusive
              onChange={handleTimeRangeChange}
              size="small"
            >
              <ToggleButton value="7d">7D</ToggleButton>
              <ToggleButton value="30d">30D</ToggleButton>
              <ToggleButton value="90d">90D</ToggleButton>
            </ToggleButtonGroup>
            
            <ToggleButtonGroup
              value={chartType}
              exclusive
              onChange={handleChartTypeChange}
              size="small"
            >
              <ToggleButton value="line">Line</ToggleButton>
              <ToggleButton value="bar">Bar</ToggleButton>
            </ToggleButtonGroup>
          </Box>
        </Box>

        <Box height={300}>
          <ResponsiveContainer width="100%" height="100%">
            {chartType === 'line' ? (
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                {userRole === 'client' ? (
                  <>
                    <Line 
                      type="monotone" 
                      dataKey="projects" 
                      stroke="#1976d2" 
                      strokeWidth={2}
                      name="Projects Posted"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="bids" 
                      stroke="#2e7d32" 
                      strokeWidth={2}
                      name="Bids Received"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="completed" 
                      stroke="#ed6c02" 
                      strokeWidth={2}
                      name="Completed"
                    />
                  </>
                ) : (
                  <>
                    <Line 
                      type="monotone" 
                      dataKey="bids" 
                      stroke="#1976d2" 
                      strokeWidth={2}
                      name="Bids Submitted"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="wins" 
                      stroke="#2e7d32" 
                      strokeWidth={2}
                      name="Bids Won"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="winRate" 
                      stroke="#ed6c02" 
                      strokeWidth={2}
                      name="Win Rate %"
                      yAxisId="right"
                    />
                  </>
                )}
              </LineChart>
            ) : (
              <BarChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                {userRole === 'client' ? (
                  <>
                    <Bar dataKey="projects" fill="#1976d2" name="Projects Posted" />
                    <Bar dataKey="bids" fill="#2e7d32" name="Bids Received" />
                    <Bar dataKey="completed" fill="#ed6c02" name="Completed" />
                  </>
                ) : (
                  <>
                    <Bar dataKey="bids" fill="#1976d2" name="Bids Submitted" />
                    <Bar dataKey="wins" fill="#2e7d32" name="Bids Won" />
                  </>
                )}
              </BarChart>
            )}
          </ResponsiveContainer>
        </Box>
      </CardContent>
    </Card>
  );
};

export default PerformanceChart;