import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Grid,
  Alert,
  Paper,
  LinearProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import { Analytics, TrendingUp, Assessment } from '@mui/icons-material';
import axios from 'axios';

const ProposalAnalyzer: React.FC = () => {
  const [formData, setFormData] = useState({
    client_id: '',
    bid_amount: '',
    industry: '',
    proposal_type: ''
  });
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = event.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleAnalyze = async () => {
    setAnalyzing(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/analyze`, formData);
      setResults(response.data);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Analysis failed');
    } finally {
      setAnalyzing(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.7) return 'success';
    if (score >= 0.5) return 'warning';
    return 'error';
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Proposal Analyzer
      </Typography>
      
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Analyze New Proposal
          </Typography>
          <Typography variant="body2" color="textSecondary" paragraph>
            Enter proposal details to get AI-powered win probability analysis and recommendations.
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Client ID"
                name="client_id"
                value={formData.client_id}
                onChange={handleInputChange}
                placeholder="e.g., CLIENT-001"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Bid Amount"
                name="bid_amount"
                type="number"
                value={formData.bid_amount}
                onChange={handleInputChange}
                placeholder="e.g., 50000"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Industry"
                name="industry"
                value={formData.industry}
                onChange={handleInputChange}
                placeholder="e.g., Technology"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Proposal Type"
                name="proposal_type"
                value={formData.proposal_type}
                onChange={handleInputChange}
                placeholder="e.g., Software Development"
              />
            </Grid>
          </Grid>

          <Box sx={{ mt: 3 }}>
            <Button
              variant="contained"
              onClick={handleAnalyze}
              disabled={analyzing || !formData.client_id || !formData.bid_amount}
              startIcon={<Analytics />}
            >
              {analyzing ? 'Analyzing...' : 'Analyze Proposal'}
            </Button>
          </Box>

          {analyzing && <LinearProgress sx={{ mt: 2 }} />}
        </CardContent>
      </Card>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {results && (
        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <TrendingUp color="primary" sx={{ mr: 1 }} />
                  <Typography variant="h6">Win Probability</Typography>
                </Box>
                <Typography variant="h3" color="primary">
                  {(results.win_probability * 100).toFixed(1)}%
                </Typography>
                <Chip 
                  label={results.confidence_level || 'Medium'} 
                  color={getScoreColor(results.win_probability)}
                  size="small"
                  sx={{ mt: 1 }}
                />
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" mb={2}>
                  <Assessment color="secondary" sx={{ mr: 1 }} />
                  <Typography variant="h6">Risk Score</Typography>
                </Box>
                <Typography variant="h3" color="secondary">
                  {results.risk_score?.toFixed(2) || 'N/A'}
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Lower is better
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Recommended Action
                </Typography>
                <Typography variant="body1" color="primary">
                  {results.recommendation || 'Submit proposal'}
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {results.factors && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Key Factors Analysis
                  </Typography>
                  <TableContainer component={Paper} variant="outlined">
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell>Factor</TableCell>
                          <TableCell>Impact</TableCell>
                          <TableCell>Score</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {Object.entries(results.factors).map(([factor, data]: [string, any]) => (
                          <TableRow key={factor}>
                            <TableCell>{factor.replace('_', ' ').toUpperCase()}</TableCell>
                            <TableCell>
                              <Chip 
                                label={data.impact || 'Neutral'} 
                                color={data.impact === 'Positive' ? 'success' : data.impact === 'Negative' ? 'error' : 'default'}
                                size="small"
                              />
                            </TableCell>
                            <TableCell>{data.score?.toFixed(2) || 'N/A'}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              </Card>
            </Grid>
          )}

          {results.suggestions && results.suggestions.length > 0 && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Optimization Suggestions
                  </Typography>
                  {results.suggestions.map((suggestion: string, index: number) => (
                    <Alert key={index} severity="info" sx={{ mb: 1 }}>
                      {suggestion}
                    </Alert>
                  ))}
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      )}
    </Box>
  );
};

export default ProposalAnalyzer;
