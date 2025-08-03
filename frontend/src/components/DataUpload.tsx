import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  LinearProgress,
  Alert,
  Paper,
  List,
  ListItem,
  ListItemText,
  Divider
} from '@mui/material';
import { CloudUpload, CheckCircle, Error } from '@mui/icons-material';
import axios from 'axios';

const DataUpload: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setUploadResult(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      setUploadResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.error || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Data Upload
      </Typography>
      
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Upload Proposal Data
          </Typography>
          <Typography variant="body2" color="textSecondary" paragraph>
            Upload a CSV file containing your proposal data. The file should include columns for 
            proposal details, pricing, and win/loss outcomes.
          </Typography>
          
          <Box sx={{ mb: 2 }}>
            <input
              accept=".csv,.xlsx,.xls"
              style={{ display: 'none' }}
              id="file-upload"
              type="file"
              onChange={handleFileSelect}
            />
            <label htmlFor="file-upload">
              <Button
                variant="outlined"
                component="span"
                startIcon={<CloudUpload />}
                sx={{ mr: 2 }}
              >
                Choose File
              </Button>
            </label>
            {file && (
              <Typography variant="body2" component="span">
                Selected: {file.name}
              </Typography>
            )}
          </Box>

          {file && (
            <Button
              variant="contained"
              onClick={handleUpload}
              disabled={uploading}
              startIcon={uploading ? undefined : <CloudUpload />}
            >
              {uploading ? 'Uploading...' : 'Upload Data'}
            </Button>
          )}

          {uploading && <LinearProgress sx={{ mt: 2 }} />}
        </CardContent>
      </Card>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }} icon={<Error />}>
          {error}
        </Alert>
      )}

      {uploadResult && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Box display="flex" alignItems="center" mb={2}>
              <CheckCircle color="success" sx={{ mr: 1 }} />
              <Typography variant="h6">Upload Successful</Typography>
            </Box>
            
            <Typography variant="body2" paragraph>
              File processed successfully. Here's a summary of your data:
            </Typography>
            
            <Paper variant="outlined" sx={{ p: 2 }}>
              <List dense>
                <ListItem>
                  <ListItemText 
                    primary="Total Records" 
                    secondary={uploadResult.total_records || 'N/A'} 
                  />
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemText 
                    primary="Columns Detected" 
                    secondary={uploadResult.columns?.join(', ') || 'N/A'} 
                  />
                </ListItem>
                <Divider />
                <ListItem>
                  <ListItemText 
                    primary="File Size" 
                    secondary={uploadResult.file_size || 'N/A'} 
                  />
                </ListItem>
              </List>
            </Paper>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Data Format Requirements
          </Typography>
          <Typography variant="body2" paragraph>
            Your CSV file should contain the following columns:
          </Typography>
          <List dense>
            <ListItem>
              <ListItemText 
                primary="proposal_id" 
                secondary="Unique identifier for each proposal" 
              />
            </ListItem>
            <ListItem>
              <ListItemText 
                primary="client_id" 
                secondary="Client identifier" 
              />
            </ListItem>
            <ListItem>
              <ListItemText 
                primary="bid_amount" 
                secondary="Proposed bid amount" 
              />
            </ListItem>
            <ListItem>
              <ListItemText 
                primary="win_loss" 
                secondary="Outcome: 'win' or 'loss'" 
              />
            </ListItem>
            <ListItem>
              <ListItemText 
                primary="industry" 
                secondary="Industry category (optional)" 
              />
            </ListItem>
          </List>
        </CardContent>
      </Card>
    </Box>
  );
};

export default DataUpload;
