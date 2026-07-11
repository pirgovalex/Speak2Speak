import { API_BASE_URL } from '../config/env';
import axios from 'axios';

// using raw axios for multipart form data upload
export const uploadPdf = async (file: File): Promise<void> => {
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    await axios.post(`${API_BASE_URL}/upload_pdf`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
  } catch (err) {
    console.error('[docgraph api] upload failed:', err);
    throw err;
  }
};
