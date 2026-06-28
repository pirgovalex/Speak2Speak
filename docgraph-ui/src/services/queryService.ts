// handles the /query endpoint - wires chat input to the mistral-7b rag pipeline
import apiClient from './apiClient';
import { QUERY_PATH } from '../config/env';

export interface QueryRequest {
  question: string;
}

export interface QuerySource {
  page: number;    // pdf page number cited by the llm
  excerpt: string; // snippet of source text from that page
}

export interface QueryResponse {
  answer: string;
  sources: QuerySource[]; // used by the pdf viewer panel to highlight pages
}

// bearer token is injected automatically by the apiClient request interceptor
export const askQuestion = async (question: string): Promise<QueryResponse> => {
  console.log('[docgraph api] sending query:', question);
  try {
    const response = await apiClient.post<QueryResponse>(QUERY_PATH, {
      question,
    } satisfies QueryRequest);
    console.log('[docgraph api] response received. sources:', response.data.sources?.length ?? 0);
    return response.data;
  } catch (queryError) {
    console.error('[docgraph api] query failed:', queryError);
    throw queryError; // re-throw so app.tsx can update the error signal
  }
};
