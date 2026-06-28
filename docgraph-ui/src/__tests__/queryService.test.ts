// tests for the /query endpoint wrapper - apiClient is mocked so no real http is made
import { describe, it, expect, vi, beforeEach } from 'vitest';

// mock the shared axios instance before importing the service under test
vi.mock('../services/apiClient', () => ({
  default: {
    post: vi.fn(),
  },
}));

import apiClient from '../services/apiClient';
import { askQuestion } from '../services/queryService';

describe('askQuestion', () => {
  // reset call history before each case to avoid cross-test pollution
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('calls the correct endpoint with the question', async () => {
    (apiClient.post as ReturnType<typeof vi.fn>).mockResolvedValue({
      data: { answer: 'trapezius', sources: [] },
    });

    await askQuestion('what muscles are in the back?');

    expect(apiClient.post).toHaveBeenCalledWith('/query', {
      question: 'what muscles are in the back?',
    });
  });

  it('returns the answer and sources from the response', async () => {
    const mockResponse = {
      data: {
        answer: 'trapezius, latissimus dorsi',
        sources: [{ page: 42, excerpt: 'the trapezius muscle...' }],
      },
    };
    (apiClient.post as ReturnType<typeof vi.fn>).mockResolvedValue(mockResponse);

    const result = await askQuestion('back muscles?');

    expect(result.answer).toBe('trapezius, latissimus dorsi');
    expect(result.sources).toHaveLength(1);
    expect(result.sources[0].page).toBe(42);
  });

  it('re-throws errors from the api', async () => {
    (apiClient.post as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('network error'));

    await expect(askQuestion('test?')).rejects.toThrow('network error');
  });
});
