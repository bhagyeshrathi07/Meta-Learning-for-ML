import { render, screen } from '@testing-library/react';
import App from './App';

jest.mock('axios', () => ({
  get: jest.fn(),
  post: jest.fn(),
}));

beforeEach(() => {
  localStorage.clear();
});

test('prompts an unauthenticated user for an API key', () => {
  render(<App />);

  expect(screen.getByText(/enter your api key to continue/i)).not.toBeNull();
  expect(screen.getByRole('button', { name: /authenticate/i }).disabled).toBe(true);
});
