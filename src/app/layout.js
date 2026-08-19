import './globals.css';

export const metadata = {
  title: "Rizal's Resume Chatbot — Ask Me Anything",
  description:
    "An AI-powered RAG chatbot that answers questions about Rizal's professional experience, skills, and background. Built with brutalist design aesthetics.",
  keywords: ["resume", "chatbot", "RAG", "AI", "portfolio"],
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
