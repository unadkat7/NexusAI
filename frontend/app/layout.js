import "./globals.css";

export const metadata = {
  title: "NexusAI",
  description: "Hybrid RAG research assistant",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
