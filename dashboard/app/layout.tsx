import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Correx",
  description: "AI Correction OS — Monitor rules, turns, growth",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="manifest" href="/manifest.json" />
        <meta name="theme-color" content="#0891b2" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
      </head>
      <body className="text-zinc-100 min-h-screen" style={{ background: "var(--bg)" }}>{children}</body>
    </html>
  );
}
