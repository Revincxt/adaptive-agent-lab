import type { Metadata, Viewport } from "next";
import { headers } from "next/headers";
import "./globals.css";

const pagesBasePath = process.env.GITHUB_PAGES_BASE_PATH ?? "/";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host");
  const forwardedProtocol = requestHeaders.get("x-forwarded-proto");
  const protocol =
    forwardedProtocol ?? (host?.startsWith("localhost") || host?.startsWith("127.0.0.1") ? "http" : "https");
  const origin = host ? `${protocol}://${host}` : "https://revincxt.github.io";

  return {
    metadataBase: new URL(pagesBasePath, `${origin}/`),
    title: "Adaptive Agent Lab — Planning and Learning in Dynamic Warehouses",
    description:
      "An interactive, non-confirmatory research artifact for comparing planning and learning controllers in a dynamic warehouse maze.",
    applicationName: "Adaptive Agent Lab",
    keywords: [
      "reinforcement learning",
      "automated planning",
      "warehouse robotics",
      "Dyna-Q",
      "DQN",
      "replanning",
    ],
    authors: [{ name: "Revincxt", url: "https://github.com/Revincxt" }],
    alternates: { canonical: "./" },
    icons: { icon: "./favicon.svg" },
    openGraph: {
      type: "website",
      url: "./",
      siteName: "Adaptive Agent Lab",
      title: "Adaptive Agent Lab — Planning × Learning",
      description:
        "A paired-episode research replay for six controllers in a dynamic warehouse maze.",
      images: [
        {
          url: "./og.png",
          width: 1536,
          height: 1024,
          alt: "Academic graphical abstract showing a robot trajectory through a warehouse maze",
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title: "Adaptive Agent Lab — Planning × Learning",
      description:
        "A paired-episode research replay for planning and reinforcement learning.",
      images: ["./og.png"],
    },
  };
}

export const viewport: Viewport = {
  colorScheme: "light",
  themeColor: "#f6f6f3",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
