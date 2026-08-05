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
    title: "Adaptive Agent Lab — Planning × Learning",
    description:
      "Replay six planning, reinforcement-learning, and hybrid controllers inside a dynamic warehouse maze.",
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
      title: "Adaptive Agent Lab — Warehouse Autonomy Replay",
      description:
        "Inspect how six controllers navigate the same orders, rack maze, and dynamic aisle closures.",
      images: [
        {
          url: "./og.png",
          width: 1731,
          height: 909,
          alt: "Adaptive Agent Lab warehouse autonomy control room",
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title: "Adaptive Agent Lab — Warehouse Autonomy Replay",
      description:
        "Planning and reinforcement learning under identical warehouse disruptions.",
      images: ["./og.png"],
    },
  };
}

export const viewport: Viewport = {
  colorScheme: "dark",
  themeColor: "#070b0f",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
