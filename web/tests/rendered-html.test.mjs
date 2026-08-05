import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the production experiment shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(
    html,
    /<title>Adaptive Agent Lab — Planning and Learning in Dynamic Warehouses<\/title>/i,
  );
  assert.match(html, /Planning × Learning/);
  assert.match(html, /Loading experiment artifact/);
  assert.match(html, /aria-live="polite"/);
  assert.doesNotMatch(html, /Your site is taking shape|react-loading-skeleton/);
});

test("ships a six-agent paired replay artifact and research controls", async () => {
  const [artifactText, page, css] = await Promise.all([
    readFile(new URL("../public/demo-data.json", import.meta.url), "utf8"),
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  const artifact = JSON.parse(artifactText);

  assert.equal(
    artifact.verificationStatus,
    "DEMO · NON-CONFIRMATORY · PAIRED TAPE",
  );
  assert.match(artifact.scenarioFingerprint, /^sha256:[0-9a-f]{64}$/);
  assert.deepEqual(
    artifact.agents.map((agent) => agent.id),
    ["planning", "replanning", "q-learning", "dyna-q", "dqn", "hybrid"],
  );
  assert.ok(artifact.agents.every((agent) => agent.trace.length > 0));
  assert.match(page, /fetch\("\.\/demo-data\.json"\)/);
  assert.match(page, /data\.verificationStatus/);
  assert.match(page, /Paired-episode replay/);
  assert.match(page, /Applied action/);
  assert.match(page, /return "expired"/);
  assert.match(page, /Interpretation notes/);
  assert.match(page, /type="range"/);
  assert.match(page, /selectAgent/);
  assert.match(css, /\.operations-grid/);
  assert.match(css, /\.warehouse-map/);
  assert.match(css, /--page: #f6f6f3/);
  assert.doesNotMatch(page, /Mission control|Replay online|Floor 07/);
  assert.match(css, /prefers-reduced-motion:\s*reduce/);
});
